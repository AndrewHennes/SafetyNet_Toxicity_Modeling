"""SafetyNet network and Lightning training steps, independent of Ray Tune."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    """One block of Linear, optional BatchNorm, activation, optional Dropout.

    The activation is passed in as an already-constructed module rather than
    built here, so a single instance is shared by every block in the trunk.
    That is safe for the stateless activations used in this script.
    """

    def __init__(
        self, in_dim, out_dim, activation, use_batch_norm=False, dropout_rate=0.0
    ):
        """Build the block.

        Args:
            in_dim: Input width.
            out_dim: Output width.
            activation: An instantiated activation module, applied after the
                optional batch norm.
            use_batch_norm: Whether to insert ``nn.BatchNorm1d`` after the
                linear layer.
            dropout_rate: Dropout probability. A value of 0 substitutes
                ``nn.Identity`` so no dropout module is created.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(in_dim, out_dim)
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        """Apply the block to a batch of shape ``(batch, in_dim)``."""
        x = self.linear(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MultiTaskFeedForward(pl.LightningModule):
    """Hard-parameter-sharing multi-task classifier over fixed fingerprints.

    A single trunk of ``MLPLayer`` blocks feeds one linear head that emits all
    ``output_dim`` logits at once, so every task shares the entire
    representation and differs only in its row of the head's weight matrix.

    The module emits logits, never probabilities. Callers that need
    probabilities apply the sigmoid themselves.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dim_size: int = 128,
        dim_shrinking_scale: float = 0.5,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        L2_weight_norm: float = 0.0,
        L1_weight_norm: float = 0.0,
        activation_function: str = "relu",
        dropout_rate: float = 0.0,
        use_residual: bool = False,
        use_batch_norm: bool = False,
        scheduler_step_size: int = 5,
        scheduler_gamma: float = 0.5,
        hidden_dims: tuple[int, ...] | None = None,
    ):
        """Store the hyperparameters and build the network.

        Args:
            input_dim: Width of the input fingerprint.
            output_dim: Number of tasks, and so the number of output logits.
            dim_size: Width of the first hidden layer.
            dim_shrinking_scale: Multiplier applied to the layer width at each
                successive layer. A value of 1.0 keeps the trunk isometric.
            num_layers: Number of hidden layers in the trunk.
            learning_rate: Initial Adam step size.
            batch_size: Recorded alongside the other hyperparameters.
                Batching itself is handled by ``MultiTaskDataModule``.
            L2_weight_norm: Coefficient passed to Adam as ``weight_decay``.
            L1_weight_norm: Coefficient on the L1 penalty added in
                ``training_step``.
            activation_function: One of ``relu``, ``tanh``, ``sigmoid`` or
                ``leaky_relu``. Any other value resolves to ReLU.
            dropout_rate: Dropout probability inside each trunk block.
            use_residual: Whether to add a skip connection around a trunk
                block. See ``_forward_shared_layers`` for the width condition
                under which one is applied.
            use_batch_norm: Whether each trunk block uses batch norm.
            scheduler_step_size: Interval, in optimizer steps, between
                learning rate decays.
            scheduler_gamma: Multiplicative learning rate decay applied at
                each interval.
        """
        super().__init__()

        # Records every argument above in self.hparams, which is what gets
        # written into the per-trial final_metrics.json.
        self.save_hyperparameters()

        self.hidden_dims = tuple(hidden_dims) if hidden_dims is not None else None
        if self.hidden_dims is not None and (
            not self.hidden_dims or any(d <= 0 for d in self.hidden_dims)
        ):
            raise ValueError("hidden_dims must contain positive layer widths.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_size = dim_size
        self.dim_shrinking_scale = dim_shrinking_scale
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.L2_weight_norm = L2_weight_norm
        self.L1_weight_norm = L1_weight_norm
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        # One shared activation instance for the whole trunk.
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        self.act_fn = activations.get(self.activation_function, nn.ReLU())

        self._build_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_shared_layers(self):
        """Build the trunk as a ``ModuleList`` of ``MLPLayer`` blocks.

        The first block maps ``input_dim`` to ``dim_size``. Each subsequent
        block narrows by ``dim_shrinking_scale``, floored at width 1.

        Returns:
            A ``nn.ModuleList`` of ``num_layers`` blocks.
        """
        layers = nn.ModuleList()
        in_dim = self.input_dim
        current_dim = self.dim_size

        widths = self.hidden_dims or self._geometric_widths()
        for current_dim in widths:
            block = MLPLayer(
                in_dim=in_dim,
                out_dim=current_dim,
                activation=self.act_fn,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
            layers.append(block)
            in_dim = current_dim

        return layers

    def _build_model(self):
        """Build the shared trunk and the single multi-output linear head."""
        self.shared_layers = self._build_shared_layers()
        self.head = nn.Linear(self._get_final_dim(), self.output_dim)

    def _geometric_widths(self):
        """Reproduce the width schedule stored in existing checkpoints."""
        widths = [self.dim_size]
        for _ in range(1, self.num_layers):
            widths.append(max(1, int(widths[-1] * self.dim_shrinking_scale)))
        return tuple(widths)

    def _get_final_dim(self):
        """Return the final width for explicit or geometric hidden layers."""
        return (self.hidden_dims or self._geometric_widths())[-1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """Map a batch of fingerprints to one logit per task.

        Args:
            x: Float tensor of shape ``(batch, input_dim)``.

        Returns:
            Logits of shape ``(batch, output_dim)``.
        """
        return self.head(self._forward_shared_layers(x))

    def _forward_shared_layers(self, x):
        """Run the trunk, optionally adding a skip connection per block.

        A skip connection is added around a block when its output width
        matches its input width, which is the case when
        ``dim_shrinking_scale`` is 1.0.

        Args:
            x: Float tensor of shape ``(batch, input_dim)``.

        Returns:
            Tensor of shape ``(batch, final_dim)``.
        """
        out = x
        for layer in self.shared_layers:
            new_out = layer(out)
            if self.use_residual and (new_out.shape == out.shape):
                out = out + new_out
            else:
                out = new_out
        return out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(self, preds, y):
        """Masked, equally weighted binary cross entropy across tasks.

        Samples whose label for a task is missing are dropped from that
        task's term only, so a compound measured against one endpoint still
        contributes to that endpoint. Every task carries weight 1, so the
        batch loss is the unweighted mean over the tasks that have at least
        one labelled sample present. That makes this independent of the
        number of tasks.

        Args:
            preds: Logits of shape ``(batch, output_dim)``.
            y: Targets of shape ``(batch, output_dim)``, with missing labels
                encoded below ``LABEL_PRESENT_THRESHOLD``.

        Returns:
            A scalar loss tensor.
        """
        device = preds.device
        total_loss = torch.tensor(0.0, device=device)
        num_scored_tasks = 0

        for t in range(self.output_dim):
            # Keep only the samples that carry an observed label for task t.
            mask = y[:, t] >= 0
            if mask.any():
                preds_t = preds[mask, t]
                y_t = y[mask, t]

                loss_t = F.binary_cross_entropy_with_logits(
                    preds_t,
                    y_t,
                    reduction="mean",
                )

                total_loss = total_loss + loss_t
                num_scored_tasks += 1

        if num_scored_tasks > 0:
            total_loss = total_loss / num_scored_tasks
        else:
            # No observed labels anywhere in this batch. Return a scalar
            # carrying requires_grad so the training step stays well formed.
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """Compute the training loss for one batch and log it.

        Args:
            batch: Dict with keys ``x`` and ``y`` from ``MultiTaskDataset``.
            batch_idx: Index of the batch within the epoch. Unused.

        Returns:
            The scalar loss Lightning backpropagates.
        """
        x, y = batch["x"], batch["y"]
        preds = self(x)
        loss = self._compute_loss(preds, y)

        # L1 is applied here as an explicit penalty summed over all
        # parameters. L2 goes through the optimizer's weight_decay in
        # configure_optimizers instead.
        if self.L1_weight_norm > 0:
            l1_penalty = sum(param.abs().sum() for param in self.parameters())
            loss = loss + self.L1_weight_norm * l1_penalty

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute and log the validation loss for one batch.

        This loss is recorded as a diagnostic. Selection runs on
        ``SELECTION_METRIC``, which ``MultiTaskEvaluationCallback`` computes
        once per epoch over the whole validation split.

        Args:
            batch: Dict with keys ``x`` and ``y``.
            batch_idx: Index of the batch within the epoch. Unused.

        Returns:
            The scalar validation loss.
        """
        x, y = batch["x"], batch["y"]
        preds = self(x)
        val_loss = self._compute_loss(preds, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Compute and log the test loss for one batch.

        Used when ``trainer.test`` is run on the selected configuration. The
        hyperparameter search does not call it.

        Args:
            batch: Dict with keys ``x`` and ``y``.
            batch_idx: Index of the batch within the epoch. Unused.

        Returns:
            The scalar test loss.
        """
        x, y = batch["x"], batch["y"]
        test_loss = self._compute_loss(self(x), y)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        """Build the Adam optimizer and its StepLR schedule.

        L2 regularization is applied through Adam's ``weight_decay``, set
        from ``L2_weight_norm``. The schedule is applied on an interval of
        ``step``, so ``scheduler_step_size`` counts optimizer steps.

        Returns:
            A ``([optimizer], [scheduler_config])`` pair.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.L2_weight_norm
        )
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
