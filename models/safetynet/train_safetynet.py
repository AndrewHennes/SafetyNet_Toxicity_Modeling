#!/usr/bin/env python3
"""Multi-task cytotoxicity classifier with a Ray Tune hyperparameter search.

Trains a hard-parameter-sharing feed-forward network on precomputed molecular
fingerprints to predict several binarized cytotoxicity endpoints at once, and
searches its hyperparameters with Optuna through Ray Tune.

Every task is weighted equally in the loss. Missing labels are encoded as -1
and are skipped on a per-task basis, so the training data does not need to be
fully overlapping across endpoints.

Model selection is unified on ``val_auprc_avg``, the mean validation average
precision across tasks. Optuna ranks trials on it, early stopping watches it,
and the retained checkpoint is the epoch that maximized it. The test split is
held out of the search entirely and is scored once, afterwards, on the single
selected configuration.

Dataset and output paths are configured at the top of ``main``.
"""

if __package__:
    from .model import MultiTaskFeedForward
else:
    from model import MultiTaskFeedForward

import ast
import json
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.optuna import OptunaSearch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# The quantity every stage of model selection agrees on: the Optuna search
# ranks trials by it, early stopping watches it, and the checkpoint that is
# kept is the epoch that maximized it. Changing it here changes all three.
SELECTION_METRIC = "val_auprc_avg"
SELECTION_MODE = "max"

# Unmeasured endpoints are encoded as -1 in the target columns. Any value at
# or above this threshold is treated as an observed label.
LABEL_PRESENT_THRESHOLD = 0.0


# ============================================================================
# 1) Read Data
# ============================================================================

def read_data(
    file_path: str,
    rep_col_name: str,
    target_col_names: list,
) -> pd.DataFrame:
    """Load a tab-separated dataset and keep only the columns needed here.

    Args:
        file_path: Absolute path to a tab-separated file.
        rep_col_name: Name of the column holding the molecular representation.
            It is parsed later by ``MultiTaskDataset``, which accepts either a
            Python list literal stored as a string or an already-parsed
            sequence.
        target_col_names: Names of the binarized target columns, in the order
            the model's output units should follow.

    Returns:
        A copy of the frame restricted to the representation column followed
        by the target columns.
    """
    df = pd.read_csv(file_path, sep='\t')
    print("Columns in the DataFrame are:", df.columns.tolist())
    columns = [rep_col_name] + target_col_names
    df = df[columns].copy()
    return df


# ============================================================================
# 2) Split Data
# ============================================================================

def train_val_test_split(
    df: pd.DataFrame,
    split_fracs: tuple = (0.85, 0.1, 0.05),
    random_state: int = 42,
):
    """Split a frame into train, validation and test partitions.

    Args:
        df: Frame to split.
        split_fracs: Train, validation and test fractions. Must sum to 1.
        random_state: Seed passed to both underlying splits, so the same
            partition is reproduced across trials and runs.

    Returns:
        A ``(df_train, df_val, df_test)`` tuple.
    """
    assert len(split_fracs) == 3, "split_fracs must have length 3."
    assert abs(sum(split_fracs) - 1.0) < 1e-7, "split_fracs must sum to 1."
    train_frac, val_frac, test_frac = split_fracs

    df_train, df_temp = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        random_state=random_state,
    )

    # df_temp holds validation and test together, so the requested validation
    # fraction is renormalized against that remainder before the second
    # split.
    relative_val_frac = val_frac / (val_frac + test_frac)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - relative_val_frac),
        random_state=random_state,
    )
    return df_train, df_val, df_test


# ============================================================================
# 3) DataModule
# ============================================================================

class MultiTaskDataset(Dataset):
    """In-memory dataset of fingerprints and their multi-task labels.

    The whole split is parsed and stacked into a dense array once, at
    construction, and then indexed directly.
    """

    def __init__(self, df, rep_col_name, target_col_names):
        """Parse the representation column and stack it into a dense array.

        Args:
            df: Frame containing the representation and target columns.
            rep_col_name: Name of the representation column. Entries may be
                strings holding a Python list literal, which is the form the
                minimol fingerprints are stored in, or already-parsed
                sequences.
            target_col_names: Names of the target columns, in output order.
        """
        super().__init__()
        parsed_features = []
        for item in df[rep_col_name]:
            if isinstance(item, str):
                arr = np.array(ast.literal_eval(item), dtype=np.float32)
            else:
                arr = np.array(item, dtype=np.float32)
            parsed_features.append(arr)

        # np.stack requires every fingerprint to have the same length, so a
        # ragged representation column raises here rather than producing an
        # object array.
        self.features = np.stack(parsed_features, axis=0)
        self.targets = df[target_col_names].values.astype(np.float32)

    def __len__(self):
        """Return the number of molecules in this split."""
        return len(self.features)

    def __getitem__(self, idx):
        """Return one molecule as a dict of its features and its labels."""
        return {'x': self.features[idx], 'y': self.targets[idx]}


class MultiTaskDataModule(pl.LightningDataModule):
    """Lightning data module wrapping the train, validation and test splits."""

    def __init__(self, df_train, df_val, df_test,
                 rep_col_name, target_col_names,
                 batch_size=32):
        """Store the frames and loader settings without building datasets yet.

        Args:
            df_train: Training split.
            df_val: Validation split, the source of the selection metric.
            df_test: Test split. Not scored during the hyperparameter search.
            rep_col_name: Name of the representation column in every frame.
            target_col_names: Names of the target columns, in output order.
            batch_size: Batch size used by all of the dataloaders.
        """
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.rep_col_name = rep_col_name
        self.target_col_names = target_col_names
        self.batch_size = batch_size

    def setup(self, stage=None):
        """Parse every frame into a ``MultiTaskDataset``.

        Args:
            stage: Lightning stage hint. All splits are built at once.
        """
        self.train_dataset = MultiTaskDataset(self.df_train,
                                              self.rep_col_name,
                                              self.target_col_names)
        self.val_dataset = MultiTaskDataset(self.df_val,
                                            self.rep_col_name,
                                            self.target_col_names)
        self.test_dataset = MultiTaskDataset(self.df_test,
                                             self.rep_col_name,
                                             self.target_col_names)

    def train_dataloader(self):
        """Return the shuffled training loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the unshuffled validation loader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return the unshuffled test loader.

        Used when ``trainer.test`` is run on the selected configuration. The
        hyperparameter search does not call it.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# ============================================================================
# 4) Model
# ============================================================================





# ============================================================================
# 5) MultiTaskEvaluationCallback
# ============================================================================

def compute_auroc_auprc(model, dataloader, device):
    """Score a model over a whole loader, one AUROC and AUPRC per task.

    Samples with a missing label for a task are excluded from that task's
    score. A task with no labelled samples, or with only one class present,
    yields NaN. The model is placed in eval mode for the duration of the
    pass.

    Args:
        model: A ``MultiTaskFeedForward``, which emits logits.
        dataloader: Loader yielding dicts with keys ``x`` and ``y``.
        device: Device the batches are moved to.

    Returns:
        Dict with ``auroc_per_task`` and ``auprc_per_task``, each a list of
        floats in task order.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["x"], batch["y"]
            x = x.to(device)
            y = y.to(device)
            # The model emits logits, so map to probabilities here. Both
            # metrics are rank based, so this leaves either score unchanged
            # and keeps the stored predictions interpretable.
            preds = torch.sigmoid(model(x))
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    num_tasks = all_targets.shape[1]
    aurocs = []
    auprcs = []
    for task_idx in range(num_tasks):
        valid_mask = (all_targets[:, task_idx] >= LABEL_PRESENT_THRESHOLD)
        y_true = all_targets[valid_mask, task_idx]
        y_pred = all_preds[valid_mask, task_idx]
        if len(y_true) == 0:
            auroc_val, auprc_val = float('nan'), float('nan')
        else:
            try:
                auroc_val = roc_auc_score(y_true, y_pred)
                auprc_val = average_precision_score(y_true, y_pred)
            except ValueError:
                # Raised when only one class is present among the labelled
                # samples for this task, which neither metric is defined for.
                auroc_val = float('nan')
                auprc_val = float('nan')
        aurocs.append(auroc_val)
        auprcs.append(auprc_val)

    print(f"auROCs are {aurocs}")
    print(f"auPRCs are {auprcs}")

    return {
        "auroc_per_task": [float(a) for a in aurocs],
        "auprc_per_task": [float(p) for p in auprcs],
    }


class MultiTaskEvaluationCallback(Callback):
    """Scores the splits each epoch and publishes the selection metric.

    This computes ``SELECTION_METRIC``, which the early stopping,
    checkpointing and Ray Tune reporting callbacks consume.
    """

    def __init__(self, save_filename="final_metrics.json"):
        """Set up the per-epoch metric accumulator.

        Args:
            save_filename: Name of the JSON written at the end of the fit,
                relative to the working directory, which under Ray Tune is
                the trial directory.
        """
        super().__init__()
        self.save_filename = save_filename
        self.results_per_epoch = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Score the splits, log the selection metric, record the epoch.

        Every metric is published through ``pl_module.log`` so that it enters
        Lightning's results machinery, where early stopping, checkpointing
        and ``TuneReportCallback`` can all read the same value.

        Args:
            trainer: The active trainer, used for the datamodule and epoch.
            pl_module: The model being trained.
        """
        current_epoch = trainer.current_epoch
        epoch_metrics = {"epoch": current_epoch}
        device = pl_module.device

        def log_per_output(prefix, values):
            """Publish one metric per task under ``<prefix>_output_<i>``."""
            if values is None:
                return
            for i, val in enumerate(values):
                pl_module.log(f"{prefix}_output_{i}", float(val))

        # Training-split scores are recorded so that the gap between train
        # and validation performance is visible in the metrics log.
        train_loader = trainer.datamodule.train_dataloader()
        train_stats = compute_auroc_auprc(pl_module, train_loader, device)
        epoch_metrics["train_auroc"] = train_stats["auroc_per_task"]
        epoch_metrics["train_auprc"] = train_stats["auprc_per_task"]

        val_loader = trainer.datamodule.val_dataloader()
        val_stats = compute_auroc_auprc(pl_module, val_loader, device)
        epoch_metrics["val_auroc"] = val_stats["auroc_per_task"]
        epoch_metrics["val_auprc"] = val_stats["auprc_per_task"]

        # The test split is not scored here. It is held out of the
        # hyperparameter search and evaluated once, afterwards, on the
        # single selected configuration.

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            epoch_metrics["val_loss"] = float(val_loss.cpu().numpy())

        # Publish the selection metric. np.nanmean returns a value whenever
        # at least one task scored, and NaN only if every task was NaN.
        val_auprc_values = epoch_metrics["val_auprc"]
        if val_auprc_values is not None:
            val_auprc_avg = float(np.nanmean(val_auprc_values))
            epoch_metrics[SELECTION_METRIC] = val_auprc_avg
            pl_module.log(SELECTION_METRIC, val_auprc_avg, prog_bar=True)

        print(f"[Epoch {current_epoch}] metrics: {epoch_metrics}")
        self.results_per_epoch.append(epoch_metrics)

        log_per_output("train_auroc", epoch_metrics["train_auroc"])
        log_per_output("train_auprc", epoch_metrics["train_auprc"])
        log_per_output("val_auroc", epoch_metrics["val_auroc"])
        log_per_output("val_auprc", epoch_metrics["val_auprc"])

    def on_fit_end(self, trainer, pl_module):
        """Write the full epoch history to JSON and report to Ray Tune.

        Args:
            trainer: The finished trainer. Unused.
            pl_module: The trained model, read for its hyperparameters.
        """
        if not self.results_per_epoch:
            print("No epochs recorded; skipping final metrics reporting.")
            return

        final_epoch_metrics = self.results_per_epoch[-1]
        hparams = dict(pl_module.hparams)
        final_dict = {
            "hyperparameters": hparams,
            "epoch_metrics": self.results_per_epoch
        }

        save_path = os.path.join(os.getcwd(), self.save_filename)
        with open(save_path, "w") as f:
            json.dump(final_dict, f, indent=4)
        print(
            f"\nSaved final results (hyperparams + metrics) "
            f"to {save_path}\n")

        final_report_dict = {}

        def store_per_output_columns(prefix, values):
            """Flatten a per-task list into ``<prefix>_output_<i>`` keys."""
            if values is None:
                return
            for i, val in enumerate(values):
                name = f"{prefix}_output_{i}"
                final_report_dict[name] = val

        store_per_output_columns(
            "train_auroc", final_epoch_metrics["train_auroc"])
        store_per_output_columns(
            "train_auprc", final_epoch_metrics["train_auprc"])
        store_per_output_columns(
            "val_auroc", final_epoch_metrics["val_auroc"])
        store_per_output_columns(
            "val_auprc", final_epoch_metrics["val_auprc"])

        final_report_dict["val_loss"] = final_epoch_metrics.get(
            "val_loss", None)
        final_report_dict[SELECTION_METRIC] = final_epoch_metrics.get(
            SELECTION_METRIC, None)

        tune.report(metrics=final_report_dict)


# ============================================================================
# 6) train_with_tune + main()
# ============================================================================

def train_with_tune(config,
                    df_train,
                    df_val,
                    df_test,
                    rep_col,
                    target_cols,
                    max_epochs=10):
    """Train one model for a single Ray Tune trial.

    Ray calls this once per sampled hyperparameter configuration. The split
    frames are bound ahead of time with ``tune.with_parameters`` so they are
    placed in the object store once rather than serialized per trial.

    Args:
        config: One sampled point from the search space in ``main``.
        df_train: Training split.
        df_val: Validation split.
        df_test: Test split. Passed through to the data module and not scored
            during the search.
        rep_col: Name of the representation column.
        target_cols: Names of the target columns, in output order.
        max_epochs: Ceiling on epochs. Early stopping normally fires first.
    """
    data_module = MultiTaskDataModule(
        df_train, df_val, df_test,
        rep_col_name=rep_col,
        target_col_names=target_cols,
        batch_size=config["batch_size"]
    )
    data_module.setup()

    # Read the fingerprint width off the parsed data rather than hard-coding
    # it, so a change of representation needs no edit here.
    input_dim = data_module.train_dataset.features.shape[1]

    model = MultiTaskFeedForward(
        input_dim=input_dim,
        output_dim=len(target_cols),
        dim_size=config["dim_size"],
        dim_shrinking_scale=config["dim_shrinking_scale"],
        num_layers=config["num_layers"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        L2_weight_norm=config["L2_weight_norm"],
        L1_weight_norm=config["L1_weight_norm"],
        activation_function=config["activation_function"],
        dropout_rate=config["dropout_rate"],
        use_residual=config["use_residual"],
        use_batch_norm=config["use_batch_norm"],
        scheduler_step_size=config["scheduler_step_size"],
        scheduler_gamma=config["scheduler_gamma"],
    )

    # Built from len(target_cols) rather than hard-coded to three outputs, so
    # this keeps working if the number of tasks changes.
    reported_metrics = {
        "train_loss": "train_loss",
        "val_loss": "val_loss",
        SELECTION_METRIC: SELECTION_METRIC,
    }
    for task_idx in range(len(target_cols)):
        key = f"val_auprc_output_{task_idx}"
        reported_metrics[key] = key

    tune_callback = TuneReportCallback(
        metrics=reported_metrics,
        on="validation_epoch_end",
    )

    # eval_callback is listed first because it produces SELECTION_METRIC,
    # which the three callbacks after it consume.
    eval_callback = MultiTaskEvaluationCallback(
        save_filename="final_metrics.json")

    early_stop_callback = EarlyStopping(
        monitor=SELECTION_METRIC,
        patience=config["patience"],
        mode=SELECTION_MODE,
        verbose=True
    )

    model_ckpt_callback = ModelCheckpoint(
        dirpath=config.get("model_file_path", "./checkpoints"),
        monitor=SELECTION_METRIC,
        mode=SELECTION_MODE,
        save_top_k=1,
        filename="best-{epoch}-{" + SELECTION_METRIC + ":.3f}",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            eval_callback,
            tune_callback,
            model_ckpt_callback,
            early_stop_callback
        ],
        enable_checkpointing=True,
        enable_progress_bar=False,
        default_root_dir=config.get("model_file_path", None),
    )

    trainer.fit(model, data_module)


class AppendMetricsCallback(tune.Callback):
    """Appends each finished trial's final result to a running TSV.

    Lets the sweep be followed as it builds up, rather than waiting for Ray's
    own summary at the end.
    """

    def __init__(self, csv_path):
        """Record where to append.

        Args:
            csv_path: Absolute path of the TSV to append to. It is opened in
                append mode, so results from an earlier run are preserved.
        """
        self.csv_path = csv_path
        self._header_written = False

    def on_trial_complete(self, iteration, trials, trial, **info):
        """Append one row holding the finished trial's final metrics.

        Args:
            iteration: Ray's tuning iteration counter. Unused.
            trials: All trials known to the tuner. Unused.
            trial: The trial that just finished.
            **info: Additional Ray callback arguments. Unused.
        """
        last_result = trial.last_result.copy()
        row_df = pd.DataFrame([last_result])

        with open(self.csv_path, "a") as f:
            row_df.to_csv(
                f,
                sep="\t",
                index=False,
                header=not self._header_written,
            )
        self._header_written = True


def main():
    """Configure the data, the search space, and run the Ray Tune sweep."""
    # ------------------------------------------------------------------
    NUM_TRIALS = 1000

    # ------------------------------------------------------------------
    # Dataset used to generate the train, validation and test splits, plus
    # the name of the column holding the molecular representation. This has
    # been run with minimol fingerprints stored as Python list literals in
    # the "Minimol_Fingerprint" column; other formats parse as long as every
    # row has the same length.
    # ------------------------------------------------------------------
    file_path = (
        "/path/to/my/fav/dataset.tsv"
    )

    rep_col_name = "Minimol_Fingerprint"

    # Column names for the different targets, in the order the model's
    # output units should follow.
    target_col_names = [
        "HepG2_10uM_Binarized",
        "HSkMC_10uM_Binarized",
        "IMR90_10uM_Binarized",
    ]

    # The tuple is the train, validation and test fractions, in that order.
    df = read_data(file_path, rep_col_name, target_col_names)
    df_train, df_val, df_test = train_val_test_split(df, (0.80, 0.10, 0.10))

    if not ray.is_initialized():
        ray.init()

    # ------------------------------------------------------------------
    # The hyperparameter search space. Entries given a single-element
    # tune.choice are effectively pinned: they still travel through the
    # config dict but contribute no dimension to the search.
    # ------------------------------------------------------------------
    search_space = {
        # Width of the first hidden layer. Ray's upper bound is exclusive,
        # so 4097 is what makes 4096 reachable.
        "dim_size": tune.lograndint(128, 4097),
        # Multiplier applied to the layer width at each successive layer,
        # for architectures that narrow with depth. A value of 1.0 keeps
        # every layer the same width.
        "dim_shrinking_scale": tune.uniform(0.5, 1.0),
        # Number of hidden layers. Upper bound is exclusive, so this gives
        # 1, 2 or 3.
        "num_layers": tune.randint(1, 4),
        # Initial Adam step size, before any scheduler decay.
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        # Batch size used during model fitting.
        "batch_size": tune.choice([32, 64, 128]),
        # Coefficient on the L2 penalty, applied as Adam's weight_decay.
        "L2_weight_norm": tune.loguniform(1e-8, 1e-2),
        # Coefficient on the L1 penalty, which places a sparsity prior on
        # the weights.
        "L1_weight_norm": tune.loguniform(1e-12, 1e-2),
        # Activation used throughout the trunk. Generally relu is fine.
        "activation_function": tune.choice(["relu", "tanh", "leaky_relu"]),
        # How many validation rounds without improvement in val_auprc_avg
        # before training is stopped early.
        "patience": tune.choice([2]),
        # Dropout probability inside each block. Set to 0 to not use
        # dropout.
        "dropout_rate": tune.uniform(0.0, 0.9),
        # Whether to add skip connections around the trunk blocks. These are
        # applied where a block's input and output widths match.
        "use_residual": tune.choice([True, False]),
        # Whether batch normalization is used.
        "use_batch_norm": tune.choice([True, False]),
        # Interval, in optimizer steps, between learning rate decays. Ray's
        # upper bound is exclusive.
        "scheduler_step_size": tune.lograndint(2, 20),
        # Factor the learning rate is multiplied by at each interval.
        "scheduler_gamma": tune.uniform(0.1, 0.9),
        # Directory checkpoints are written to.
        "model_file_path": tune.choice(
            ["/path/to/where/I/want/my/models/to/be/"]
        )
    }

    algo = OptunaSearch(metric=SELECTION_METRIC, mode=SELECTION_MODE)

    summary_metrics_path = (
        "/path/to/where/I/want/my/metrics/to/be/saved/to/summary_metrics.tsv"
    )
    append_metrics_callback = AppendMetricsCallback(
        csv_path=summary_metrics_path)

    # max_epochs is the ceiling on how long any one model trains. Early
    # stopping normally fires well before it.
    tuner = tune.run(
        tune.with_parameters(
            train_with_tune,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            rep_col=rep_col_name,
            target_cols=target_col_names,
            max_epochs=25
        ),
        config=search_space,
        search_alg=algo,
        num_samples=NUM_TRIALS,
        resources_per_trial={"cpu": 1, "gpu": 0},
        metric=SELECTION_METRIC,
        mode=SELECTION_MODE,
        callbacks=[append_metrics_callback]
    )

    best_config = tuner.get_best_config(
        metric=SELECTION_METRIC, mode=SELECTION_MODE)
    print("Best hyperparameters found were:", best_config)


if __name__ == "__main__":
    main()