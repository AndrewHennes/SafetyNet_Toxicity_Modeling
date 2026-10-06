"""Train a multitask SafetyNet model and evaluate its best validation checkpoint."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from safetynet.cli._common import read_model_table
from safetynet.featurization.table_utils import positive_int, resolve_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Simple multitask training script using MultiTaskFeedForward. "
            "Works for tox or ABX-like datasets with binary labels."
        )
    )

    # ---- Data ----
    p.add_argument(
        "--input",
        required=True,
        help="TSV/CSV with SMILES and/or representation + binary label columns.",
    )
    p.add_argument(
        "--sep",
        default=None,
        help="Input delimiter; inferred from extension by default.",
    )
    p.add_argument(
        "--rep-col",
        default="Minimol_Fingerprint",
        help="Name of representation column (default: Minimol_Fingerprint).",
    )
    p.add_argument(
        "--smiles-col",
        default=None,
        help=(
            "If provided and the representation column is missing, "
            "we will featurize this SMILES column with MiniMol."
        ),
    )

    p.add_argument(
        "--target-col",
        dest="target_cols",
        action="append",
        help=(
            "Name of a binary label column. "
            "Can be passed multiple times, e.g. "
            "--target-col HepG2_10uM_Binarized "
            "--target-col HSkMC_10uM_Binarized ..."
        ),
        default=None,
    )
    # Backwards compatibility: old --label-cols option
    p.add_argument(
        "--label-cols",
        nargs="+",
        help=(
            "[DEPRECATED] List of binary label columns. "
            "Prefer using multiple --target-col flags instead."
        ),
        default=None,
    )

    # ---- Output / training params ----
    p.add_argument(
        "--out-dir",
        default=str(resolve_path("outputs/safetynet_train")),
        help="Directory to save checkpoints and logs (default: outputs/safetynet_train).",
    )
    p.add_argument(
        "--batch-size",
        type=positive_int,
        default=256,
        help="Batch size (default: 256).",
    )
    p.add_argument(
        "--max-epochs",
        type=positive_int,
        default=50,
        help="Max epochs (default: 50).",
    )
    p.add_argument(
        "--hidden-dims",
        type=str,
        default="512,512",
        help="Comma-separated hidden sizes (default: '512,512').",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Weight decay (default: 1e-6).",
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (0 = CPU, default: 1).",
    )

    return p


def main() -> int:
    args = _build_argparser().parse_args()
    if args.target_cols and args.label_cols:
        raise ValueError("Use either --target-col or --label-cols, not both.")
    label_cols = args.target_cols or args.label_cols
    if not label_cols or len(label_cols) != len(set(label_cols)):
        raise ValueError("Provide at least one unique binary label column.")
    hidden_dims = tuple(int(value.strip()) for value in args.hidden_dims.split(","))
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("Hidden layer widths must be positive integers.")
    if (
        not np.isfinite([args.dropout, args.lr, args.weight_decay]).all()
        or not 0 <= args.dropout < 1
        or args.lr <= 0
        or args.weight_decay < 0
    ):
        raise ValueError("Require 0 <= dropout < 1, lr > 0, and weight_decay >= 0.")
    if args.gpus < 0:
        raise ValueError("--gpus must be nonnegative.")
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from safetynet.toxicity.model import MultiTaskFeedForward
    from torch.utils.data import DataLoader, Dataset, random_split

    pl.seed_everything(42, workers=True)
    frame, features = read_model_table(
        args.input, args.rep_col, args.smiles_col, args.sep
    )
    if any(name not in frame for name in label_cols):
        raise ValueError("One or more requested label columns are missing.")
    # Only blank cells and -1 represent missing labels. Reject malformed values.
    labels = (
        frame[label_cols]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32)
    )
    if not np.isin(labels[~np.isnan(labels)], [-1, 0, 1]).all():
        raise ValueError("Labels must be 0, 1, -1 (missing), or blank.")
    labels = np.where(np.isnan(labels), -1, labels)
    observed = (labels != -1).any(axis=1)
    if not observed.all():
        print(f"Excluded {int((~observed).sum())} rows with no observed labels.")
    features, labels = features[observed], labels[observed]
    if len(features) < 10:
        raise ValueError(
            "At least 10 labelled rows are required for train/validation/test splits."
        )
    if not (labels != -1).any(axis=0).all():
        raise ValueError("Every task must have at least one observed label.")

    class TableDataset(Dataset):
        def __len__(self):
            return len(features)

        def __getitem__(self, index):
            return {
                "x": torch.from_numpy(features[index]),
                "y": torch.from_numpy(labels[index]),
            }

    total = len(features)
    lengths = [int(0.8 * total), int(0.1 * total)]
    lengths.append(total - sum(lengths))
    splits = random_split(
        TableDataset(), lengths, generator=torch.Generator().manual_seed(42)
    )
    if not (labels[splits[0].indices] != -1).any(axis=0).all():
        raise ValueError(
            "The training split has a task with no labels; provide more labelled examples."
        )
    loaders = [
        DataLoader(split, batch_size=args.batch_size, shuffle=index == 0, num_workers=0)
        for index, split in enumerate(splits)
    ]
    model = MultiTaskFeedForward(
        input_dim=features.shape[1],
        output_dim=len(label_cols),
        hidden_dims=hidden_dims,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
        L2_weight_norm=args.weight_decay,
        batch_size=args.batch_size,
    )
    output = resolve_path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(output),
        filename="safetynet_checkpoint_{epoch}",
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    use_gpu = args.gpus > 0 and torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=args.gpus if use_gpu else 1,
        max_epochs=args.max_epochs,
        default_root_dir=str(output),
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders[1])
    trainer.test(model=model, dataloaders=loaders[2], ckpt_path="best")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
