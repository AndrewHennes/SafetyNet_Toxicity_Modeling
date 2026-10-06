"""Predict toxicity from a SafetyNet checkpoint and a molecular table."""

from __future__ import annotations

import argparse

import numpy as np
from safetynet.cli._common import predict_probabilities, read_model_table
from safetynet.featurization.table_utils import (
    positive_int,
    require_distinct_paths,
    resolve_path,
    separator,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Predict with a custom SafetyNet MultiTaskFeedForward checkpoint. "
            "You can either: (A) start from raw SMILES (we featurize with MiniMol), "
            "or (B) start from a table that already contains MiniMol_Fingerprint."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input CSV/TSV file (SMILES-only or pre-featurized).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV/TSV with prediction columns appended.",
    )
    p.add_argument(
        "--sep",
        default=None,
        help="Input separator; inferred from extension by default.",
    )
    p.add_argument(
        "--smiles-col",
        default=None,
        help=(
            "If provided, treat --input as SMILES table and featurize "
            "with MiniMol. Otherwise, assume the file already has a "
            "representation column."
        ),
    )
    p.add_argument(
        "--rep-col",
        default="Minimol_Fingerprint",
        help="Name of representation column (default: Minimol_Fingerprint).",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to MultiTaskFeedForward .ckpt (from safetynet_train).",
    )
    p.add_argument(
        "--task-names",
        default=None,
        help=(
            "Comma-separated names for tasks, e.g. 'HepG2,HSkMC,IMR90'. "
            "If omitted, uses task_0, task_1, ... based on model output_dim."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional probability threshold. If set, adds binary columns "
            "Pred_<task>_binary (0/1) using this cutoff."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=positive_int,
        default=1024,
        help="Batch size for prediction (default: 1024).",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for inference (default: cuda, falls back to cpu if unavailable).",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    if args.threshold is not None and (
        not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1
    ):
        raise ValueError("--threshold must be between zero and one.")
    source, output = resolve_path(args.input), resolve_path(args.output)
    require_distinct_paths(source, output)
    checkpoint = resolve_path(args.ckpt)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import torch
    from safetynet.toxicity.model import MultiTaskFeedForward

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = "cpu"
    frame, features = read_model_table(source, args.rep_col, args.smiles_col, args.sep)
    model = MultiTaskFeedForward.load_from_checkpoint(
        str(checkpoint), map_location="cpu"
    )
    names = (
        [name.strip() for name in args.task_names.split(",")]
        if args.task_names
        else [f"task_{i}" for i in range(model.output_dim)]
    )
    if (
        len(names) != model.output_dim
        or len(set(names)) != len(names)
        or not all(names)
    ):
        raise ValueError(
            "Task names must be nonempty, unique, and match the checkpoint output count."
        )
    columns = [f"Pred_{name}" for name in names]
    if args.threshold is not None:
        columns += [f"Pred_{name}_binary" for name in names]
    if set(columns) & set(frame):
        raise ValueError("Prediction columns already exist in the input table.")
    probabilities = predict_probabilities(model, features, args.batch_size, device)
    for index, name in enumerate(names):
        frame[f"Pred_{name}"] = probabilities[:, index]
        if args.threshold is not None:
            frame[f"Pred_{name}_binary"] = (
                probabilities[:, index] >= args.threshold
            ).astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep=separator(output), index=False)
    print(f"Wrote {len(frame)} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
