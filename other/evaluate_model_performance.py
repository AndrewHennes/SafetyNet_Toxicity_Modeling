"""Evaluate binary predictions with ROC and precision-recall curves."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

if __package__:
    from ._common import append_suffix, finite_columns, resolve_path, separator
else:
    from _common import append_suffix, finite_columns, resolve_path, separator


def evaluate_predictions(frame, label_col, score_col, pos_label=1.0):
    """Compute binary metrics after explicitly excluding nonfinite observations."""
    values, dropped = finite_columns(frame, [label_col, score_col])
    labels = values[label_col].to_numpy()
    scores = values[score_col].to_numpy()
    classes = np.unique(labels)
    if len(classes) != 2 or pos_label not in classes:
        raise ValueError(
            "Evaluation requires exactly two label classes, including --pos-label."
        )
    binary = (labels == pos_label).astype(int)
    metrics = {
        "label_col": label_col,
        "score_col": score_col,
        "pos_label": float(pos_label),
        "n_samples": len(binary),
        "n_dropped": dropped,
        "positive_fraction": float(binary.mean()),
        "roc_auc": float(roc_auc_score(binary, scores)),
        "average_precision": float(average_precision_score(binary, scores)),
    }
    return metrics, roc_curve(binary, scores), precision_recall_curve(binary, scores)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model ROC/PRC performance.")
    p.add_argument("--input", type=str, required=True, help="Input TSV/CSV file.")
    p.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Delimiter (default: guessed from extension, .tsv -> '\\t', else ',').",
    )
    p.add_argument(
        "--label-col",
        type=str,
        required=True,
        help="Name of binary label column (0/1, False/True).",
    )
    p.add_argument(
        "--score-col",
        type=str,
        required=True,
        help="Name of score column (predicted probability or continuous score).",
    )
    p.add_argument(
        "--pos-label",
        type=float,
        default=1.0,
        help="Positive class label encoding (default: 1).",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Output prefix for plot and metrics JSON.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title (will appear on ROC subplot).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    source = resolve_path(args.input)
    frame = pd.read_csv(source, sep=separator(source, args.sep))
    metrics, (fpr, tpr, _), (precision, recall, _) = evaluate_predictions(
        frame, args.label_col, args.score_col, args.pos_label
    )
    metrics["input_file"] = str(source)
    image_path = append_suffix(args.out_prefix, "_roc_prc.png")
    json_path = append_suffix(args.out_prefix, "_metrics.json")
    if source in {image_path, json_path}:
        raise ValueError("Output paths must differ from the input path.")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(8, 4))
        try:
            roc_axis.plot(fpr, tpr, linewidth=1.5)
            roc_axis.plot([0, 1], [0, 1], "k--", linewidth=1)
            roc_title = f"ROC (AUC = {metrics['roc_auc']:.3f})"
            roc_axis.set(
                xlabel="False positive rate",
                ylabel="True positive rate",
                title=f"{args.title}\n{roc_title}" if args.title else roc_title,
            )
            pr_axis.step(recall, precision, where="post", linewidth=1.5)
            pr_axis.axhline(
                metrics["positive_fraction"], color="black", linestyle="--", linewidth=1
            )
            pr_axis.set(
                xlabel="Recall",
                ylabel="Precision",
                title=f"Precision-recall (AP = {metrics['average_precision']:.3f})",
            )
            figure.tight_layout()
            figure.savefig(image_path, bbox_inches="tight")
        finally:
            plt.close(figure)
    json_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(
        f"Evaluated {metrics['n_samples']} rows; excluded {metrics['n_dropped']} nonfinite rows."
    )
    print(
        f"ROC AUC={metrics['roc_auc']:.4f}; average precision={metrics['average_precision']:.4f}"
    )
    print(f"Saved {image_path} and {json_path}")


if __name__ == "__main__":
    main()
