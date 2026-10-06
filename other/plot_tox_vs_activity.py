"""Plot activity versus toxicity and report counts at explicit thresholds."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from ._common import append_suffix, finite_columns, resolve_path, separator
else:
    from _common import append_suffix, finite_columns, resolve_path, separator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot activity vs toxicity scatter.")
    p.add_argument("--input", type=str, required=True, help="Input TSV/CSV file.")
    p.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Delimiter (default: guessed from extension, .tsv -> '\\t', else ',').",
    )
    p.add_argument(
        "--activity-col",
        type=str,
        required=True,
        help="Column name for activity score (x-axis).",
    )
    p.add_argument(
        "--tox-col",
        type=str,
        required=True,
        help="Column name for toxicity score (y-axis).",
    )
    p.add_argument(
        "--activity-threshold",
        type=float,
        default=0.5,
        help="Vertical cutoff for activity (default: 0.5).",
    )
    p.add_argument(
        "--tox-threshold",
        type=float,
        default=0.2,
        help="Horizontal cutoff for toxicity (default: 0.2).",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Output prefix for figure files (PNG/SVG).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    return p.parse_args()


def quadrant_counts(x, y, activity_threshold=0.5, toxicity_threshold=0.2):
    """Equality belongs to the high-score side of each threshold."""
    if not np.isfinite([activity_threshold, toxicity_threshold]).all():
        raise ValueError("Thresholds must be finite.")
    high_activity = x >= activity_threshold
    high_toxicity = y >= toxicity_threshold
    return {
        "high_activity_low_toxicity": int(np.sum(high_activity & ~high_toxicity)),
        "high_activity_high_toxicity": int(np.sum(high_activity & high_toxicity)),
        "low_activity_low_toxicity": int(np.sum(~high_activity & ~high_toxicity)),
        "low_activity_high_toxicity": int(np.sum(~high_activity & high_toxicity)),
    }


def main():
    args = parse_args()
    source = resolve_path(args.input)
    frame = pd.read_csv(source, sep=separator(source, args.sep))
    values, dropped = finite_columns(frame, [args.activity_col, args.tox_col])
    x, y = values[args.activity_col].to_numpy(), values[args.tox_col].to_numpy()
    counts = quadrant_counts(x, y, args.activity_threshold, args.tox_threshold)
    paths = [append_suffix(args.out_prefix, suffix) for suffix in (".png", ".svg")]
    if source in paths:
        raise ValueError("Output paths must differ from the input path.")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(4.5, 4.5))
        try:
            axis.scatter(x, y, s=4, alpha=0.5, linewidths=0, rasterized=True)
            axis.axvline(args.activity_threshold, linestyle="--", linewidth=1)
            axis.axhline(args.tox_threshold, linestyle="--", linewidth=1)
            axis.set(
                xlabel=args.activity_col, ylabel=args.tox_col, title=args.title or ""
            )
            figure.tight_layout()
            for path in paths:
                figure.savefig(path, bbox_inches="tight")
        finally:
            plt.close(figure)
    print(f"Plotted {len(x)} rows; excluded {dropped} nonfinite rows.")
    for label, count in counts.items():
        print(f"{label}: {count}")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
