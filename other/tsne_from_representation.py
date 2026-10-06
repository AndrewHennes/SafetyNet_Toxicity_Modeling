"""Project fingerprint arrays or table representations with reproducible t-SNE."""

from __future__ import annotations

import argparse
import ast
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

if __package__:
    from ._common import resolve_path, separator
else:
    from _common import resolve_path, separator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="t-SNE over ECFP or other representations.")
    p.add_argument(
        "--mode",
        type=str,
        choices=["npz", "table"],
        required=True,
        help="Input mode: 'npz' (ECFP .npz) or 'table' (representation column in table).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(resolve_path("datasets")),
        help="Base directory for files (default: repository datasets).",
    )
    # npz mode
    p.add_argument(
        "--npz-files",
        type=str,
        nargs="*",
        default=None,
        help="For --mode npz: list of .npz files containing 'X'.",
    )
    # table mode
    p.add_argument(
        "--tables",
        type=str,
        nargs="*",
        default=None,
        help="For --mode table: one or more CSV/TSV tables with representation column.",
    )
    p.add_argument(
        "--rep-col",
        type=str,
        default="Minimol_Fingerprint",
        help="Representation column name (for --mode table). Default: Minimol_Fingerprint.",
    )
    p.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Delimiter for table mode. If not set, guessed from file extension.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(resolve_path("outputs/tsne_embeddings.npz")),
        help="Output .npz path (relative to --data-dir unless absolute).",
    )
    # t-SNE hyperparameters (default to your previous config)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--early-exaggeration", type=float, default=12.0)
    p.add_argument("--angle", type=float, default=0.5)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def _validate_matrix(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} must contain a nonempty, finite 2D representation matrix."
        )
    return matrix


def _parse_rep_rows(series):
    vectors, rows, width = [], [], None
    for row, value in enumerate(series):
        try:
            vector = np.asarray(
                ast.literal_eval(value) if isinstance(value, str) else value,
                dtype=np.float32,
            )
        except (ValueError, TypeError, SyntaxError):
            continue
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            continue
        if width is not None and len(vector) != width:
            raise ValueError(f"Inconsistent representation width at row {row}.")
        width = len(vector)
        vectors.append(vector)
        rows.append(row)
    if not vectors:
        raise ValueError("No valid vectors in the representation column.")
    if len(vectors) != len(series):
        warnings.warn(
            f"Excluded {len(series) - len(vectors)} invalid vectors; source row indices are saved.",
            stacklevel=2,
        )
    return np.stack(vectors), np.asarray(rows, dtype=np.int64)


def parse_rep_column(series):
    """Parse finite vectors, reporting excluded rows and rejecting mixed widths."""
    return _parse_rep_rows(series)[0]


def load_stack(base_dir, filenames, mode, rep_col="Minimol_Fingerprint", sep=None):
    """Concatenate arrays and retain each point's source-file and row indices."""
    arrays, labels, names, counts, source_rows = [], [], [], [], []
    for index, filename in enumerate(filenames):
        path = resolve_path(filename, base_dir)
        if mode == "npz":
            with np.load(path, allow_pickle=False) as archive:
                key = "X" if "X" in archive else "features"
                if key not in archive:
                    raise ValueError(f"{path} must contain X or features.")
                matrix = _validate_matrix(archive[key], path)
                rows = (
                    archive["row_indices"]
                    if "row_indices" in archive
                    else np.arange(len(matrix))
                )
                if rows.shape != (len(matrix),) or not np.issubdtype(
                    rows.dtype, np.integer
                ):
                    raise ValueError(f"Invalid row_indices in {path}.")
        else:
            frame = pd.read_csv(path, sep=separator(path, sep))
            if rep_col not in frame:
                raise ValueError(
                    f"Missing representation column {rep_col!r} in {path}."
                )
            matrix, rows = _parse_rep_rows(frame[rep_col])
        if arrays and matrix.shape[1] != arrays[0].shape[1]:
            raise ValueError("All input files must use the same representation width.")
        arrays.append(matrix)
        source_rows.append(rows)
        labels.append(np.full(len(matrix), index, dtype=np.int32))
        names.append(path.stem)
        counts.append(len(matrix))
    if not arrays:
        raise ValueError("Supply at least one input file.")
    return (
        np.concatenate(arrays),
        np.concatenate(labels),
        names,
        counts,
        np.concatenate(source_rows),
    )


def load_npz_stack(base_dir, filenames):
    """Compatibility wrapper returning matrix, labels, names, and counts."""
    return load_stack(base_dir, filenames, "npz")[:4]


def load_table_stack(base_dir, filenames, rep_col, sep):
    """Compatibility wrapper returning matrix, labels, names, and counts."""
    return load_stack(base_dir, filenames, "table", rep_col, sep)[:4]


def main():
    args = parse_args()
    base = resolve_path(args.data_dir)
    filenames = args.npz_files if args.mode == "npz" else args.tables
    if not filenames:
        raise ValueError("Provide --npz-files for npz mode or --tables for table mode.")
    matrix, labels, names, counts, rows = load_stack(
        base, filenames, args.mode, args.rep_col, args.sep
    )
    if min(matrix.shape) < 2:
        raise ValueError(
            "PCA initialization requires at least two rows and two features."
        )
    if not np.isfinite(args.perplexity) or not 0 < args.perplexity < len(matrix):
        raise ValueError(
            "Perplexity must be positive and smaller than the number of points."
        )
    if args.n_iter < 250:
        raise ValueError("--n-iter must be at least 250.")
    if (
        not np.isfinite(args.early_exaggeration)
        or args.early_exaggeration < 1
        or not 0 <= args.angle <= 1
    ):
        raise ValueError("Require early_exaggeration >= 1 and 0 <= angle <= 1.")
    output = resolve_path(args.output, base)
    if output in {resolve_path(name, base) for name in filenames}:
        raise ValueError("Output must not overwrite an input file.")
    embedding = TSNE(
        n_components=2,
        init="pca",
        perplexity=args.perplexity,
        learning_rate="auto",
        max_iter=args.n_iter,
        early_exaggeration=args.early_exaggeration,
        angle=args.angle,
        method="barnes_hut",
        random_state=args.random_state,
    ).fit_transform(matrix)
    metadata = {
        "mode": args.mode,
        "input_files": [str(resolve_path(name, base)) for name in filenames],
        "rep_col": args.rep_col,
        "perplexity": args.perplexity,
        "max_iter": args.n_iter,
        "early_exaggeration": args.early_exaggeration,
        "angle": args.angle,
        "random_state": args.random_state,
        "metric": "euclidean",
        "row_indices": "zero-based input data rows",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(
            stream,
            Y=embedding,
            y=labels,
            label_names=np.asarray(names, dtype=str),
            counts=np.asarray(counts),
            row_indices=rows,
            meta=json.dumps(metadata, allow_nan=False),
        )
    print(f"Saved {len(matrix)} points to {output}")


if __name__ == "__main__":
    main()
