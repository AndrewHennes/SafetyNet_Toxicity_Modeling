"""Shared path, delimiter, and representation validation."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path = REPOSITORY_ROOT) -> Path:
    """Resolve relative paths against the repository, independently of cwd."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def separator(path: str | Path, override: str | None = None) -> str:
    """Infer CSV/TSV delimiters and accept a literal ``\\t`` override."""
    if override is not None:
        value = "\t" if override == r"\t" else override
        if len(value) != 1:
            raise ValueError("The field separator must be one character.")
        return value
    return "\t" if Path(path).suffix.lower() in {".tsv", ".tab", ".txt"} else ","


def parse_vector(value) -> np.ndarray:
    """Parse one nonempty, finite, one-dimensional numeric representation."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Expected a numeric vector such as [0.1, 0.2].") from exc
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Representation contains nonnumeric values.") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Representation must be a nonempty, finite numeric vector.")
    return vector


def representation_matrix(series: pd.Series) -> np.ndarray:
    """Validate every row; never silently drop compounds or reorder them."""
    if series.empty:
        raise ValueError("The input table contains no rows.")
    vectors = []
    width = None
    for row, value in series.items():
        try:
            vector = parse_vector(value)
            if width is not None and vector.size != width:
                raise ValueError(f"Expected {width} features, found {vector.size}.")
        except ValueError as exc:
            raise ValueError(
                f"Invalid representation at row index {row!r}: {exc}"
            ) from exc
        width = vector.size
        vectors.append(vector)
    return np.stack(vectors)


def require_distinct_paths(source: Path, destination: Path) -> None:
    """Prevent an output from replacing its input, including symlink aliases."""
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must be different.")


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise ValueError("Expected a positive integer.")
    return number
