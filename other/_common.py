"""Path and table helpers for standalone analysis scripts."""

from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path, base=REPOSITORY_ROOT):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def separator(path, override=None):
    if override is not None:
        value = "\t" if override == r"\t" else override
        if len(value) != 1:
            raise ValueError("The field separator must be one character.")
        return value
    return "\t" if Path(path).suffix.lower() in {".tsv", ".tab", ".txt"} else ","


def finite_columns(frame, columns):
    """Return finite numeric rows plus the number of excluded rows."""
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(values.to_numpy()).all(axis=1)
    if not keep.any():
        raise ValueError("No finite numeric observations remain.")
    return values.loc[keep], int((~keep).sum())


def append_suffix(prefix, suffix):
    """Append a filename suffix without removing dots from the prefix."""
    prefix = resolve_path(prefix)
    return prefix.with_name(prefix.name + suffix)
