#!/usr/bin/env python3
"""Generate MiniMol representations while preserving input row order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from safetynet.featurization.table_utils import (
    positive_int,
    require_distinct_paths,
    resolve_path,
    separator,
)


class MinimolFeaturizer:
    """Load MiniMol lazily and validate the shape of every embedding batch."""

    def __init__(self):
        self.model = None

    def embed(
        self, smiles: Sequence[str], batch_size: int = 1024, show_progress: bool = True
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not len(smiles):
            raise ValueError("No SMILES were supplied.")
        from rdkit import Chem

        for index, value in enumerate(smiles):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing SMILES at row {index}.")
            molecule = Chem.MolFromSmiles(value)
            if (
                molecule is None
                or molecule.GetNumAtoms() == 0
                or any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())
            ):
                raise ValueError(
                    f"Invalid or placeholder SMILES at row {index}: {value!r}"
                )
        if self.model is None:
            from minimol import Minimol

            self.model = Minimol()
        starts = range(0, len(smiles), batch_size)
        if show_progress:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc="MiniMol featurization")
        arrays = []
        width = None
        for start in starts:
            batch = list(smiles[start : start + batch_size])
            embeddings = self.model(batch)
            if len(embeddings) != len(batch):
                raise ValueError(
                    "MiniMol returned a different number of rows than its input."
                )
            array = np.stack(
                [embedding.detach().cpu().numpy() for embedding in embeddings]
            )
            if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
                raise ValueError("MiniMol returned invalid embeddings.")
            if width is not None and array.shape[1] != width:
                raise ValueError("MiniMol returned inconsistent embedding widths.")
            width = array.shape[1]
            arrays.append(array.astype(np.float32))
        return np.concatenate(arrays, axis=0)


def featurize_smiles(
    smiles: Sequence[str], batch_size: int = 1024, show_progress: bool = True
) -> np.ndarray:
    """Return a finite float32 matrix with one row per SMILES string."""
    return MinimolFeaturizer().embed(smiles, batch_size, show_progress)


def featurize_file(
    input_path: str | Path,
    smiles_col: str = "SMILES",
    output_npz: str | Path | None = None,
    output_table: str | Path | None = None,
    rep_col: str = "Minimol_Fingerprint",
    batch_size: int = 1024,
    sep: str | None = None,
) -> pd.DataFrame:
    """Read a table and optionally save embeddings and an annotated table.

    ``sep`` controls input parsing. Output tables use their own extension to
    select CSV or TSV. Relative paths are anchored to the repository root.
    """
    source = resolve_path(input_path)
    outputs = [resolve_path(p) for p in (output_npz, output_table) if p is not None]
    for target in outputs:
        require_distinct_paths(source, target)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Array and table output paths must be different.")
    frame = pd.read_csv(
        source, sep=separator(source, sep), dtype=str, keep_default_na=False
    )
    if smiles_col not in frame:
        raise ValueError(f"SMILES column {smiles_col!r} is missing from {source}.")
    if rep_col in frame or rep_col == smiles_col:
        raise ValueError(f"Output representation column {rep_col!r} already exists.")
    smiles = frame[smiles_col].tolist()
    features = featurize_smiles(smiles, batch_size)
    frame[rep_col] = [
        json.dumps(vector.tolist(), allow_nan=False) for vector in features
    ]
    if output_npz is not None:
        target = resolve_path(output_npz)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            np.savez_compressed(
                stream, features=features, smiles=np.asarray(smiles, dtype=str)
            )
    if output_table is not None:
        target = resolve_path(output_table)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, sep=separator(target), index=False)
    return frame


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--rep-col", default="Minimol_Fingerprint")
    parser.add_argument("--output-npz")
    parser.add_argument("--output-table")
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument(
        "--sep", help="Input delimiter; inferred from extension by default."
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    if not args.output_npz and not args.output_table:
        parser.error("Provide --output-npz or --output-table to save the embeddings.")
    featurize_file(
        args.input,
        args.smiles_col,
        args.output_npz,
        args.output_table,
        args.rep_col,
        args.batch_size,
        args.sep,
    )


if __name__ == "__main__":
    main()
