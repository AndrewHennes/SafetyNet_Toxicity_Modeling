"""Compute cached Morgan fingerprints with reproducible source metadata."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from safetynet.featurization.table_utils import resolve_path, separator

DEFAULT_SMILES_CANDIDATES = ["SMILES", "smiles", "Smiles"]
DEFAULT_NAME_CANDIDATES = ["Name", "name", "Compound", "compound", "ID", "id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute / cache RDKit Morgan (ECFP) fingerprints."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(resolve_path("datasets")),
        help="Base directory containing input files (default: repository datasets).",
    )
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV/TSV files (relative to --data-dir) to featurize.",
    )
    p.add_argument(
        "--fp-bits",
        type=int,
        default=2048,
        help="Number of bits for Morgan/ECFP fingerprints (default: 2048).",
    )
    p.add_argument(
        "--fp-radius",
        type=int,
        default=2,
        help="Radius for Morgan/ECFP (default: 2, i.e. ECFP4).",
    )
    p.add_argument(
        "--no-chiral",
        action="store_true",
        help="Disable chirality in the fingerprints (default: use chirality).",
    )
    p.add_argument(
        "--smiles-candidates",
        type=str,
        nargs="*",
        default=DEFAULT_SMILES_CANDIDATES,
        help="Candidate SMILES column names to search.",
    )
    p.add_argument(
        "--name-candidates",
        type=str,
        nargs="*",
        default=DEFAULT_NAME_CANDIDATES,
        help="Candidate name/ID column names to search.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz outputs (default: skip if exists).",
    )
    return p.parse_args()


def find_column(frame, candidates):
    return next((name for name in candidates if name in frame), None)


def mol_from_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if (
        molecule is None
        or molecule.GetNumAtoms() == 0
        or any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())
    ):
        return None
    return molecule


def morgan_fp_array(mol, fp_bits, fp_radius, use_chirality):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fp_radius, fpSize=fp_bits, includeChirality=use_chirality
    )
    return np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=bool)


def compute_fingerprints(
    source,
    output,
    fp_bits=2048,
    fp_radius=2,
    use_chirality=True,
    smiles_candidates=None,
    name_candidates=None,
    overwrite=False,
):
    """Save fingerprints and source row indices; reject stale cache reuse."""
    source, output = resolve_path(source), resolve_path(output)
    if source == output:
        raise ValueError("Input and output must be different files.")
    if fp_bits <= 0 or fp_radius < 0:
        raise ValueError("Fingerprint width must be positive and radius nonnegative.")
    smiles_candidates = (
        DEFAULT_SMILES_CANDIDATES if smiles_candidates is None else smiles_candidates
    )
    name_candidates = (
        DEFAULT_NAME_CANDIDATES if name_candidates is None else name_candidates
    )
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    identity = {
        "source_csv": str(source),
        "source_sha256": digest.hexdigest(),
        "fp_bits": fp_bits,
        "fp_radius": fp_radius,
        "fp_chiral": use_chirality,
        "smiles_candidates": list(smiles_candidates),
        "name_candidates": list(name_candidates),
        "rdkit_version": rdBase.rdkitVersion,
    }
    if output.exists() and not overwrite:
        with np.load(output, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["meta"]))
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise ValueError(
                    f"Cached fingerprint settings or source changed; use --overwrite for {output}."
                )
        print(f"Using verified cache {output}")
        return output
    frame = pd.read_csv(source, sep=separator(source), dtype=str, keep_default_na=False)
    smiles_col = find_column(frame, smiles_candidates)
    name_col = find_column(frame, name_candidates)
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {source}.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fp_radius, fpSize=fp_bits, includeChirality=use_chirality
    )
    fingerprints, smiles, names, rows = [], [], [], []
    for index, value in enumerate(frame[smiles_col]):
        molecule = mol_from_smiles(value)
        if molecule is None:
            continue
        fingerprints.append(generator.GetFingerprintAsNumPy(molecule))
        smiles.append(value)
        name = frame.iloc[index][name_col] if name_col else f"{source.stem}_{index}"
        names.append("" if pd.isna(name) else str(name))
        rows.append(index)
    if not fingerprints:
        raise ValueError(f"No valid molecules found in {source}.")
    metadata = {
        **identity,
        "smiles_col": smiles_col,
        "name_col": name_col,
        "n_molecules": len(rows),
        "n_excluded": len(frame) - len(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(
            stream,
            X=np.asarray(fingerprints, dtype=bool),
            smiles=np.asarray(smiles, dtype=str),
            names=np.asarray(names, dtype=str),
            row_indices=np.asarray(rows, dtype=np.int64),
            meta=json.dumps(metadata),
        )
    print(
        f"Saved {len(rows)} fingerprints to {output}; excluded {metadata['n_excluded']} invalid SMILES."
    )
    return output


def main():
    args = parse_args()
    base = resolve_path(args.data_dir)
    sources = [resolve_path(name, base) for name in args.inputs]
    outputs = [
        resolve_path(
            f"outputs/fingerprints/{path.stem}_ecfp{args.fp_bits}_r{args.fp_radius}.npz"
        )
        for path in sources
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError(
            "Input filenames share a stem and would overwrite the same fingerprint output."
        )
    for source, output in zip(sources, outputs):
        compute_fingerprints(
            source,
            output,
            args.fp_bits,
            args.fp_radius,
            not args.no_chiral,
            args.smiles_candidates,
            args.name_candidates,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
