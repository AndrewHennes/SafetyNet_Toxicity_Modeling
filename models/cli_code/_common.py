"""Shared input validation and batch prediction for the existing interfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetynet.featurization.table_utils import (
    representation_matrix,
    resolve_path,
    separator,
)


def read_model_table(input_path, rep_col, smiles_col=None, sep=None):
    """Read representations or featurize SMILES without temporary table files."""
    path = resolve_path(input_path)
    frame = pd.read_csv(
        path, sep=separator(path, sep), dtype=str, keep_default_na=False
    )
    if frame.empty:
        raise ValueError("The input table contains no rows.")
    if rep_col not in frame:
        if not smiles_col or smiles_col not in frame:
            raise ValueError(f"Provide column {rep_col!r}, or a valid --smiles-col.")
        from safetynet.featurization.minimol_featurization import featurize_smiles

        features = featurize_smiles(frame[smiles_col].tolist())
        frame[rep_col] = [json.dumps(row.tolist(), allow_nan=False) for row in features]
    return frame, representation_matrix(frame[rep_col])


def predict_probabilities(model, features, batch_size, device):
    """Transfer only the current batch to the inference device."""
    import torch

    if batch_size <= 0 or len(features) == 0:
        raise ValueError(
            "Prediction requires a positive batch size and nonempty input."
        )
    if features.shape[1] != model.input_dim:
        raise ValueError(
            f"Checkpoint expects {model.input_dim} features; input has {features.shape[1]}."
        )
    results = []
    model = model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            results.append(torch.sigmoid(model(batch)).cpu().numpy())
    probabilities = np.concatenate(results)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced nonfinite predictions.")
    return probabilities


def checkpoint_order(path: Path):
    """Sort numbered checkpoint IDs numerically, with a stable name fallback."""
    import re

    match = re.search(r"_(\d+)\.ckpt$", path.name)
    return (int(match[1]) if match else -1, path.name)
