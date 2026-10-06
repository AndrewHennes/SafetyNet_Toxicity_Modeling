#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Antibiotic-activity inference pipeline – streamed, memory-safe, multi-CPU
Created on Sat Feb 15 12:22:59 2025
@author: asselism

Split out of ScoreToxAndAM.py; the toxicity half lives in score_with_toxicity_models.py.

Activate environment:
    source /Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/activate
"""

from __future__ import annotations

import ast
import json
import os
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
#     Every path below falls back to an environment variable first, so a
#     user of this repo can run the script without editing it:
#         AM_INPUT=... AM_MODEL_DIR=... python score_with_antibiotic_models.py
# ──────────────────────────────────────────────────────────────────────
# Directory this file lives in.  The `antibiotic_model_checkpoints` checkpoint folder
# is expected to sit right next to this script, so the script moves with its
# checkpoints and works from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

RAW_TSV = os.getenv(
    "AM_INPUT",
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Revision_Related/300K_Analyses/SA_Related/SA1_with_Minimol_Representations.csv",
)  # ← original input (comma-separated)

ACTIVITY_TSV = os.getenv(
    "AM_PRED_OUTPUT",
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Revision_Related/300K_Analyses/SA_Related/SA1_with_Minimol_Representations_int.tsv",
)  # ← intermediate: one column per ensemble member

FINAL_TSV = os.getenv(
    "AM_OUTPUT",
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Revision_Related/300K_Analyses/SA_Related/SA1_with_Minimol_Representations_with_AM.tsv",
)  # ← final output: per-species mean activity appended

# ──────────────────────────────────────────────────────────────────────
#  PARALLELISM (override via env-vars if you want)
# ──────────────────────────────────────────────────────────────────────
N_WORKERS: int = int(os.getenv("N_WORKERS", "0")) or (os.cpu_count() or 1)
USE_PROCESSES: bool = bool(int(os.getenv("USE_PROCESSES", "0")))  # 0 = threads (default)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "50000"))  # rows per chunk
REPORT_EVERY: int = 100_000                               # status print cadence

# ──────────────────────────────────────────────────────────────────────
# DIRECTORY HOLDING CHECKPOINTS
#     `antibiotic_model_checkpoints/` alongside this script, holding the *.pt files.
#     Override with AM_MODEL_DIR only if the checkpoints live elsewhere.
# ──────────────────────────────────────────────────────────────────────
antibiotic_model_dir = os.getenv(
    "AM_MODEL_DIR",
    str(SCRIPT_DIR / "antibiotic_model_checkpoints"),
)

# ──────────────────────────────────────────────────────────────────────
#  Progress-bar helper (uses tqdm if available, else silent fallback)
# ──────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:  # fallback to no-op if tqdm is not installed

    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

    tqdm.write = print  # `tqdm.write` is used for status lines below

# ──────────────────────────────────────────────────────────────────────
#  COLUMN NAME FOR MINIMOL VECTORS
# ──────────────────────────────────────────────────────────────────────
MINIMOL_COL = "Minimol_Representation"   # change here if your TSV uses a different header

# =====================================================================
# 1)  ANTIBIOTIC-ACTIVITY PREDICTIONS  (original definition, unchanged)
# =====================================================================
def FNN_with_attention(
    # Data / IO
    train_representations: Optional[np.ndarray] = None,
    train_targets: Optional[np.ndarray] = None,
    test_representations: Optional[np.ndarray] = None,
    test_targets: Optional[np.ndarray] = None,
    score_data: Optional[List[dict]] = None,
    f_in_path: Optional[str] = None,
    f_out: Optional[str] = None,
    # Hyper-params
    loss_function: Optional[str] = None,
    batch_size: int = 300,
    hidden_dim: int = 1024,
    learning_rate: float = 0.1,
    n_iters: int = 1000,
    model_path: Optional[str] = None,
    num_layers: int = 2,
    dropout_prob: float = 0.1,
    mode: str = "train",  # {"train", "evaluate", "score"}
    patience: int = 10,
    lr_decay_step: int = 10,
    lr_decay_gamma: float = 0.1,
    model_type: str = "attention",  # {"attention", "mlp"}
    weight_decay: float = 1e-4,
    num_heads: int = 4,
):
    """
    Flexible feed-forward network with optional single-token self-attention.
    Only ``mode="score"`` is used by this pipeline.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ─────────────────────────────────────────────
    #  Internal model definitions
    # ─────────────────────────────────────────────
    class TaskHeadWithAttention(nn.Module):
        """MLP + skip connection + multi-head self-attention (single token)."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout_rate: float = 0.1,
            num_heads: int = 4,
        ):
            super().__init__()
            assert input_dim % num_heads == 0, "`input_dim` must be divisible by `num_heads`"

            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, batch_first=True
            )

            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.final_dense = nn.Linear(input_dim + hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            attn_out, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
            x = attn_out.squeeze(1)

            for dense, bn in zip(self.layers, self.batch_norms):
                x = self.dropout(F.relu(bn(dense(x))))

            return self.final_dense(torch.cat([residual, x], dim=1))

    class TaskHead(nn.Module):
        """Plain MLP with skip connection."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout_rate: float = 0.1,
        ):
            super().__init__()
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.final_dense = nn.Linear(input_dim + hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            for dense, bn in zip(self.layers, self.batch_norms):
                x = self.dropout(F.relu(bn(dense(x))))
            return self.final_dense(torch.cat([residual, x], dim=1))

    # ─────────────────────────────────────────────
    #  MODE: SCORE
    # ─────────────────────────────────────────────
    if mode == "score":
        if f_in_path is None or not os.path.exists(f_in_path):
            raise ValueError("'score' mode requires a valid `f_in_path`")
        if score_data is None:
            raise ValueError("'score' mode requires `score_data`")

        # ---------- load checkpoint & infer architecture -----------
        state_dict = torch.load(
            f_in_path, map_location="cpu", weights_only=True  # suppress pickle warning
        )
        layer0 = next(k for k in state_dict if k.endswith("layers.0.weight"))
        input_dim = state_dict[layer0].shape[1]
        hidden_dim_ckpt = state_dict[layer0].shape[0]
        output_dim = state_dict["final_dense.weight"].shape[0]
        num_layers_ckpt = len(
            {k.split(".")[1] for k in state_dict if k.startswith("layers.") and ".weight" in k}
        )

        if any("attention.in_proj_weight" in k for k in state_dict):
            model = TaskHeadWithAttention(
                input_dim,
                output_dim,
                hidden_dim_ckpt,
                num_layers=num_layers_ckpt,
                dropout_rate=dropout_prob,
                num_heads=num_heads,
            )
        else:
            model = TaskHead(
                input_dim,
                output_dim,
                hidden_dim_ckpt,
                num_layers=num_layers_ckpt,
                dropout_rate=dropout_prob,
            )

        model.load_state_dict(state_dict)
        model.eval()

        # ---------- prepare input ----------
        reps = [
            torch.as_tensor(d["featurized_smiles"], dtype=torch.float32)
            if not isinstance(d["featurized_smiles"], torch.Tensor)
            else d["featurized_smiles"].float()
            for d in score_data
        ]
        X = torch.stack(reps)
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

        preds: List[List[float]] = []
        for (batch_x,) in loader:
            with torch.no_grad():
                preds.extend(torch.sigmoid(model(batch_x)).cpu().tolist())

        # ---------- optional disk output ----------
        if f_out:
            with open(f_out, "w") as fh:
                fh.write("\t".join([f"pred_{i}" for i in range(output_dim)]) + "\n")
                for row in preds:
                    fh.write("\t".join(map(str, row)) + "\n")
            print(f"[activity] saved predictions → {f_out}")

        return preds

    raise NotImplementedError(f"Mode '{mode}' not implemented in this script.")

# ---------------------------------------------------------------------
#  Helper – safe vector parser  (replaces ``eval``)
# ---------------------------------------------------------------------
def safe_literal_vector(txt) -> Optional[List[float]]:
    """
    Return list[float] on success, else *None* (NaNs, bad JSON etc.).
    """
    if txt is None or (isinstance(txt, float) and np.isnan(txt)):
        return None
    if not isinstance(txt, str):
        return None
    try:
        return ast.literal_eval(txt)
    except Exception:
        try:
            return json.loads(txt)
        except Exception:
            return None

# ---------------------------------------------------------------------
#  ACTIVITY STAGE – streaming & multi-CPU
# ---------------------------------------------------------------------
def _score_model(model_file: str, score_data: list[dict], bs: int):
    preds = FNN_with_attention(
        mode="score", f_in_path=model_file, score_data=score_data, batch_size=bs
    )
    return model_file, preds


def stream_activity_on_tsv(
    raw_tsv: str,
    out_tsv: str,
    model_dir: str,
    chunk_size: int = CHUNK_SIZE,
    max_workers: int = N_WORKERS,
):
    model_files = sorted(glob(os.path.join(model_dir, "*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No .pt files found in {model_dir}")

    Exec = ProcessPoolExecutor if USE_PROCESSES else ThreadPoolExecutor

    rows_seen = rows_bad = 0
    header_written = False
    last_report = 0

    reader = pd.read_csv(raw_tsv, sep=",", dtype=str, chunksize=chunk_size)

    for df in tqdm(reader, desc="Activity | TSV chunks"):
        vecs = df[MINIMOL_COL].map(safe_literal_vector)
        good = vecs.notnull()
        bad_here = (~good).sum()
        rows_seen += len(df)
        rows_bad += bad_here
        if bad_here:
            df = df.loc[good].copy()
            vecs = vecs.loc[good]
        if df.empty:
            continue

        score_data = [{"featurized_smiles": v} for v in vecs]

        preds_by_model: dict[str, list[list[float]]] = {}
        with Exec(max_workers=min(max_workers, len(model_files))) as pool:
            futs = {
                pool.submit(
                    _score_model,
                    mf,
                    score_data,
                    min(8192, len(score_data)),
                ): mf
                for mf in model_files
            }
            for fut in as_completed(futs):
                mf, preds = fut.result()
                preds_by_model[mf] = preds

        for mf, preds in preds_by_model.items():
            base = Path(mf).stem
            if len(preds[0]) == 1:
                df[f"{base}_output"] = [p[0] for p in preds]
            else:
                for i in range(len(preds[0])):
                    df[f"{base}_output_{i}"] = [p[i] for p in preds]

        df.to_csv(
            out_tsv,
            sep="\t",
            index=False,
            mode="w" if not header_written else "a",
            header=not header_written,
        )
        header_written = True

        if rows_seen - last_report >= REPORT_EVERY:
            tqdm.write(
                f"[activity] {rows_seen:,} rows  |  dropped {rows_bad:,} "
                f"({rows_bad / rows_seen:.2%})"
            )
            last_report = rows_seen

# ---------------------------------------------------------------------
#  STREAMED SCORE AGGREGATION  – memory-safe
# ---------------------------------------------------------------------
def add_activity_scores_streamed(
    tsv_in: Union[str, Path],
    tsv_out: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    na_strategy: str = "skip",
):
    """
    Read *tsv_in* in chunks, append per-species mean activity columns,
    write to *tsv_out*.  Uses O(chunk_size) memory.
    """
    species_prefix = {
        "Acinetobacter": "ab_am_mtl_rep_{i}_col_1_is_pred_output_0",
        "Escherichia":   "ec_am_mtl_rep_{i}_col_2_is_pred_output_1",
        "Klebsiella":    "kp_am_mtl_rep_{i}_col_3_is_pred_output_2",
        "Pseudomonal":   "pa_am_mtl_rep_{i}_col_4_is_pred_output_3",
    }

    header_written = False
    reader = pd.read_csv(tsv_in, sep="\t", chunksize=chunk_size, dtype=str)

    for chunk in tqdm(reader, desc="Aggregate | TSV chunks"):
        # convert only once per chunk – faster & less RAM
        chunk = chunk.apply(pd.to_numeric, errors="ignore")

        # Accept columns from both lowercase and older uppercase checkpoint names.
        columns_by_lower = {column.lower(): column for column in chunk.columns}

        # 1) activity means
        for sp, pat in species_prefix.items():
            cols = [columns_by_lower[pat.format(i=r)] for r in (1, 2, 3)
                    if pat.format(i=r) in columns_by_lower]
            if cols:
                chunk[f"Mean_{sp}_Activity"] = chunk[cols].astype(float).mean(
                    axis=1, skipna=True
                )
            elif na_strategy == "raise":
                raise KeyError(f"Missing columns for {sp}: {pat.format(i=1)} …")

        # 2) flush
        chunk.to_csv(
            tsv_out,
            sep="\t",
            index=False,
            mode="w" if not header_written else "a",
            header=not header_written,
        )
        header_written = True

    tqdm.write(f"[aggregate] complete → {tsv_out}")


# =====================================================================
#  MAIN PIPELINE
# =====================================================================
if __name__ == "__main__":

    import time
    start_time = time.perf_counter()

    # 1) activity predictions → ACTIVITY_TSV
    stream_activity_on_tsv(
        raw_tsv=RAW_TSV,
        out_tsv=ACTIVITY_TSV,
        model_dir=antibiotic_model_dir,
        chunk_size=CHUNK_SIZE,
        max_workers=N_WORKERS,
    )

    # 2) aggregate activity scores → FINAL_TSV
    add_activity_scores_streamed(
        tsv_in=ACTIVITY_TSV,
        tsv_out=FINAL_TSV,
        chunk_size=CHUNK_SIZE,
    )

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")

    print(f"✔ Activity scoring done – results saved in: {FINAL_TSV}")
