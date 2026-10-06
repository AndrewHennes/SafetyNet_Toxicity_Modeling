#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Toxicity inference pipeline – streamed, memory-safe
Created on Sat Feb 15 12:22:59 2025
@author: asselism

Split out of ScoreToxAndAM.py; the activity half lives in
score_with_antibiotic_models.py.  This script can be run on that script's output
(to reproduce the original combined pipeline) or on any TSV that carries a
Minimol representation column.

Requires ``model.py`` (defines ``MultiTaskFeedForward``) to be
importable – the simplest option is to keep it in the same directory as this
script, since Python puts the script's directory on ``sys.path``.

Activate environment:
    source /Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/activate
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
#     Every path below falls back to an environment variable first, so a
#     user of this repo can run the script without editing it:
#         TOX_INPUT=... TOX_MODEL_DIR=... python score_with_toxicity_models.py
# ──────────────────────────────────────────────────────────────────────
# Directory this file lives in.  The `safetynet_checkpoints` checkpoint folder
# is expected to sit right next to this script, so the script moves with its
# checkpoints and works from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_TSV = os.getenv(
    "TOX_INPUT",
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Revision_Related/300K_Analyses/SA_Related/SA1_with_Minimol_Representations_with_AM.tsv",
)  # ← input (tab-separated; e.g. the output of score_with_antibiotic_models.py)

TOX_TSV = os.getenv(
    "TOX_PRED_OUTPUT",
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Revision_Related/300K_Analyses/SA_Related/SA1_with_Minimol_Representations_int_tox.tsv",
)  # ← intermediate: one column per ensemble member per task

FINAL_TSV = os.getenv(
    "TOX_OUTPUT",
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Revision_Related/300K_Analyses/SA_Related/SA1_with_Minimol_Representations_with_Tox_AM.tsv",
)  # ← final output: per-cell-line mean toxicity, max tox, non-toxic score

# ──────────────────────────────────────────────────────────────────────
#  CHUNKING (override via env-vars if you want)
# ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "50000"))  # rows per chunk
REPORT_EVERY: int = 100_000                               # status print cadence

# ──────────────────────────────────────────────────────────────────────
# DIRECTORY HOLDING CHECKPOINTS
#     `safetynet_checkpoints/` alongside this script, holding the *.ckpt files.
#     Override with TOX_MODEL_DIR only if the checkpoints live elsewhere.
# ──────────────────────────────────────────────────────────────────────
tox_models_dir = os.getenv(
    "TOX_MODEL_DIR",
    str(SCRIPT_DIR / "safetynet_checkpoints"),
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
#  TOXICITY STAGE – streamed, memory-safe
# ---------------------------------------------------------------------
def stream_tox_on_tsv(
    input_tsv: str,
    output_tsv: str,
    ckpt_dir: str,
    rep_col: str = MINIMOL_COL,
    chunk_size: int = CHUNK_SIZE,
    device: str = "cpu",
):
    ckpts = sorted(Path(ckpt_dir).glob("*.ckpt"))
    print(ckpts)

    if not ckpts:
        warnings.warn("[tox] no checkpoints found – skipping")
        Path(output_tsv).write_text(Path(input_tsv).read_text())
        return

    tmp_out = Path(output_tsv).with_suffix(".tox_tmp.tsv")
    rows_seen = rows_bad = 0
    header_written = False
    last_report = 0

    reader = pd.read_csv(input_tsv, sep="\t", dtype=str, chunksize=chunk_size)

    for df in tqdm(reader, desc="Tox | TSV chunks"):
        vecs = df[rep_col].map(safe_literal_vector)
        good = vecs.notnull()
        bad_here = (~good).sum()
        rows_seen += len(df)
        rows_bad += bad_here
        if bad_here:
            df = df.loc[good].copy()
            vecs = vecs.loc[good]
        if df.empty:
            continue

        reps = torch.tensor(np.stack(vecs), dtype=torch.float32).to(device)

        for idx, ckpt in enumerate(ckpts):
            if __package__:
                from .model import MultiTaskFeedForward
            else:
                from model import MultiTaskFeedForward
            model = (
                MultiTaskFeedForward.load_from_checkpoint(str(ckpt))
                .to(device)
                .eval()
            )
            preds_parts = []
            bs = 1024
            with torch.no_grad():
                for i in range(0, len(reps), bs):
                    preds_parts.append(torch.sigmoid(model(reps[i : i + bs])).cpu())
            preds = torch.cat(preds_parts).numpy()
            for t in range(preds.shape[1]):
                df[f"Ensemble_Member_{idx}_prediction_task_{t}"] = preds[:, t]
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        df.to_csv(
            tmp_out,
            sep="\t",
            index=False,
            mode="w" if not header_written else "a",
            header=not header_written,
        )
        header_written = True

        if rows_seen - last_report >= REPORT_EVERY:
            tqdm.write(
                f"[tox] {rows_seen:,} rows  |  dropped {rows_bad:,} "
                f"({rows_bad / rows_seen:.2%})"
            )
            last_report = rows_seen

    if not header_written:
        raise ValueError("No valid representations were available for toxicity scoring.")
    tmp_out.replace(Path(output_tsv))
    tqdm.write(
        f"[tox] done – total rows: {rows_seen:,}   dropped: {rows_bad:,} "
        f"({rows_bad / rows_seen:.2%})"
    )

# ---------------------------------------------------------------------
#  STREAMED SCORE AGGREGATION  – memory-safe
# ---------------------------------------------------------------------
def add_tox_scores_streamed(
    tsv_in: Union[str, Path],
    tsv_out: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    na_strategy: str = "skip",
):
    """
    Read *tsv_in* in chunks, append per-cell-line mean toxicity, max toxicity
    and non-toxic columns, write to *tsv_out*.  Uses O(chunk_size) memory.
    """
    tox_tasks = {0: "HepG2", 1: "HSkMC", 2: "IMR90"}

    header_written = False
    reader = pd.read_csv(tsv_in, sep="\t", chunksize=chunk_size, dtype=str)

    for chunk in tqdm(reader, desc="Aggregate | TSV chunks"):
        # Convert prediction columns only, preserving identifiers and metadata.

        # 1) toxicity means
        for task_id, cell in tox_tasks.items():
            cols = [
                column for column in chunk.columns
                if re.fullmatch(rf"Ensemble_Member_\d+_prediction_task_{task_id}", column)
            ]
            if cols:
                chunk[f"Mean_{cell}_Toxicity"] = chunk[cols].astype(float).mean(
                    axis=1, skipna=True
                )
            elif na_strategy == "raise":
                raise KeyError(f"Missing Tox columns task {task_id}")

        # 2) per-row max tox & non-tox
        tox_mean_cols = [c for c in chunk.columns if c.startswith("Mean_") and c.endswith("_Toxicity")]
        if tox_mean_cols:
            chunk["Max_Tox_Score"] = chunk[tox_mean_cols].max(axis=1, skipna=True)
            chunk["Non-toxic Score"] = 1 - chunk["Max_Tox_Score"]

        # 3) flush
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

    # 1) toxicity ensemble (streamed) → TOX_TSV
    stream_tox_on_tsv(
        input_tsv=INPUT_TSV,
        output_tsv=TOX_TSV,
        ckpt_dir=tox_models_dir,
        rep_col=MINIMOL_COL,
        chunk_size=CHUNK_SIZE,
        device="cpu",
    )

    # 2) aggregate tox scores → FINAL_TSV
    add_tox_scores_streamed(
        tsv_in=TOX_TSV,
        tsv_out=FINAL_TSV,
        chunk_size=CHUNK_SIZE,
    )

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")

    print(f"✔ Toxicity scoring done – results saved in: {FINAL_TSV}")
