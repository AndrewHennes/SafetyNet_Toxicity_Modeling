"""Stream molecular tables through antibiotic and toxicity ensembles."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from safetynet.cli._common import checkpoint_order, predict_probabilities
from safetynet.featurization.table_utils import (
    positive_int,
    representation_matrix,
    require_distinct_paths,
    resolve_path,
    separator,
)

SPECIES = {1: "Acinetobacter", 2: "Escherichia", 3: "Klebsiella", 4: "Pseudomonal"}
CELL_LINES = ("HepG2", "HSkMC", "IMR90")


def _score_antibiotic(path, features):
    """Score one checkpoint; top-level placement supports process workers."""
    from safetynet.antibiotic.score_with_antibiotic_models import FNN_with_attention

    data = [{"featurized_smiles": row} for row in features]
    predictions = FNN_with_attention(
        f_in_path=str(path), score_data=data, mode="score", batch_size=1024
    )
    array = np.asarray(predictions)
    if array.shape != (len(features), len(SPECIES)) or not np.isfinite(array).all():
        raise ValueError(f"Unexpected predictions from {path.name}: {array.shape}")
    return path, array


def aggregate_predictions(
    frame, antibiotic_predictions, toxicity_predictions, keep_columns
):
    """Average every available ensemble member and preserve original metadata."""
    additions = {}
    by_species = {column: [] for column in SPECIES}
    for checkpoint, probabilities in antibiotic_predictions:
        match = re.search(r"_col_(\d+)_is_pred$", checkpoint.stem)
        if match is None or int(match[1]) not in SPECIES:
            raise ValueError(
                f"Unrecognized antibiotic checkpoint name: {checkpoint.name}"
            )
        column = int(match[1])
        by_species[column].append(probabilities[:, column - 1])
        if keep_columns:
            for task in range(probabilities.shape[1]):
                additions[f"{checkpoint.stem}_output_{task}"] = probabilities[:, task]
    for column, species in SPECIES.items():
        if not by_species[column]:
            raise ValueError(f"No antibiotic checkpoints available for {species}.")
        additions[f"Mean_{species}_Activity"] = np.mean(by_species[column], axis=0)
    if not toxicity_predictions:
        raise ValueError("No toxicity predictions were supplied.")
    for index, probabilities in enumerate(toxicity_predictions):
        if probabilities.shape != (len(frame), len(CELL_LINES)):
            raise ValueError(
                "Toxicity checkpoints must predict the three documented cell lines."
            )
        if keep_columns:
            for task in range(len(CELL_LINES)):
                additions[f"Ensemble_Member_{index}_prediction_task_{task}"] = (
                    probabilities[:, task]
                )
    means = np.mean(toxicity_predictions, axis=0)
    for task, cell in enumerate(CELL_LINES):
        additions[f"Mean_{cell}_Toxicity"] = means[:, task]
    additions["Max_Tox_Score"] = means.max(axis=1)
    additions["Non_Toxic_Score"] = 1 - means.max(axis=1)
    if set(frame) & set(additions):
        raise ValueError("The input already contains prediction or summary columns.")
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end SafetyNet scoring: SMILES → MiniMol → ABX + tox + summary scores."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input CSV/TSV with a SMILES column.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Final scored TSV output path.",
    )
    p.add_argument(
        "--smiles-col",
        default="SMILES",
        help="SMILES column name in the input file (default: SMILES).",
    )
    p.add_argument(
        "--sep",
        default=",",
        help="Delimiter for the input file (default: ',').",
    )
    p.add_argument(
        "--abx-model-dir",
        required=True,
        help="Directory containing ABX FNN `.pt` models (models/antibiotic_activity_model/antibiotic_model_checkpoints).",
    )
    p.add_argument(
        "--tox-model-dir",
        required=True,
        help="Directory containing toxicity `.ckpt` models (models/safetynet/safetynet_checkpoints).",
    )
    p.add_argument(
        "--rep-col",
        default="Minimol_Fingerprint",
        help="Name of MiniMol representation column (default: Minimol_Fingerprint).",
    )
    p.add_argument(
        "--chunk-size",
        type=positive_int,
        default=50_000,
        help="Chunk size for streaming (rows per chunk, default: 50,000).",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Number of workers for ABX scoring (0 = os.cpu_count()).",
    )
    p.add_argument(
        "--use-processes",
        action="store_true",
        help="Use processes instead of threads for ABX ensemble.",
    )
    p.add_argument(
        "--tox-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for toxicity inference (default: cpu).",
    )
    p.add_argument(
        "--keep-ensemble-columns",
        action="store_true",
        help=(
            "If set, keep per-model columns (Ensemble_Member_* and *_is_pred_output_*) "
            "in the final scored file instead of only summary scores."
        ),
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the intermediate featurized table beside the output.",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    if args.n_workers < 0:
        raise ValueError("--n-workers must be nonnegative.")
    source, output = resolve_path(args.input), resolve_path(args.output)
    require_distinct_paths(source, output)
    antibiotic_dir, toxicity_dir = (
        resolve_path(args.abx_model_dir),
        resolve_path(args.tox_model_dir),
    )
    antibiotic_files = sorted(antibiotic_dir.glob("*.pt"))
    toxicity_files = sorted(toxicity_dir.glob("*.ckpt"), key=checkpoint_order)
    if not antibiotic_files or not toxicity_files:
        raise ValueError(
            "Both checkpoint directories must contain downloaded model files."
        )
    columns = {
        int(match[1])
        for path in antibiotic_files
        if (match := re.search(r"_col_(\d+)_is_pred$", path.stem))
    }
    if columns != set(SPECIES):
        raise ValueError(
            "Antibiotic checkpoints must cover prediction columns 1 through 4."
        )
    import torch
    from safetynet.featurization.minimol_featurization import MinimolFeaturizer
    from safetynet.toxicity.model import MultiTaskFeedForward

    device = args.tox_device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = "cpu"
    toxicity_models = [
        MultiTaskFeedForward.load_from_checkpoint(str(path), map_location="cpu")
        .to(device)
        .eval()
        for path in toxicity_files
    ]
    if any(model.output_dim != len(CELL_LINES) for model in toxicity_models):
        raise ValueError("This ensemble interface expects three toxicity endpoints.")
    featurizer = MinimolFeaturizer()
    workers = min(args.n_workers or (os.cpu_count() or 1), len(antibiotic_files))
    executor_type = ProcessPoolExecutor if args.use_processes else ThreadPoolExecutor
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="safetynet_score_", dir=output.parent))
    result_path = temporary / "scored.tsv"
    rows = 0
    try:
        with executor_type(max_workers=workers) as executor:
            for frame in pd.read_csv(
                source,
                sep=separator(source, args.sep),
                chunksize=args.chunk_size,
                dtype=str,
                keep_default_na=False,
            ):
                if frame.empty:
                    continue
                if args.rep_col not in frame:
                    if args.smiles_col not in frame:
                        raise ValueError(f"Missing SMILES column {args.smiles_col!r}.")
                    features = featurizer.embed(frame[args.smiles_col].tolist())
                    frame[args.rep_col] = [
                        json.dumps(row.tolist(), allow_nan=False) for row in features
                    ]
                else:
                    features = representation_matrix(frame[args.rep_col])
                futures = [
                    executor.submit(_score_antibiotic, path, features)
                    for path in antibiotic_files
                ]
                antibiotic_predictions = [future.result() for future in futures]
                toxicity_predictions = [
                    predict_probabilities(model, features, 1024, device)
                    for model in toxicity_models
                ]
                scored = aggregate_predictions(
                    frame,
                    antibiotic_predictions,
                    toxicity_predictions,
                    args.keep_ensemble_columns,
                )
                scored.to_csv(
                    result_path,
                    sep="\t",
                    index=False,
                    mode="a" if rows else "w",
                    header=rows == 0,
                )
                if args.keep_temp:
                    frame.to_csv(
                        temporary / "featurized.tsv",
                        sep="\t",
                        index=False,
                        mode="a" if rows else "w",
                        header=rows == 0,
                    )
                rows += len(frame)
        if rows == 0:
            raise ValueError("The input table contains no rows.")
        os.replace(result_path, output)
    finally:
        if args.keep_temp:
            print(f"Intermediate files: {temporary}")
        else:
            shutil.rmtree(temporary)
    print(f"Wrote {rows} scored compounds to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
