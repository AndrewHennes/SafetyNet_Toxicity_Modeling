#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Use the active Python environment, with this repository installed editable.
python -m safetynet.cli.safetynet_score \
  --input "${REPOSITORY_ROOT}/examples/example_smiles.csv" \
  --output "${REPOSITORY_ROOT}/outputs/example_scored.tsv" \
  --smiles-col SMILES \
  --sep ',' \
  --rep-col Minimol_Fingerprint \
  --abx-model-dir "${REPOSITORY_ROOT}/models/antibiotic_activity_model/antibiotic_model_checkpoints" \
  --tox-model-dir "${REPOSITORY_ROOT}/models/safetynet/safetynet_checkpoints" \
  --chunk-size 256 \
  --n-workers 2 \
  --tox-device cpu
