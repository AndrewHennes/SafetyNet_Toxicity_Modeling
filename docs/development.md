# Development and validation

The installed Python package is `safetynet`. Setuptools maps its modules onto the organized source directories.

| Import namespace | Source directory |
| --- | --- |
| `safetynet.cli` | `models/cli_code` |
| `safetynet.featurization` | `models/featurization` |
| `safetynet.toxicity` | `models/safetynet` |
| `safetynet.antibiotic` | `models/antibiotic_activity_model` |

Install the checkout in editable mode before running its Python modules. Setuptools metadata is maintained in `pyproject.toml`; `setup.py` is only a compatibility entry point. Checkpoints and datasets are not bundled into the Python wheel.

```bash
SAFETYNET_ROOT="/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private"
python -m pip install -e "${SAFETYNET_ROOT}"
python -m unittest discover -s "${SAFETYNET_ROOT}/tests" -v
```

The regression suite uses temporary fixtures to verify input validation, delimiter handling, aggregate scores, cache integrity, model configuration, and small file-based workflows. MiniMol uses a deterministic test double in unit tests.

## Model interfaces

The network definition lives in `models/safetynet/model.py`. Training, checkpoint loading, and standalone toxicity scoring use that shared definition. Ray Tune dependencies are needed only for `train_safetynet.py`; install the `tune` extra to run that search.

The general training interface accepts binary labels, blank values, or `-1` for missing endpoints. It excludes rows with no observed labels, applies the requested layer widths and optimizer settings, and evaluates the best validation-loss checkpoint on the held-out test split. The separate Ray Tune search retains its own validation-average-precision selection policy.

Prediction validates representation dimensions before inference. Scoring averages every available ensemble member, preserving metadata and row order. The ensemble interface expects three toxicity outputs in the order HepG2, HSkMC, and IMR90. Confirm that convention before substituting different checkpoints.

## Historical notes

The dated repository-audit files are historical records. Current workflows are documented above.
