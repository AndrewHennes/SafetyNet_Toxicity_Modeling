# Score the example molecules

The example contains ten molecules encoded with the Simplified Molecular Input Line Entry System (SMILES). It exercises MiniMol featurization, the antibiotic activity ensemble, and the SafetyNet toxicity ensemble.

## Setup

Use an environment with the dependencies in `pyproject.toml`, including a working MiniMol installation. MiniMol may download pretrained weights on first use. The model checkpoints supplied with this repository are stored through Git Large File Storage (LFS).

Set the absolute checkout path once. The commands below then work from any directory.

```bash
SAFETYNET_ROOT="/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private"
python -m pip install -e "${SAFETYNET_ROOT}"
git -C "${SAFETYNET_ROOT}" lfs pull
bash "${SAFETYNET_ROOT}/examples/run_example_score.sh"
```

The helper uses the active Python environment and runs toxicity inference on the CPU. It writes `outputs/example_scored.tsv` beneath the checkout. It does not modify the input molecules or model weights.

## Output

Each input row is retained. The output includes the input metadata, `Minimol_Fingerprint`, and these summary columns.

| Column | Meaning |
| --- | --- |
| `Mean_Acinetobacter_Activity` | Mean selected output across the corresponding antibiotic checkpoints |
| `Mean_Escherichia_Activity` | Mean selected output across the corresponding antibiotic checkpoints |
| `Mean_Klebsiella_Activity` | Mean selected output across the corresponding antibiotic checkpoints |
| `Mean_Pseudomonal_Activity` | Mean selected output across the corresponding antibiotic checkpoints |
| `Mean_HepG2_Toxicity` | Mean toxicity probability across all SafetyNet checkpoints |
| `Mean_HSkMC_Toxicity` | Mean toxicity probability across all SafetyNet checkpoints |
| `Mean_IMR90_Toxicity` | Mean toxicity probability across all SafetyNet checkpoints |
| `Max_Tox_Score` | Maximum of the three mean toxicity probabilities |
| `Non_Toxic_Score` | One minus `Max_Tox_Score` |

Per-checkpoint columns are omitted by default. The existing `--keep-ensemble-columns` option retains them. Existing representation columns are reused after validation, so precomputed embeddings do not require MiniMol inference.

```python
from pathlib import Path
import pandas as pd

repository = Path("/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private")
results = pd.read_csv(repository / "outputs/example_scored.tsv", sep="\t")
print(results.head())
print(results.columns.tolist())
```

The script validates SMILES, representations, checkpoint families, and prediction-column names before scoring. Outputs are model prediction scores.
