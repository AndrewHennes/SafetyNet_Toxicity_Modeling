# Analysis utilities

These utilities operate on molecular tables and representation arrays. Relative paths are resolved against the repository, not the shell's current directory. Use absolute paths for inputs and outputs outside the checkout.

The analysis environment requires NumPy, pandas, scikit-learn 1.5 or newer, Matplotlib, and RDKit for molecular fingerprints. The repository's editable installation exposes the featurization modules through the `safetynet` package.

## Morgan fingerprints

`models/featurization/compute_morgan_fingerprints.py` computes extended-connectivity fingerprints (ECFP) using RDKit's Morgan generator. Defaults are radius 2, 2048 bits, and chirality enabled.

```bash
SAFETYNET_ROOT="/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private"
python -m safetynet.featurization.compute_morgan_fingerprints \
  --data-dir "${SAFETYNET_ROOT}/examples" \
  --inputs example_smiles.csv
```

Outputs go to `outputs/fingerprints/` beneath the checkout. The NumPy archive contains `X`, `smiles`, `names`, zero-based `row_indices`, and JSON metadata. Invalid molecules and wildcard atoms are excluded and counted. Source hashes, fingerprint settings, column candidates, and the RDKit version are checked before an existing cache is reused. Use `--overwrite` to recompute a stale cache.

## Two-dimensional representation plots

`other/tsne_from_representation.py` creates exploratory t-distributed stochastic neighbor embedding (t-SNE) plots, with principal component analysis (PCA) initialization, a fixed default random seed, and Euclidean distances.

```bash
python "${SAFETYNET_ROOT}/other/tsne_from_representation.py" \
  --mode npz \
  --data-dir "${SAFETYNET_ROOT}/outputs/fingerprints" \
  --npz-files example_smiles_ecfp2048_r2.npz \
  --perplexity 3 \
  --output "${SAFETYNET_ROOT}/outputs/example_tsne.npz"
```

NumPy input supports both `X` arrays from Morgan fingerprints and `features` arrays from MiniMol. Table mode uses `--mode table --tables FILE --rep-col Minimol_Fingerprint` with an optional `--sep`. All files must use matching representation widths. Perplexity must be smaller than the number of points. The existing `--n-iter` argument is passed to scikit-learn as `max_iter`.

Output arrays are `Y` (coordinates), `y` (source dataset index), `label_names`, `counts`, and `row_indices` (zero-based source data rows). Metadata records the parameters and source paths. Archives load with `allow_pickle=False`. Invalid table vectors are excluded with a warning and retained row mappings; inconsistent vector lengths cause an error.

## Activity versus toxicity

```bash
python "${SAFETYNET_ROOT}/other/plot_tox_vs_activity.py" \
  --input "${SAFETYNET_ROOT}/outputs/example_scored.tsv" \
  --activity-col Mean_Klebsiella_Activity \
  --tox-col Max_Tox_Score \
  --activity-threshold 0.5 \
  --tox-threshold 0.2 \
  --out-prefix "${SAFETYNET_ROOT}/outputs/plots/example_kp_vs_tox"
```

The script saves PNG and SVG figures and reports the number of compounds in each quadrant. Equality belongs to the high-score side of each threshold. Missing, nonnumeric, and infinite values are excluded and counted. Thresholds are configured for the analysis.

## Binary model evaluation

`other/evaluate_model_performance.py` computes the area under the receiver operating characteristic curve (ROC AUC) and average precision (AP), and saves a combined ROC and precision-recall figure plus a metrics JSON file.

```bash
python "${SAFETYNET_ROOT}/other/evaluate_model_performance.py" \
  --input "${SAFETYNET_ROOT}/outputs/labelled_predictions.tsv" \
  --label-col HepG2_Binarized_Toxicity \
  --score-col Pred_HepG2 \
  --pos-label 1 \
  --out-prefix "${SAFETYNET_ROOT}/outputs/metrics/hepg2"
```

This command requires your own labelled prediction table. Exactly two observed numeric classes must remain after excluding nonfinite rows, and one must match `--pos-label`. Scores can be probabilities or continuous decision scores. AP is reported explicitly rather than described as trapezoidal precision-recall area. Outputs are `<prefix>_roc_prc.png` and `<prefix>_metrics.json`; dots in the prefix are preserved.
