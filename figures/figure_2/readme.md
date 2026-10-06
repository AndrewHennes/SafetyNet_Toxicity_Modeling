# Figure 2 data analysis

`figure_2.py` computes panels d, e, f and h from molecular assay labels, molecular structures and supplied model prediction scores. No original SVG geometry, manuscript metric table or precomputed metric summary is an analysis input. `figure.svg` and `source_preview.png` remain reference artwork only.

Run from any working directory:

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/figure_2/figure_2.py
```

The code has no command-line options. `shared/analysis.py` supplies the calculations shared with Supplementary Figure 2. Inputs are resolved to absolute repository paths. Each panel has its own `panel_a` through `panel_h` directory containing its runner, notes and results. The figure-level runner calls those panel runners in order. Shared code and cross-panel audits are in `shared/`. The workflow requires Python, NumPy, pandas, matplotlib, scikit-learn and RDKit. The first ClinTox overlap calculation takes roughly a minute on the available machine. Its cache is reused only when source hashes, RDKit version and fingerprint settings match.

## Panel directories

| Directory | Contents |
|---|---|
| `panel_a/` | Schematic status and notes |
| `panel_b/` | Internal benchmark input requirements and status runner |
| `panel_c/` | External benchmark input requirements and status runner |
| `panel_d/` | ECBD ROC/precision-recall analysis |
| `panel_e/` | ECBD structural-filter analysis |
| `panel_f/` | Tox21 assay analysis |
| `panel_g/` | Animal endpoint input requirements and status runner |
| `panel_h/` | ClinTox novelty and filter analysis |
| `shared/` | Scientific code, common ECBD inputs and figure-wide audits |

Each panel’s figure-specific script can be run independently. Schematic and input-dependent panels write status reports. All paths are resolved from the saved script location, so the working directory does not matter.

## What is computed

- **Panel d** calculates receiver operating characteristic (ROC) and precision-recall curves directly from ECBD labels and scores. The declared primary candidate is the restricted ECBD table with native labels and stored maximum training similarity ≤0.9, retaining 35,964 compounds and 322 positives. auROC is 0.924213, average precision is 0.403710, and trapezoidal precision-recall area is 0.402999. These distinct metrics are saved separately.
- **Panel e** evaluates SafetyNet and eight structural-alert catalogs on the same ECBD candidate cohort. Every alert, including Brenk, is recalculated from the full submitted molecular structure with RDKit. Supplied flags are compared separately.
- **Panel f** calculates assay metrics from the 7,973-row Tox21 table. Of 46 assay columns, four contain only positives and cannot define auROC. Requiring both classes recovers 42 assays without reading the 42-row summary table. The methods-style similarity filter retains 6,040 molecules before per-assay missing-label exclusion. The resulting auROC range is 0.572905–0.885160. Both unfiltered and filtered plots are produced.
- **Panel h** evaluates ClinTox labels, SafetyNet classifications, and eight structural catalogs. The primary analysis uses achiral Morgan radius-2, 2,048-bit fingerprints and maximum Tanimoto similarity ≤0.9, retaining 990 compounds with 46 positives. Full-cohort and chirality-enabled analyses are saved as sensitivity checks. The fingerprint parameters are the configured analysis settings.

The ClinTox reference contains the 100,352 supplied internal rows. Exact duplicate raw SMILES are evaluated once for efficiency without changing a maximum-similarity result. A documented format repair inserts the missing space before CXSMILES extensions, preserving the graph and all extension metadata. Seventy-two unique structures are recovered by this format repair, and four remain invalid or unspecified. Repaired and excluded rows are listed individually. The repaired reference removes one additional ClinTox compound compared with the initial uncorrected-reference calculation, changing the primary cohort from 991 to 990. The analysis does not strip salts or make silent chemical changes.

## Cohort and ensemble analyses

1. Both full-cohort and novelty-filtered Tox21 evaluations are saved with their cohort membership and per-assay metrics.
2. ECBD was tested across all three provided source versions, their native labels, explicit growth-inhibition thresholds of >20% and >50%, available overlap policies, cell-specific scores and score-aggregation rules. Native labels in the 101,097-row table exactly implement >20% growth inhibition. Moving to >50% reduces positives from 5,031 to 776 and changes auROC/average precision from 0.766055/0.307558 to 0.924863/0.464907.
3. The saved ECBD prediction table contains six ensemble members. The stored maximum score equals the maximum across per-task ensemble means. Ensemble members are evaluated as saved predictions, separately from benchmark folds.

## Audit outputs

Results are grouped by panel. `panel_d/` holds ECBD curves, `panel_e/` holds the ECBD structural-filter analysis, `panel_f/` holds Tox21 data/results and `panel_h/` holds ClinTox data/results. Shared ECBD cohorts and version/label/ensemble audits are in `shared/ecbd/`. Together these directories preserve every source row identifier, molecular cohort, label, score and derived inclusion flag required to audit the calculations. `source_row` is zero-based within the data body, excluding the header; `row_id` combines the source dataset key and that index. Molecular SMILES are also retained. Per-assay missing labels remain missing rather than being turned into negatives.

Key tables include `tox21_metrics_all_policies_and_scores.csv`, `ecbd_cohort_score_label_sensitivity.csv`, `ecbd_label_definition_audit.csv`, `ecbd_input_version_comparison.csv`, `ecbd_saved_member_and_aggregation_metrics.csv`, `panel_e_ecbd_filter_metrics.csv`, `panel_h_clintox_filter_metrics.csv` (primary achiral novelty-filtered cohort), `clintox_unfiltered_filter_metrics.csv` and `clintox_recomputed_overlap_filter_metrics.csv`. Source hashes, catalog versions, policy choices and panel dependencies are recorded beside them.

`shared/test_analysis.py` checks a hand-calculated ROC/precision-recall example, confusion counts, exact cutoff ties, missing labels, single-class assays, invalid scores and similarity boundaries. It also checks that maximum-of-means and mean-of-maxima can differ.

Panels b and c require fold assignments, baseline model predictions, and training/evaluation configurations. Panel g requires compound-level animal toxicity observations and predictions. Input requirements are listed in `missing_inputs.md`.
