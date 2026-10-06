# Supplementary Figure 2a: Tox21 assays

This runner recalculates all assay metrics from molecular outcomes and prediction scores. It produces an initial unfiltered plot and the primary maximum-similarity ≤0.9 plot. The 42 evaluable assays are identified from outcome-class availability rather than a precomputed performance table.

This directory contains its molecular cohort, per-assay exclusions, alternative-policy metrics, provenance, plot files and validation. The scientific calculation library is loaded from the absolute location resolved from `../../figure_2/shared/analysis.py`; the current working directory is irrelevant. Panel b is not executed by this runner.

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/supplementary_figure_2/panel_a/supplementary_figure_2a.py
```
