# Figure 2f: Tox21 assay performance

Recomputes assay metrics directly from molecular labels and three saved task scores. Both the initial unfiltered analysis and the methods-style maximum training similarity ≤0.9 analysis are produced. Four single-class assays are excluded from area under the receiver operating characteristic curve (auROC) calculations, yielding 42 evaluable assays.

All Tox21 molecular rows, assay labels, score and inclusion fields, per-assay exclusions, metrics under alternative policies, provenance and plots are saved here. Calculations come from `../shared/analysis.py`.

Run this panel independently from any working directory:

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/figure_2/panel_f/figure_2f.py
```

The runner accepts no command-line options and changes only this panel’s results and any required shared inputs. The scientific formulas and cohort policies are unchanged by the directory reorganization.
