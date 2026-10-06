# Figure 2d: ECBD receiver operating characteristic and precision-recall curves

Recalculates curves from the declared restricted ECBD cohort and its saved molecular prediction scores. The primary rule is native binary HepG2 labels and maximum training similarity ≤0.9.

Curve coordinates, `panel_d_ecbd_roc_pr.svg`, its PNG preview and panel status/validation files are saved here. Shared ECBD cohort, input-version, label-definition and ensemble-aggregation audits are in `../shared/ecbd/`.

Run this panel independently from any working directory:

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/figure_2/panel_d/figure_2d.py
```

The runner accepts no command-line options and changes only this panel’s results and any required shared inputs. The scientific formulas and cohort policies are unchanged by the directory reorganization.
