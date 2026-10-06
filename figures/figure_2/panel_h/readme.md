# Figure 2h: ClinTox performance and novelty filtering

Recomputes all eight structural-alert catalogs and evaluates the primary assumed achiral Morgan radius-2, 2,048-bit novelty filter at maximum Tanimoto similarity ≤0.9. The complete-cohort and chirality-enabled analyses remain explicit sensitivity checks.

Molecular cohorts, labels, scores, computed alerts, similarity calculations, CXSMILES format-repair records, exclusion records, metric tables and plots are all saved here. The similarity cache is reused only when source hashes, fingerprint settings and RDKit version match.

The primary cohort contains 990 compounds and 46 toxicity positives. SafetyNet Matthews correlation coefficient (MCC), F1, balanced accuracy and precision are 0.157308, 0.176678, 0.659451 and 0.105485.

Run this panel independently from any working directory:

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/figure_2/panel_h/figure_2h.py
```

The runner accepts no command-line options and changes only this panel’s results and any required shared inputs. The scientific formulas and cohort policies are unchanged by the directory reorganization.
