# Figure 3f · E. coli structural-filter comparison

Run `figure_3f.py` directly from any working directory. It has no command-line interface. The script constructs the shared cohort from the absolute repository datasets, then draws only panel f. The whole-figure orchestrator can supply the same prepared cohort to avoid repeating shared preparation.

The analysis cohort contains 228 molecules, including 112 nontoxic and 116 toxic hits from ECBD, CO-ADD, and the Drug Repurposing Hub. Shared construction rules, source IDs, exclusions, and sensitivity audits are documented one level up in `readme.md` and `shared/`.

The four plots compare Matthews correlation coefficient, F1 score, balanced accuracy, and precision. SafetyNet predicts toxicity at scores strictly greater than 0.2. All eight structural-alert comparators are computed from the selected molecular structures using RDKit catalogs. SafetyNet is placed first and the other methods are ordered by the displayed metric. `classification_metrics.csv` records all four confusion-matrix counts and metrics, and `cohort_summary.csv` supplies the prevalence. Dashed reference lines are 0 for Matthews correlation coefficient, 0.5 for balanced accuracy, and toxic prevalence for precision and F1. The F1 reference assumes independent random predictions whose probability of calling a molecule toxic equals the observed toxic prevalence.

`figure_3f.svg` and `figure_3f.png` are calculated from the molecular records. Outputs are written beside this script, independently of the process working directory.
