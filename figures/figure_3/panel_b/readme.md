# Figure 3b · E. coli safetynet score distribution

Run `figure_3b.py` directly from any working directory. It has no command-line interface. The script constructs the shared cohort from the absolute repository datasets, then draws only panel b. The whole-figure orchestrator can supply the same prepared cohort to avoid repeating shared preparation.

The analysis cohort contains 228 molecules, including 112 nontoxic and 116 toxic hits from ECBD, CO-ADD, and the Drug Repurposing Hub. Shared construction rules, source IDs, exclusions, and sensitivity audits are documented one level up in `readme.md` and `shared/`.

The histogram uses 20 equal-width bins over [0, 1] and normalizes each toxicity class separately. Kernel density estimates use SciPy’s Scott bandwidth without boundary correction. The numeric outputs are `histogram_counts.csv`, `kde_values.csv`, and `cohort_summary.csv`.

`figure_3b.svg` and `figure_3b.png` are calculated from the molecular records. Outputs are written beside this script, independently of the process working directory.
