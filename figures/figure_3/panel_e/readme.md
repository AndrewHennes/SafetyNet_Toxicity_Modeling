# Figure 3e · K. pneumoniae threshold retention and removal

Run `figure_3e.py` directly from any working directory. It has no command-line interface. The script constructs the shared cohort from the absolute repository datasets, then draws only panel e. The whole-figure orchestrator can supply the same prepared cohort to avoid repeating shared preparation.

The analysis cohort contains 110 molecules, including 85 nontoxic and 25 toxic hits from ECBD. Shared construction rules, source IDs, exclusions, and sensitivity audits are documented one level up in `readme.md` and `shared/`.

At thresholds 0.1 through 0.9, scores strictly greater than the threshold are predicted toxic, so equal scores are retained. The bars show the fraction of experimentally nontoxic hits retained and toxic hits removed. Geometric and harmonic means summarize those two fractions. `threshold_metrics.csv` includes all four confusion-matrix counts and derived fractions; `cohort_summary.csv` records the cohort size and summary statistics.

`figure_3e.svg` and `figure_3e.png` are calculated from the molecular records. Outputs are written beside this script, independently of the process working directory.
