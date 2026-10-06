# Shared Figure 3 analysis

`analysis.py` builds the combined molecular cohorts from absolute repository dataset paths. Run it directly to refresh the shared cohort and construction audits without drawing any panel, or call `prepare_cohort()` from Python. `analyze(cohort)` calculates statistics from molecular rows and returns tables without writing panel files. It does not read source artwork, prior plot tables, or files in `iterations/`.

`panel_support.py` contains the common imports, preparation, style, and output conventions used by panels b–g. Each panel's own script selects its organism and draws its associated plot.

`audit_ecbd_versions.py` independently recomputes the three-release ECBD label/novelty sensitivity analysis and writes its tables here. `validate_analysis.py` checks the censoring cases, shared molecular records, and all six numerical panel results. Run the whole-figure orchestrator to refresh panels and both audits together.

Cohort construction is documented in the parent `readme.md`. `iterations/` holds earlier composite renders.
