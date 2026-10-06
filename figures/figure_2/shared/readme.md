# Shared Figure 2 resources

`analysis.py` contains the scientific calculations used by Figure 2 and Supplementary Figure 2a. It was moved without changing its contents. `test_analysis.py` supplies the existing scientific checks. `workflow.py` provides shared ECBD preparation, common filter-comparison plotting, styling and figure-wide audits. Each panel script owns its analysis steps; panel d also owns ROC and precision-recall plotting, and panel f owns Tox21 plotting.

`ecbd/` contains the cohorts, source-version comparisons, label-policy sensitivity analyses and saved-member aggregation audits needed across panels d/e. Shared preparations are reused only when source hashes and calculation-code hashes match. A panel runner does not run unrelated panels.

Figure-wide dependencies, assumptions, runtime versions, aggregate validation and panel-run status are stored here. `relocation_integrity.json` records the hashes verified immediately when existing files were moved. Generating a result can subsequently update its provenance or timestamp.
