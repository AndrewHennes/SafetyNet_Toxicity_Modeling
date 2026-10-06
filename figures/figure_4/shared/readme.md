# Shared Figure 4 preparation

`prepare.py` validates and builds the actual antibacterial assay cohort used by panels b and c. Shared lineage, exclusions, repeated-measurement records and validation reports are beside this readme. Panel-specific calculations, sensitivity comparisons, CSVs and plots are implemented by the individual panel runners. The source-input hash and preparation-code hash control reuse of these tables.

Calling either panel’s runner prepares its dependencies automatically. The numerical rules are unchanged by the directory reorganization.
