# Figure 3 input requirements

Panels b–g use a compound table containing source identifiers, molecular structures, antibacterial and mammalian toxicity labels, assay concentrations, cell-specific prediction scores, the maximum score, and inclusion/exclusion flags.

Cohort construction uses the following metadata.

1. The ECBD release, model-prediction file, antibacterial and HepG2 label thresholds, and chemical-similarity exclusion rules.
2. CO-ADD toxicity-label rules, concentration settings, and the source data dictionary.
3. CO-ADD censoring and concentration-unit policies.
4. Drug Repurposing Hub molecular matching, hit definition, HepG2 dose/screen, and replicate policy.
5. Cross-source molecular deduplication and endpoint combination rules.
6. The inference ensemble and preprocessing version, or archived compound-level scores used for analysis.
7. Kernel-density bandwidth, molecular cohort and ordering, representation, preprocessing, random state, and projection settings or saved coordinates.
