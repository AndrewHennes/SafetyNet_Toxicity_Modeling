# Figure analysis input requirements

The active pipelines compute statistics from supplied assay records, structures and saved compound-level predictions.

1. **Internal cohorts and molecular comparators.** Cytotoxicity and antibacterial cohort membership, preprocessing rules, FDA comparator structures, and chemical-space settings.
2. **Benchmark training/test records.** Figure 2b/c uses per-fold identities, outcomes, and predictions for each architecture, with split and hyperparameter provenance.
3. **External-cohort definitions.** ECBD release, label cutoff, score aggregation, and similarity policy. Tox21 analyses record both full and similarity-filtered cohorts.
4. **Retrospective hit inclusion rules.** CC50 cutoff and censoring policy, molecular matching, replicate handling, and overlap exclusions for the combined ECBD, CO-ADD, and Drug Repurposing Hub analysis.
5. **Prospective compound-level measurements.** Figure 4e–i uses candidate rankings, library membership, selection identities, antibacterial measurements, and three-cell-line cytotoxicity results.
6. **Animal dose and DrugBank records.** Figure 2g and Supplementary Figure 2b use dose-level toxicity observations, units, transformations, prediction scores, and endpoint/cohort definitions. Supplementary Figure 3 uses DrugBank molecule identities, inclusion metadata, and three cell-specific prediction scores.

Panel status reports identify the available analyses. Per-figure notes describe their inputs and cohort settings.
