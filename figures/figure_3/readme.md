# Figure 3 analysis

`figure_3.py` coordinates the seven panel directories and shared audits. The analysis constructs compound-level cohorts from the supplied datasets, recalculates the statistics, evaluates all eight structural-filter comparators, and redraws panels b–g. `figure.svg` is retained separately as the source-artwork reference.

The primary reconstruction produces the following results.

| Organism | Nontoxic | Toxic | Total | Area under the ROC curve | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| E. coli | 112 | 116 | 228 | 0.6883 | 0.6937 |
| K. pneumoniae | 85 | 25 | 110 | 0.8871 | 0.8073 |

## Dataset construction

**European Chemical Biology Database (ECBD).** The primary analysis uses the 101,097-row assay-label/prediction table, its existing binarized antibacterial hit labels, its HepG2 toxicity labels, and `Max_Tox_Score`. This gives 62 E. coli hits and 110 K. pneumoniae hits before cross-source chemical deduplication. The supplied activity labels use an 80% growth-inhibition cutoff; HepG2 labels use 20% growth inhibition. A 50% antibacterial cutoff is evaluated separately.

**Community for Open Antimicrobial Drug Discovery (CO-ADD).** The analysis joins antimicrobial and HEK293 observations by compound ID, deduplicates repeated E. coli assay records, and requires consensus antibacterial labels and matching structures. The configured toxicity endpoint is the cytotoxic concentration for 50% effect (CC50) at or below the paired antibacterial screening concentration. Censoring intervals are retained, indeterminate cases are excluded, and concentration conversions use the molecular mass of the supplied molecular form. This yields 121 CO-ADD candidates before cross-source deduplication. Fixed-dose alternatives are saved as sensitivity analyses. Assay context is described in the CO-ADD [workflow](https://www.co-add.org/sites/co-add.org/files/CO-ADD_ScreeningWorkflow_May2016.pdf) and [assay conditions](https://db.co-add.org/organism).

**Drug Repurposing Hub (DRH).** The analysis selects E. coli growth-inhibition hits, requires exact canonical isomeric identity between antibacterial and HepG2 structures, and retains consensus HepG2 labels. This yields 54 candidates. Alternative matching, hit, and toxicity-label rules are evaluated separately.

**Combined cohort.** Eligible ECBD, CO-ADD, and DRH records are deduplicated by canonical isomeric structure. Labels must agree and saved scores must agree within 0.000001. The 228 E. coli molecules are represented by 59 ECBD, 120 CO-ADD, and 49 DRH records. Source identifiers and row numbers are retained. The K. pneumoniae cohort uses ECBD.

## Computed analysis

The analysis uses the saved per-molecule `Max_Tox_Score`. Prediction scores are analytical inputs. Predictions above the threshold are classified as toxic; equality is retained. Neither primary cohort contains a score exactly equal to 0.2. Histograms use 20 equal-width bins on [0, 1]. Kernel density estimates use SciPy's Scott bandwidth without a boundary correction. Geometric and harmonic means are calculated from nontoxic retention and toxic removal at thresholds 0.1 through 0.9.

All eight structural-alert systems are calculated from the selected molecules using RDKit 2024.09.4. Brenk uses the `BRENK` catalog; the other seven use the corresponding `CHEMBL_*` catalogs. Full supplied molecular forms are parsed without silently removing salts. RDKit documents these distinct catalogs ([API documentation](https://rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html)). All 1,183 available source ECBD alert flags agree with their computed counterparts in this reconstruction. Catalog names, entry counts, software version, and input-file hashes are recorded in `shared/analysis_manifest.json`.

The filter plots retain a prevalence-based random-classifier reference for precision and F1, zero for Matthews correlation coefficient, and 0.5 for balanced accuracy. The F1 reference specifically assumes a random classifier whose probability of predicting the toxic class equals the observed prevalence.

## ECBD version and novelty audit

`shared/audit_ecbd_versions.py`, also invoked by `figure_3.py`, evaluates all three supplied ECBD releases. It retains each release's own labels and saved prediction scores. Native activity and toxicity labels are compared with independently specified numerical growth-inhibition cutoffs. It then calculates maximum Tanimoto similarity for every eligible hit structure against the complete valid unique 100K and 39K reference libraries using 2,048-bit Morgan fingerprints of radius 2, without chirality. Query structures use the full recorded molecular form, while reference structures use the supplied standardized-SMILES column. These preprocessing choices and fingerprint settings are recorded with the analysis.

Native-label cohorts with known toxicity outcomes illustrate the differences between releases.

| ECBD release | E. coli nontoxic/toxic | K. pneumoniae nontoxic/toxic | K. pneumoniae after computed 100K maximum similarity <0.9 |
| --- | ---: | ---: | ---: |
| 101,097-row assay table | 41 / 21 | 85 / 25 | 66 / 21 |
| 106,373-row merged-panel table | 64 / 47 | 125 / 61 | 97 / 56 |
| 43,993-row restricted prediction table | 16 / 17 | 45 / 23 | 37 / 22 |

Provided similarity values from the restricted table are also tested, with exact isomeric structure joins used solely to annotate similarity. The audit distinguishes source rows from unique molecules with consistent labels/scores, and separately tests no novelty filter, 100K or 39K similarity cutoffs, a combined cutoff, exact-fingerprint overlap exclusion, and the provided known-antibiotic cutoff.

Missing whitespace before a CXSMILES extension is repaired while preserving the annotation. All reference-library row counts, valid structure counts, repair counts, input hashes, and fingerprint assumptions are saved in `shared/ecbd_version_similarity_manifest.json`. The full candidate row table and explicit variant memberships allow each sensitivity result to be traced back to its source records.

## Panel directories and execution

Every panel directory contains its own figure-specific Python script, `readme.md`, and generated results. A panel runner prepares the shared molecular cohort from repository datasets when needed, then writes only that panel's plot and numeric tables. Calling a panel runner does not redraw any other panel. The whole-figure `figure_3.py` prepares the common cohort once, passes it to panels b–g, runs panel a, and refreshes the shared ECBD sensitivity audit and validation.

| Directory | Panel content | Numeric outputs |
| --- | --- | --- |
| `panel_a/` | Chemical-space projection | Links to molecular coordinates and their manifest in `chemical_space_status.json` |
| `panel_b/` | E. coli score distributions | `histogram_counts.csv`, `kde_values.csv`, `cohort_summary.csv` |
| `panel_c/` | K. pneumoniae score distributions | `histogram_counts.csv`, `kde_values.csv`, `cohort_summary.csv` |
| `panel_d/` | E. coli threshold retention and removal | `threshold_metrics.csv`, `cohort_summary.csv` |
| `panel_e/` | K. pneumoniae threshold retention and removal | `threshold_metrics.csv`, `cohort_summary.csv` |
| `panel_f/` | E. coli structural-filter comparison | `classification_metrics.csv`, `cohort_summary.csv` |
| `panel_g/` | K. pneumoniae structural-filter comparison | `classification_metrics.csv`, `cohort_summary.csv` |

The six numerical panels each write `regenerated.svg` and `regenerated.png`. Their own scripts contain the panel-specific selection and rendering logic. Shared functions in `shared/analysis.py` implement cohort construction and statistics; `shared/panel_support.py` supplies common imports and output conventions. Both whole-figure and individual scripts accept no command-line arguments. Repository inputs use absolute paths, while output paths resolve from each script's own location. The core dependencies are pandas, NumPy, SciPy, scikit-learn, Matplotlib, and RDKit; panel a also uses the repository's shared chemical-space dependencies.

The common analysis files are retained directly under `shared/`.

- `reconstructed_cohort.csv` contains every selected molecule, toxicity label, saved score, contributing source IDs, and recalculated structural alerts. `eligible_source_records.csv` preserves the pooled records before cross-source deduplication.
- `coadd_joined_antibacterial_facts.csv`, `coadd_candidate_classification_audit.csv`, and `drh_candidate_identity_audit.csv` document the joins and label decisions.
- `excluded_records.csv` records unresolved molecular, label, score, and censoring cases.
- `cohort_variant_audit.csv` compares cohort definitions.
- `supplied_vs_recomputed_alerts.csv` checks the alert implementations against existing ECBD flags.
- `ecbd_version_hit_candidates_with_similarity.csv`, `ecbd_version_variant_memberships.csv`, and `ecbd_version_label_novelty_sensitivity.csv` document the three-release label and overlap audit.
- `iterations/` preserves the original initial and refined b–g composite layouts as historical evidence. No file there is read as an analysis input. Current reruns produce the six separate panels.

`shared/validate_analysis.py` checks twenty hand-worked CC50 censoring/unit-conversion cases, empty-structure handling, molecular uniqueness, threshold confusion counts, histogram normalization, and the computed alert flags. It also verifies every panel's numeric tables against direct recalculation from the molecular cohort. Results are saved as `shared/validation_report.json`.

Panel a uses the repository's shared chemical-space analysis, which recalculates a molecular-fingerprint embedding or reuses its validated coordinate cache. Its outputs are `panel_a/chemical_space_initial_svd.svg`, `panel_a/chemical_space_tsne.svg`, and `panel_a/chemical_space_status.json`. Fingerprint and projection settings are recorded with the saved coordinates. Input requirements are documented in `missing_inputs.md`.

`relocation_mapping.json` records the previous relative paths and their current locations. This reorganization changes file placement and panel layout without changing cohort definitions or numerical results.
