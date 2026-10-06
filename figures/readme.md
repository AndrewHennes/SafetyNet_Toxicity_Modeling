# SafetyNet scientific figure analyses

`generate_all_figures.py` reruns the analyses supported by the supplied compound-level datasets. It reads assay values, molecular structures and saved per-compound model predictions; constructs explicit cohorts; recomputes statistics; and produces plotting tables and figures. It does not use SVG paths, artwork-digitized counts or published metric summaries as numerical substitutes for those analyses.

Each figure directory contains a `panel_a/`, `panel_b/`, and so on for every labeled subfigure. Each panel has a script named for its figure and letter, such as `figure_1a.py` or `supplementary_figure_1a.py`, results and notes. Files genuinely used by several panels live in `shared/`. The figure-level script, such as `figure_1.py`, runs its panel entry points; the top-level runner runs all seven figures. A panel runner resolves its inputs and outputs to absolute paths, so it can be executed from any working directory. `figure.svg` and `source_preview.png` preserve the original figure for comparison only. They are not read by the scientific pipeline.

## Analyses

| Figure | Analysis from supplied observations |
|---|---|
| 1 | Calculate molecular descriptors and densities; define toxicity from three relative-viability measurements; count intersections and compute Pearson correlations; compute a new Morgan-fingerprint chemical-space projection |
| 2 | Compute external ECBD and ClinTox performance from labels/predictions; compute structural alerts from molecules; compute Tox21 assay metrics from row-level outcomes; examine cohort, similarity and ensemble definitions |
| 3 | Join antibacterial and cytotoxicity observations across ECBD, CO-ADD and Drug Repurposing Hub; handle censored CC50 intervals and units; audit identities/deduplication; compute score distributions, threshold performance and structural-filter metrics |
| 4 | Compute antibacterial hit labels from inhibition; audit replicates, missing assays and duplicates; compute intersections and correlations; compute chemical-space projection |
| Supplementary 1 | Compute the three relative-viability histograms |
| Supplementary 2 | Compute individual Tox21 assay metrics from labels/predictions |
| Supplementary 3 | Validate the input schema and report missing inputs; analyze DrugBank scores if a real compound-level input is provided |

Saved model predictions are treated as supplied analytical inputs. The pipeline evaluates those predictions. Model-fitting configurations, hyperparameters and fold assignments are separate inputs.

## Dataset organization

The original internal assay tables remain under `datasets/toxicity_datasets/` and `datasets/antibiotic_activity_datasets/`. External compound-level records are organized by source under `datasets/external_validation/{ecbd,coadd,drug_repurposing_hub,clintox,tox21}/`. Published or previously computed summary tables are under `datasets/reference_summaries/`, clearly separated from the compound-level inputs. New descriptor caches and molecular projection coordinates are under `datasets/derived/`.

`datasets/dataset_manifest.csv` lists canonical paths, provenance, delimiters, sizes and hashes. `datasets/dataset_name_mapping.csv` records the moves and filename changes. Dataset values and scientific column names are preserved. The CO-ADD HEK293 and Drug Repurposing Hub tables now have `.tsv` extensions matching their actual delimiters. All filenames use lowercase snake case.

## Run from any directory

The recorded dependencies are in the absolute requirements path below. The existing scientific environment already contains the scientific libraries; openTSNE supplies the scalable t-SNE implementation.

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python -m pip install --requirement /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/requirements.txt
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/generate_all_figures.py
```

Each script can also be run separately by its absolute path. There are no command-line options. Outputs resolve from the script location, not the terminal's working directory. A first molecular projection run is substantially more expensive than subsequent runs; its cache is validated against source hashes, numerical parameters, software versions and the coordinate file hash. Descriptor caches likewise check their molecular sources and RDKit version.

## Chemical-space methods and interpretation

`chemical_space.py` starts from actual SMILES strings. It parses molecules, preserves the first source row for each canonical isomeric structure within each library, computes 2,048-bit radius-2 Morgan fingerprints, reduces them to 50 components with TruncatedSVD, and fits t-SNE with perplexity 30, seed 42 and the recorded optimization settings. All valid unique molecules are used, without subsampling. Source-row identities, exclusions, intermediate coordinates and final coordinates are saved.

The projection uses the fingerprint, preprocessing and optimization settings recorded in its manifest. Figures 3 and 4 share the same antibacterial embedding; Figure 4 omits the Drug Repurposing Hub points from display. See the [openTSNE API documentation](https://opentsne.readthedocs.io/en/stable/api/index.html) for the implementation's optimization and nearest-neighbor settings.

## Checking the result

Input requirements are documented in each figure's `missing_inputs.md`. Panel status reports record which analyses ran.
