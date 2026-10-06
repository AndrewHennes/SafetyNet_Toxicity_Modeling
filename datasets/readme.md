# SafetyNet analysis datasets

Source measurements and predictions used by the figure analyses are organized here by purpose and source. All filenames use lowercase snake case. Scientific column names and retained values are preserved; documented column removals are noted below.

- `antibiotic_activity_datasets/` contains the supplied Broad 39K, CO-ADD and ECBD activity measurements.
- `toxicity_datasets/` contains the supplied internal 100K and ECBD cytotoxicity measurements.
- `external_validation/` contains compound-level external assay labels, saved model predictions, structural alerts and, where supplied, embeddings. Subdirectories identify ECBD, CO-ADD, Drug Repurposing Hub, ClinTox and Tox21.
- `reference_summaries/` contains previously computed Tox21 metrics and animal-toxicity correlations.
- `derived/` contains reproducible intermediate molecular descriptors and chemical-space coordinates, with source hashes, exclusion records and numerical settings. These can be regenerated from the source datasets by the figure scripts.

`dataset_manifest.csv` records current paths, original filenames, delimiters, row counts where known, file sizes and SHA-256 digests. `dataset_name_mapping.csv` provides the complete relocation map for the formerly separate `relevant_datasets` files. The `contents_modified` field identifies datasets whose columns have changed. The two tab-delimited files previously named `.csv` now end in `.tsv`.

Each ECBD version retains its own cohort, toxicity labels, and model predictions. Figure analyses record the version and cohort rules used.

Regeneration starts at `/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/generate_all_figures.py`. Panel-specific plots and numerical results live under `figures/*/panel_*/`. Cohort exports, preparation code and audits used by multiple panels live under the corresponding `figures/*/shared/` directories. The `datasets/derived/` tables serve as reusable intermediate inputs to plotting.

The ECBD ensemble-prediction TSV is managed by Git Large File Storage. Its `Minimol_Representation` column has been removed, reducing the file from 1.19 GB to 152.34 MB. All 101,097 rows and the remaining 90 columns are preserved, including assay measurements and model predictions. The filename remains unchanged for compatibility with the figure scripts. The dataset manifest records the updated size and SHA-256 digest.
