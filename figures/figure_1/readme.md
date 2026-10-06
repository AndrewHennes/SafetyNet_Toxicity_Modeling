# Figure 1 analysis

`figure_1.py` reads the supplied 100K relative-viability table and Tox21 molecular structures. It calculates molecular weight, logP, topological polar surface area and quantitative estimate of drug-likeness with RDKit; estimates densities from those calculated descriptors; labels cytotoxicity separately for each cell line at relative viability below 0.8; and recomputes intersection counts and Pearson correlations.

No artwork-derived values are used to calculate or draw these results. Published intersection values appear only in a comparison audit. The original `figure.svg` is a reference.

Each subfigure has its own `panel_a/` through `panel_f/` directory containing its runner, plots, numerical outputs and notes. Cohort preparation and intersection tables used by several panels are under `shared/`. The figure-level `figure_1.py` runs all six panel entry points. The descriptor cache, its input hashes and the exact screening cohort used are under `datasets/derived/figure_1/`. Descriptor calculation preserves molecular graph text; a missing space before a CXSMILES extension is repaired without altering the graph or metadata. Four supplied SMILES strings remain invalid. Their assay observations remain included in the viability analysis; only their unavailable molecular descriptors are omitted.

The shared `figures/chemical_space.py` independently computes panel b from the supplied molecules and writes the plot under `panel_b/`. Morgan fingerprint and t-SNE settings are recorded in the analysis manifest. Panel c requires the FDA molecular list to include that comparator.

The supplied table contains 100,352 measurements. At the specified viability threshold, 20,185 compounds are toxic in at least one cell line and 1,991 in all three.
