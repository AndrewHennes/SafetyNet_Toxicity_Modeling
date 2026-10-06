# Figure 4 scientific reproduction

`figure_4.py` runs panels a–i through their local runners. Panels b and c compute antibacterial intersections and two explicitly distinguished correlation interpretations from the supplied Broad antibacterial assay measurements. Shared cohort preparation and validation live in `shared/`; each panel keeps its own script, results and readme. The complete source artwork remains at the figure root. Published values are used only for comparison CSVs and do not set any plotted value.

The default compound definition is the supplied raw SMILES identity, retaining differences in formulations and stereochemistry. Missing assays are excluded explicitly. Repeated distinct inhibition measurements for the same identity are averaged per endpoint before applying the stated >80% growth-inhibition rule. Seven cohort/identity policies are audited, including exact-row deduplication, first/last records, canonicalized SMILES and provided standardized SMILES. The selected cohort is recorded with its inclusion and aggregation rules.

`figure.svg` and `source_preview.png` are reference artwork. The shared chemical-space code supports panel a, and panel d is a conceptual architecture diagram. Panels e–i write status reports until the prospective selection and experimental inputs are supplied.

Run the saved script from any directory using a Python environment containing pandas, NumPy, Matplotlib and RDKit. It has no command-line arguments and uses absolute source and output paths.
