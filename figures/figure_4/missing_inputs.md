# Figure 4 input requirements

## Panel b cohort

The complete cohort contains 39,190 unique raw SMILES. The default averages distinct endpoint measurements for repeated structures. First/last selection, exact-row deduplication, standardized identity, and RDKit canonicalization are evaluated separately. Replicate-level analysis requires plate, well, lot, and compound identifiers.

Derived >80% growth-inhibition flags agree exactly with the supplied binary labels for complete records. The recorded mean optical density equals the mean of the two supplied replicate columns within about 5×10⁻¹⁰. All supplied records use 50 µM. No value is exactly at the 80% boundary.

Cohort selection requires the compound membership table and exclusion/deduplication rules.

## Panel c measurement definition

Panel c calculates separate Pearson correlation matrices for continuous growth inhibition and binary hit labels. The primary analysis uses supplied normalized inhibition values without clipping. Plate-level normalization requires controls, blanks, plate identifiers, and the normalization procedure.

## Remaining panels

- **a.** Molecular cohort and chemical-space projection settings, or saved coordinates.
- **d.** This is a conceptual architecture diagram preserved in the source SVG, not a dataset-derived result.
- **e.** The virtual library, retained membership, known-antibiotic reference structures, similarity settings/results, scores, and species-specific selection ranks.
- **f–i.** Compound identities, antibacterial model scores, measured activity, three-cell viability measurements, SafetyNet scores, testing species, and the vertical-axis score definition.
