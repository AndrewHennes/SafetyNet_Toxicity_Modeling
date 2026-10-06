# Supplementary Figure 3b

`supplementary_figure_3b.py` calculates score-density histograms and Gaussian kernel density estimates from the shared DrugBank compound-level input. It retains 20 equal bins over [0,1], independent density normalization and Scott bandwidth. Distribution CSVs and SVG/PNG plots are local to this panel directory.

The input schema is in `../shared/required_input_schema.json`. The script writes its current input status before analysis.
