# Supplementary Figure 3 scientific reproduction

The script writes its input schema and status to `shared/`. Analysis requires the DrugBank compound lists and per-compound SafetyNet predictions.

`figure.svg` and `source_preview.png` preserve the original artwork as a reference. Once the actual cohort and predictions are supplied in the documented schema, the root `supplementary_figure_3.py` orchestrates `panel_a/supplementary_figure_3a.py` for toxicity counts and `panel_b/supplementary_figure_3b.py` for histograms and density curves. Each panel keeps its results in its own directory, while shared input validation and the cohort remain in `shared/`.

The configured input path is absolute and the script writes beside itself, so it can run from any working directory. It has no command-line arguments.
