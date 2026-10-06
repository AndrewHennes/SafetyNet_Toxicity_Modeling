# Supplementary Figure 2 analysis

`supplementary_figure_2.py` recalculates panel a from the Tox21 molecular assay labels and three saved SafetyNet task scores. It uses the shared, local `figure_2/shared/analysis.py` calculation library. It does not read the original artwork or precomputed assay/animal performance summaries.

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/supplementary_figure_2/supplementary_figure_2.py
```

The figure-level script runs the individual `panel_a/supplementary_figure_2a.py` and `panel_b/supplementary_figure_2b.py` runners from any working directory. Panel a stores its data and plots in `panel_a/`; panel b records its blocked status in `panel_b/`. Shared dependencies and figure-wide validation are in `shared/`. Each panel runner can also be run independently. It accepts no command-line options.

The initial analysis uses all 7,973 supplied molecules and reproduces the supplied 42-assay auROC summary values from the underlying labels and scores. The refined analysis implements the manuscript's stated novelty rule using the stored maximum similarity ≤0.9, leaving 6,040 molecules before assay-specific missing-label exclusions. Four of the 46 assay columns contain only positives; the remaining 42 support auROC. Strict `<0.9` and `<1` alternatives, all three individual task scores, maximum task scores, average precision and trapezoidal precision-recall area are exported for comparison.

The filtered auROC range is 0.572905–0.885160. The molecular cohort table preserves source row identifiers, structures, all outcome columns, prediction scores and inclusion flags. Provenance and per-assay exclusions are saved with the output.

Panel b writes a status report until its compound-level animal toxicity input is supplied. `figure.svg` is the original-artwork reference.
