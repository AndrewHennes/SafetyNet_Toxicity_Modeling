# Supplementary Figure 1 analysis

`supplementary_figure_1.py` reads the three original `*_Signal_at_10uM` columns and recalculates histogram counts directly. The three cell-line plots together form panel a. Its script, plots, numerical tables and validation report are in `panel_a/`; the figure-level runner delegates to that panel. No original SVG or artwork histogram heights are numerical inputs.

The final histogram uses fixed 0.01-wide bins between 0 and 2, with a viability threshold at 0.8. Counts are calculated directly from the relative-viability observations. Observations beyond the displayed range are counted and reported, not silently removed from the dataset. The initial 0.02-wide histogram remains as an analysis comparison.

The three distributions are recalculated from the shared 100,352-row dataset.
