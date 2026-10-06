"""Compute Figure 3d from molecular records; no command-line arguments."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
specification = spec_from_file_location("figure_3_panel_support", HERE.parent / "shared" / "panel_support.py")
support = module_from_spec(specification)
specification.loader.exec_module(support)
ORGANISM = "e_coli"


import numpy as np


def main(cohort=None):
    frames = support.frames_for(ORGANISM, cohort)
    support.write_tables(frames, ["threshold_metrics"], HERE)
    support.configure_style()
    figure, axis = support.plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    table = frames["threshold_metrics"].reset_index(drop=True)
    x = np.arange(len(table))
    axis.bar(x - .18, table.nontoxic_retained_fraction, .36,
             color=support.analysis.COLORS[0], label="Nontoxic retained")
    axis.bar(x + .18, table.toxic_removed_fraction, .36,
             color=support.analysis.COLORS[1], label="Toxic removed")
    axis.plot(x, table.g_mean, "o-", color="black", markersize=3, label="G-mean")
    axis.plot(x, table.h_mean, "s--", color="#888888", markersize=2, label="H-mean")
    best = table.g_mean.idxmax()
    axis.annotate(f"{table.g_mean[best]:.2f}", (x[best], table.g_mean[best]),
                  xytext=(0, 8), textcoords="offset points", ha="center")
    axis.set_xticks(x, [f">{value:.1f}" for value in table.threshold], rotation=45)
    axis.set(ylim=(0, 1.15), ylabel="Fraction / mean",
             xlabel="SafetyNet toxicity score threshold",
             title="d  " + support.analysis.SPECIES[ORGANISM])
    axis.legend(ncol=2, frameon=False, fontsize=8)
    support.save_figure(figure, HERE)
    return frames


if __name__ == "__main__":
    main()
