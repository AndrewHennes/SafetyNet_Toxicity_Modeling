"""Compute Figure 3b from molecular records; no command-line arguments."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
specification = spec_from_file_location("figure_3_panel_support", HERE.parent / "shared" / "panel_support.py")
support = module_from_spec(specification)
specification.loader.exec_module(support)
ORGANISM = "e_coli"


def main(cohort=None):
    frames = support.frames_for(ORGANISM, cohort)
    support.write_tables(frames, ["histogram_counts", "kde_values"], HERE)
    support.configure_style()
    figure, axis = support.plt.subplots(figsize=(5.2, 3.8), layout="constrained")
    for label in [0, 1]:
        hist = frames["histogram_counts"].loc[lambda table: table.toxicity_label.eq(label)]
        axis.bar(hist.bin_left, hist.density, width=.05, align="edge", alpha=.75,
                 color=support.analysis.COLORS[label], edgecolor="#444444", linewidth=.4,
                 label=f"{'Nontoxic' if label == 0 else 'Toxic'} hits ({int(hist['count'].sum())})")
        kde = frames["kde_values"].loc[lambda table: table.toxicity_label.eq(label)]
        axis.plot(kde.score, kde.density, color="#d3529c" if label == 0 else "#7b4ea0",
                  linewidth=.9, linestyle="-" if label == 0 else ":")
    axis.set(xlim=(-.05, 1.05), xlabel="SafetyNet toxicity score", ylabel="Density",
             title="b  " + support.analysis.SPECIES[ORGANISM])
    axis.legend(frameon=False)
    support.save_figure(figure, HERE)
    return frames


if __name__ == "__main__":
    main()
