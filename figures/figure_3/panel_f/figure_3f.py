"""Compute Figure 3f from molecular records; no command-line arguments."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
specification = spec_from_file_location("figure_3_panel_support", HERE.parent / "shared" / "panel_support.py")
support = module_from_spec(specification)
specification.loader.exec_module(support)
ORGANISM = "e_coli"


import pandas as pd


def main(cohort=None):
    frames = support.frames_for(ORGANISM, cohort)
    support.write_tables(frames, ["classification_metrics"], HERE)
    support.configure_style()
    figure, axes = support.plt.subplots(2, 2, figsize=(7.0, 6.2), layout="constrained")
    metrics = frames["classification_metrics"]
    summary = frames["cohort_summary"].iloc[0]
    prevalence = summary.toxic / summary.n
    baselines = {"mcc": 0, "balanced_accuracy": .5, "f1": prevalence, "precision": prevalence}
    for axis, (key, title) in zip(axes.flat, support.analysis.METRIC_NAMES.items()):
        ordered = pd.concat([metrics[metrics.method.eq("SafetyNet")],
                             metrics[~metrics.method.eq("SafetyNet")].sort_values(key, ascending=False)])
        axis.bar(ordered.method, ordered[key], color=["#e8a0c6"] + ["#d4dae0"] * (len(ordered) - 1),
                 edgecolor="white")
        axis.tick_params(axis="x", rotation=55, labelsize=8)
        axis.set_title(title, fontsize=10)
        axis.axhline(baselines[key], color="#aaaaaa", linestyle="--", linewidth=.7)
    figure.suptitle("f  " + support.analysis.SPECIES[ORGANISM], fontsize=12)
    support.save_figure(figure, HERE)
    return frames


if __name__ == "__main__":
    main()
