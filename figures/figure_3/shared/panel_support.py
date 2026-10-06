"""Shared imports, data preparation, and output conventions for Figure 3 panels."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
specification = spec_from_file_location("figure_3_cohort_analysis", HERE / "analysis.py")
analysis = module_from_spec(specification)
specification.loader.exec_module(analysis)


def frames_for(organism, cohort=None):
    """Compute panel statistics from newly prepared or explicitly supplied molecular rows."""
    if cohort is None:
        cohort = analysis.prepare_cohort()
    selected = cohort.loc[cohort.organism.eq(organism)].copy()
    if selected.empty:
        raise ValueError(f"No molecular rows available for {organism}.")
    return analysis.analyze(selected)


def write_tables(frames, names, output):
    """Write only the selected panel's numeric results and cohort summary."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in [*names, "cohort_summary"]:
        frames[name].to_csv(output / (name + ".csv"), index=False)


def configure_style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                        "axes.linewidth": .6, "svg.fonttype": "path"})


def save_figure(figure, output):
    output = Path(output).resolve()
    stem = output.parent.name + output.name.removeprefix("panel_")
    figure.savefig(output / f"{stem}.svg")
    figure.savefig(output / f"{stem}.png", dpi=180)
    plt.close(figure)
