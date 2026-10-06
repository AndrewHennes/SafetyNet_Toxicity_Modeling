"""Shared viability cohort and intersection calculations for Figure 1d–f."""
from pathlib import Path
import os
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
REPOSITORY = Path("/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private")
DERIVED = REPOSITORY / "datasets/derived/figure_1"
SCREEN = REPOSITORY / "datasets/toxicity_datasets/safetynet_100k_cytotoxicity_dataset.csv"
DISPLAY = ("HepG2", "HSkMC", "IMR-90")
SIGNALS = [f"{cell}_Signal_at_10uM" for cell in ("HepG2", "HSkMC", "IMR90")]
SOURCE_COUNTS = {1: 4593, 2: 4952, 4: 4709, 3: 1969, 5: 957, 6: 1005, 7: 1992}
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE.parent / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": .65,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})


def viability_statistics():
    """Compute the unchanged relative-viability cohort and exclusive counts."""
    viability = pd.read_csv(SCREEN, usecols=SIGNALS)
    if viability.isna().any().any():
        raise ValueError("Missing viability requires an explicit cohort policy.")
    masks = (viability.to_numpy() < .8).astype(int) @ np.array([1, 2, 4])
    counts = {mask: int((masks == mask).sum()) for mask in range(1, 8)}
    DERIVED.mkdir(parents=True, exist_ok=True)
    cohort = viability.copy()
    cohort.insert(0, "source_row", np.arange(len(cohort)))
    cohort["toxicity_intersection_mask"] = masks
    cohort.to_csv(DERIVED / "screening_analysis_cohort.csv", index=False)
    comparison = pd.DataFrame([
        {"hep_g2": bool(mask & 1), "hskmc": bool(mask & 2), "imr90": bool(mask & 4),
         "computed": count, "artwork": SOURCE_COUNTS[mask],
         "difference": count - SOURCE_COUNTS[mask]}
        for mask, count in counts.items()
    ])
    comparison.to_csv(HERE / "intersection_comparison.csv", index=False)
    correlation = viability.corr()
    correlation.index = DISPLAY
    correlation.columns = DISPLAY
    return viability, counts, correlation


def save_figure(figure, output, stem):
    figure.savefig(output / f"{stem}.svg")
    figure.savefig(output / f"{stem}.png", dpi=180)
    plt.close(figure)
