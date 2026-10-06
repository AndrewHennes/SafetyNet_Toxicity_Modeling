"""Compute Figure 1f Pearson correlations of raw relative viability."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from analysis import DISPLAY, configure_style, save_figure, viability_statistics
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def main():
    configure_style()
    _, _, correlation = viability_statistics()
    correlation.to_csv(HERE / "viability_correlations.csv")
    fig, ax = plt.subplots(figsize=(3.7, 3.6), layout="constrained")
    matrix = np.ma.masked_where(np.triu(np.ones((3, 3)), 1).astype(bool), correlation.to_numpy())
    cmap = LinearSegmentedColormap.from_list("viability", ["#6ecde4", "white", "#a384b7"])
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
    for y in range(3):
        for x in range(y + 1):
            ax.text(x, y, f"{matrix[y,x]:.2f}", ha="center", va="center",
                    color="white" if x == y else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=DISPLAY, yticklabels=DISPLAY,
           title="f  Pearson correlation\nof viability at 10 µM")
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#555555", linewidth=.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=.04, pad=.05, label="Pearson R")
    save_figure(fig, HERE, "viability_correlations")


if __name__ == "__main__":
    main()
