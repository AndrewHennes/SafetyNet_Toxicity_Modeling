"""Compute Figure 1d toxicity intersections from relative viability."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from analysis import DISPLAY, configure_style, save_figure, viability_statistics
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_venn(ax, counts, total):
    for center, color in zip(((.38, .61), (.66, .61), (.52, .36)),
                             ("#bce2eb", "#058fad", "#77d1e7")):
        ax.add_patch(Circle(center, .29, facecolor=color, edgecolor="none", alpha=.55))
    positions = {1:(.25,.63), 2:(.79,.63), 4:(.52,.16), 3:(.52,.71),
                 5:(.34,.39), 6:(.70,.39), 7:(.52,.48)}
    for mask, pos in positions.items():
        label = str(counts[mask])
        if mask in (1,2,4):
            label += "\n" + DISPLAY[(1,2,4).index(mask)]
        ax.text(*pos, label, ha="center", va="center", fontsize=8)
    union = sum(counts.values())
    ax.set_title(f"{union:,} toxic compounds\n({100*union/total:.1f}%)", fontsize=9)
    ax.set(xlim=(0,1.05), ylim=(0,1))
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    configure_style()
    viability, counts, _ = viability_statistics()
    fig, ax = plt.subplots(figsize=(3.5, 3.5), layout="constrained")
    plot_venn(ax, counts, len(viability))
    fig.suptitle("d  Toxicity overlap", fontsize=11)
    save_figure(fig, HERE, "toxicity_overlap")


if __name__ == "__main__":
    main()
