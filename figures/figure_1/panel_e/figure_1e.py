"""Compute Figure 1e exclusive toxicity intersections."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from analysis import DISPLAY, configure_style, save_figure, viability_statistics
import matplotlib.pyplot as plt


def main():
    configure_style()
    viability, counts, _ = viability_statistics()
    fig, (upper, lower) = plt.subplots(2, 1, figsize=(4.8, 3.7),
        gridspec_kw={"height_ratios": (3, 1.25), "hspace": .05}, layout="constrained")
    order = sorted(counts, key=lambda mask: counts[mask], reverse=True)
    heights = [counts[mask] for mask in order]
    upper.bar(range(7), heights, color=["#058fad", "#77d1e7", "#bce2eb"] + ["#b5b5b5"] * 4,
              edgecolor="#555555", linewidth=.6)
    for x, y in enumerate(heights):
        upper.text(x, y + 120, f"{100*y/len(viability):.1f}%", ha="center", fontsize=8)
    upper.set(title="e  Toxicity intersections", ylabel="Compounds", ylim=(0, 6500),
              xticks=[], xlim=(-.7, 6.7))
    for x, mask in enumerate(order):
        lower.scatter([x] * 3, range(3), s=12, color="#dddddd")
        present = [y for y, bit in enumerate((1, 2, 4)) if mask & bit]
        lower.plot([x] * len(present), present, color="#555555", lw=1)
        lower.scatter([x] * len(present), present, s=24,
                      c=[["#bce2eb", "#058fad", "#77d1e7"][row] for row in present],
                      edgecolor="#333333", linewidth=.5)
    lower.set(xlim=(-.7, 6.7), ylim=(-.5, 2.5), xticks=[], yticks=range(3), yticklabels=DISPLAY)
    save_figure(fig, HERE, "toxicity_intersections")


if __name__ == "__main__":
    main()
