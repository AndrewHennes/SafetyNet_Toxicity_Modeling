"""Plot the three relative-viability distributions in Supplementary Figure 1a."""
from __future__ import annotations
import json
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUTPUT = HERE
os.environ.setdefault("MPLCONFIGDIR",str(HERE.parents[1] / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE = Path("/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/datasets/toxicity_datasets/safetynet_100k_cytotoxicity_dataset.csv")
COLUMNS = [f"{name}_Signal_at_10uM" for name in ("HepG2","HSkMC","IMR90")]


def render(frame, bin_count, stem):
    edges = np.linspace(0,2,bin_count+1)
    plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.linewidth":.7,"svg.fonttype":"none"})
    fig, axes = plt.subplots(1,3,figsize=(8.4,2.65),sharey=True,layout="constrained")
    table,counts_outside = [],{}
    for ax,column,label in zip(axes,COLUMNS,("HepG2","HSkMC","IMR-90")):
        values = frame[column].dropna().to_numpy()
        heights,_ = np.histogram(values,bins=edges)
        centers = (edges[:-1]+edges[1:])/2
        colors = np.where(centers<.8,"#d4509c","#999999")
        ax.bar(centers,heights,width=np.diff(edges)*.7,color=colors,linewidth=0)
        ax.axvline(.8,color="#d4509c",lw=1)
        ax.text(.98,.94,"Viability <0.8",ha="right",va="top",transform=ax.transAxes,color="#d4509c",fontsize=8)
        ax.set(title=label,xlim=(0,2),ylim=(0,9000),xlabel="Relative viability",xticks=[0,.5,1,1.5,2])
        counts_outside[label] = int(((values<0)|(values>2)).sum())
        table.extend({"cell_line":label,"bin_left":lo,"bin_right":hi,"count":int(n)} for lo,hi,n in zip(edges[:-1],edges[1:],heights))
    axes[0].set_ylabel("Number of compounds")
    fig.suptitle(f"Supplementary Figure 1 · {len(frame):,} supplied compounds",fontsize=11)
    fig.savefig(OUTPUT/f"{stem}.svg")
    fig.savefig(OUTPUT/f"{stem}.png",dpi=180)
    plt.close(fig)
    pd.DataFrame(table).to_csv(OUTPUT/f"{stem}_bins.csv",index=False)
    return {"bin_count":bin_count,"bin_width":float(edges[1]),"outside_display_range":counts_outside,
            "histogram_maxima":{label:int(max(r["count"] for r in table if r["cell_line"]==label))
                                 for label in ("HepG2","HSkMC","IMR-90")}}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE,usecols=COLUMNS)
    if frame.isna().any().any():
        raise ValueError("Missing viability values require an explicit handling policy.")
    # Compare 0.02-wide and 0.01-wide bins across the displayed viability range.
    initial = render(frame,100,"iteration_1")
    refined = render(frame,200,"supplementary_figure_1a")
    report = {"input_rows":len(frame),"artwork_rows":100390,"cutoff":.8,
              "initial":initial,"refined":refined,
              "remaining_difference":"Input cohort: 100,352 rows. Reference cohort: 100,390 rows.",
              "normalization":"Existing relative viability used directly; values outside 0–2 counted but not drawn."}
    (OUTPUT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
