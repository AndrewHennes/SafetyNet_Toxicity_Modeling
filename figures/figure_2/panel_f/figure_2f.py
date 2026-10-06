"""Compute Figure 2 panel f from its molecular data and required inputs."""
from __future__ import annotations
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent / 'shared' / 'workflow.py'
spec = importlib.util.spec_from_file_location('figure2_panel_f_workflow', LIBRARY)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
analysis = workflow.analysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_tox21(metrics: pd.DataFrame, name: str, title: str, output: Path) -> None:
    figure, ax = plt.subplots(figsize=(3, 4.6))
    groups = [metrics.loc[metrics.category.eq(category), 'auroc'].to_numpy() for category in range(1, 6)]
    ax.boxplot(groups, patch_artist=True, showfliers=False, widths=.55,
               boxprops={'facecolor': 'white', 'linewidth': .8},
               medianprops={'color': '#222222', 'linewidth': 1.1})
    for x, scores in enumerate(groups, start=1):
        offsets = np.linspace(-.10, .10, len(scores)) if len(scores) > 1 else np.zeros(len(scores))
        ax.scatter(x + offsets, scores, s=13, color=workflow.CYAN, zorder=3)
    ax.set(ylabel='auROC', ylim=(.48, .91), xticks=range(1, 6), title=title)
    ax.axhline(.5, color='#888888', linestyle='--', linewidth=.8)
    ax.grid(axis='y', linewidth=.45, color='#dddddd', linestyle=':')
    ax.set_axisbelow(True)
    figure.subplots_adjust(left=.2, right=.98, top=.9, bottom=.31)
    figure.text(.18, .24, '1: Enzyme activities\n2: Cellular stress responses\n3: Signaling pathways\n'
                          '4: Receptors / hormone signaling\n5: Cytotoxicity / ion channels',
                va='top', fontsize=8, linespacing=1.35)
    workflow.save(figure, name, output)


def main() -> dict:
    workflow.configure_style()
    output = HERE
    metrics = analysis.analyze_tox21(output)
    alternatives = pd.read_csv(output / 'tox21_metrics_all_policies_and_scores.csv')
    initial = alternatives.loc[alternatives.policy.eq('unfiltered') &
                               alternatives.score.eq('recomputed_max_toxicity') &
                               alternatives.status.eq('evaluated')]
    plot_tox21(initial, 'iteration_1_panel_f_unfiltered',
               'Tox21 · all supplied molecules', output)
    plot_tox21(metrics, 'panel_f_tox21_recomputed',
               'Tox21 · Tanimoto ≤ 0.9', output)
    return workflow.write_status('f', {
        'status': 'recomputed_from_rows', 'evaluable_assays': len(metrics),
        'primary_auroc_range': [float(metrics.auroc.min()), float(metrics.auroc.max())],
        'source_figure_or_summary_values_used': False})


if __name__ == '__main__':
    main()
