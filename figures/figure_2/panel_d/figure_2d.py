"""Compute Figure 2 panel d from its molecular data and required inputs."""
from __future__ import annotations
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent / 'shared' / 'workflow.py'
spec = importlib.util.spec_from_file_location('figure2_panel_d_workflow', LIBRARY)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
analysis = workflow.analysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc(frame: pd.DataFrame, output: Path) -> dict:
    labels = frame.native_toxicity_label.astype(int).to_numpy()
    scores = frame.MaxTox.to_numpy()
    metrics = analysis.evaluated_metrics(labels, scores)
    false_positive, true_positive, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pd.DataFrame({'false_positive_rate': false_positive, 'true_positive_rate': true_positive,
                  'threshold': roc_thresholds}).to_csv(output / 'panel_d_roc_values.csv', index=False)
    pd.DataFrame({'recall': recall, 'precision': precision,
                  'threshold': np.append(pr_thresholds, np.nan)}).to_csv(output / 'panel_d_pr_values.csv', index=False)
    figure, axes = plt.subplots(2, 1, figsize=(3.1, 6))
    axes[0].plot(false_positive, true_positive, color=workflow.PINK_LINE, linewidth=1.5)
    axes[0].plot([0, 1], [0, 1], color='#aab0b9', linestyle='--', linewidth=.9)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title=f'auROC = {metrics["auroc"]:.3f}')
    axes[1].plot(recall, precision, color=workflow.PINK_LINE, linewidth=1.5)
    axes[1].axhline(metrics['prevalence'], color='#aab0b9', linestyle='--', linewidth=.9,
                    label=f'Prevalence {metrics["prevalence"]:.4f}')
    axes[1].set(xlabel='Recall', ylabel='Precision', title=f'Average precision = {metrics["average_precision"]:.3f}')
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_box_aspect(1)
    figure.suptitle('ECBD supplied restricted cohort\n'
                     f'Tanimoto ≤ 0.9; n = {len(frame):,}', fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .94), h_pad=1.4)
    workflow.save(figure, 'panel_d_ecbd_roc_pr', output)
    return metrics


def main() -> dict:
    workflow.configure_style()
    output = HERE
    cohort = workflow.prepare_ecbd()
    metrics = plot_roc(cohort, output)
    ensemble = workflow.prepare_member_audit()
    return workflow.write_status('d', {
        'status': 'recomputed_from_rows', 'metrics': metrics,
        'saved_ensemble': ensemble,
        'shared_cohort': str(workflow.ECBD / 'primary_ecbd_cohort.csv'),
        'source_figure_or_summary_values_used': False,
        'limitation': 'Evaluation inputs: cohort membership and confidence-band construction.'})


if __name__ == '__main__':
    main()
