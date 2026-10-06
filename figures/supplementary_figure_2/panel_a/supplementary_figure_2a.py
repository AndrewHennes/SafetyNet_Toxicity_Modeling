"""Calculate Supplementary Figure 2a assay metrics from Tox21 labels and predictions."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parent.parent / '.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIBRARY = HERE.parent.parent / 'figure_2' / 'shared' / 'analysis.py'
spec = importlib.util.spec_from_file_location('safetynet_tox21_analysis', LIBRARY)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def render(metrics: pd.DataFrame, stem: str, policy: str) -> None:
    values = metrics.sort_values('auroc', ascending=False, kind='stable')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': .6,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    figure, axis = plt.subplots(figsize=(11.3, 4.7))
    x = np.arange(len(values))
    axis.bar(x, values.auroc, width=.75, color='#aaaaaa')
    axis.set_xticks(x, values.assay_id, rotation=90, fontsize=6)
    axis.axhline(.5, color='#888888', linewidth=.7, linestyle='--')
    axis.set(xlim=(-.8, len(values) - .2), ylim=(.4, .92), ylabel='auROC',
             title=f'Tox21 · {len(values)} assays recalculated from molecular outcomes\n{policy}')
    axis.spines[['top', 'right']].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT / f'{stem}.svg')
    figure.savefig(OUTPUT / f'{stem}.png', dpi=180)
    plt.close(figure)


def main() -> dict:
    selected = analysis.analyze_tox21(OUTPUT)
    alternatives = pd.read_csv(OUTPUT / 'tox21_metrics_all_policies_and_scores.csv')
    initial = alternatives.loc[alternatives.policy.eq('unfiltered') & alternatives.score.eq('recomputed_max_toxicity') & alternatives.status.eq('evaluated')]
    render(initial, 'iteration_1_unfiltered_assays', 'All supplied molecules; per-assay missing labels excluded')
    render(selected, 'panel_a_recomputed_tox21', 'Stored maximum training similarity ≤ 0.9; per-assay missing labels excluded')
    dependencies = [
        {'panel': 'a', 'status': 'recomputed_from_molecular_labels_and_predictions',
         'source': str(analysis.input_path('tox21')), 'missing': 'Similarity-reference identity and fingerprint settings; selected cohort has 6040 rows'},
        {'panel': 'b', 'status': 'blocked_raw_observations_absent',
         'source': 'Compound-level animal toxicity table',
         'missing': 'Compound identity, species/route/endpoint, LD50/LDLo/TDLo measurement, concentration units, toxicity score, inclusion policy and target transformation'},
    ]
    pd.DataFrame(dependencies).to_csv(HERE.parent / 'shared' / 'panel_dependencies.csv', index=False)
    report={'panel':'a','status':'recomputed_from_rows',
            'workflow':'Raw assay-label and prediction-score analysis',
            'precomputed_metric_or_artwork_values_read':False,
            'recomputed_assays':len(selected),
            'primary_auroc_range':[float(selected.auroc.min()),float(selected.auroc.max())],
            'shared_calculation_library':str(LIBRARY)}
    (OUTPUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUTPUT/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    main()
