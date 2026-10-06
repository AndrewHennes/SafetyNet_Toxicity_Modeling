"""Compute Figure 2 panel e from its molecular data and required inputs."""
from __future__ import annotations
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent / 'shared' / 'workflow.py'
spec = importlib.util.spec_from_file_location('figure2_panel_e_workflow', LIBRARY)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
analysis = workflow.analysis


def main() -> dict:
    workflow.configure_style()
    output = HERE
    cohort = workflow.prepare_ecbd()
    alerts = analysis.compute_alerts(cohort, 'SMILES', output, 'ecbd')
    metrics = analysis.filter_metrics(alerts, 'native_toxicity_label', 'MaxTox',
                                      'ecbd_restricted_similarity_le_0_9')
    metrics.to_csv(output / 'panel_e_ecbd_filter_metrics.csv', index=False)
    workflow.plot_filters(metrics, 'panel_e_ecbd_filters_recomputed',
                 'ECBD · structural alerts', output)
    return workflow.write_status('e', {
        'status': 'recomputed_from_rows_and_structures',
        'complete_cohort_rows': int(metrics.n.iloc[0]),
        'methods': metrics.method.tolist(),
        'shared_cohort': str(workflow.ECBD / 'primary_ecbd_cohort.csv'),
        'source_figure_or_summary_values_used': False})


if __name__ == '__main__':
    main()
