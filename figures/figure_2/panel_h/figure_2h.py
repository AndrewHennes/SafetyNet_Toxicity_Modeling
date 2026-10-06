"""Compute Figure 2 panel h from its molecular data and required inputs."""
from __future__ import annotations
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent / 'shared' / 'workflow.py'
spec = importlib.util.spec_from_file_location('figure2_panel_h_workflow', LIBRARY)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
analysis = workflow.analysis


def main() -> dict:
    workflow.configure_style()
    output = HERE
    cohort, _ = analysis.analyze_clintox(output)
    alerts = analysis.compute_alerts(cohort, 'SMILES', output, 'clintox')
    unfiltered = analysis.filter_metrics(alerts, 'target', 'max_of_three_task_scores',
                                        'clintox_all_supplied')
    unfiltered.to_csv(output / 'clintox_unfiltered_filter_metrics.csv', index=False)
    workflow.plot_filters(unfiltered, 'iteration_1_panel_h_clintox_unfiltered',
                 'ClinTox · all supplied compounds', output)
    overlap = analysis.clintox_overlap_sensitivity(alerts, output)
    chiral = overlap.loc[overlap.chirality.eq(True) &
                         overlap.similarity_policy.eq('similarity_le_0_9')]
    workflow.plot_filters(chiral, 'panel_h_clintox_morgan_overlap_sensitivity',
                 'ClinTox · assumed Morgan r2/2048/chiral overlap ≤0.9', output)
    primary = overlap.loc[overlap.chirality.eq(False) &
                          overlap.similarity_policy.eq('similarity_le_0_9')]
    primary.to_csv(output / 'panel_h_clintox_filter_metrics.csv', index=False)
    workflow.plot_filters(primary, 'panel_h_clintox_filters_recomputed',
                 'ClinTox · assumed Morgan r2/2048/achiral overlap ≤0.9', output)
    return workflow.write_status('h', {
        'status': 'recomputed_from_rows_and_structures',
        'input_rows': len(cohort), 'primary_rows': int(primary.n.iloc[0]),
        'primary_positive_count': int(primary.positive_count.iloc[0]),
        'primary_policy': 'Computed Morgan radius2/2048-bit/achiral maximum similarity <=0.9; achiral Morgan fingerprint policy',
        'safetynet_metrics': primary.loc[primary.method.eq('safetynet')].iloc[0].to_dict(),
        'chiral_sensitivity_rows': int(chiral.n.iloc[0]),
        'chiral_sensitivity_positive_count': int(chiral.positive_count.iloc[0]),
        'source_figure_or_summary_values_used': False})


if __name__ == '__main__':
    main()
