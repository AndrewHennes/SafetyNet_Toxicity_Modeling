"""Reusable Figure 2 preparations, common plotting styles and figure-level audits.

Each panel owns its analysis orchestration. Scientific calculations shared with
Supplementary Figure 2 are in the unchanged sibling analysis.py.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
SHARED=Path(__file__).resolve().parent
FIGURE=SHARED.parent
ECBD=SHARED/'ecbd'
os.environ.setdefault('MPLCONFIGDIR',str(FIGURE.parent/'.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('safetynet_figure2_science',SHARED/'analysis.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

PINK = '#e9a4c6'
PINK_LINE = '#ed3c98'
CYAN = '#009fcc'
GREY = '#d7dde4'
DISPLAY = {'safetynet': 'SafetyNet', 'glaxo': 'Glaxo', 'dundee': 'Dundee', 'bms': 'BMS',
           'brenk': 'Brenk', 'surechembl': 'SureChEMBL', 'mlsmr': 'MLSMR',
           'inpharmatica': 'Inpharmatica', 'lint': 'LINT'}
METRICS = {'mcc': 'MCC', 'f1': 'F1 score', 'balanced_accuracy': 'Balanced Accuracy', 'precision': 'Precision'}


def save(figure: plt.Figure, name: str, output: Path) -> None:
    figure.savefig(output / f'{name}.svg', bbox_inches='tight')
    figure.savefig(output / f'{name}.png', bbox_inches='tight', dpi=180)
    plt.close(figure)


def plot_filters(metrics: pd.DataFrame, name: str, title: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.5, 4.9))
    for ax, (metric, label) in zip(axes.ravel(), METRICS.items()):
        ordered = metrics.sort_values(metric, ascending=False)
        ordered = pd.concat([ordered.loc[ordered.method.eq('safetynet')],
                             ordered.loc[~ordered.method.eq('safetynet')]])
        ax.bar(np.arange(len(ordered)), ordered[metric],
               color=[PINK if method == 'safetynet' else GREY for method in ordered.method],
               edgecolor=['#444444' if method == 'safetynet' else '#9aaabd' for method in ordered.method], linewidth=.7)
        ax.set_xticks(np.arange(len(ordered)), ordered.method.map(DISPLAY), rotation=50, ha='right', fontsize=8)
        ax.get_xticklabels()[0].set(color=PINK_LINE, fontweight='bold')
        baseline = 0 if metric == 'mcc' else (.5 if metric == 'balanced_accuracy' else metrics.prevalence.iloc[0])
        ax.axhline(baseline, color='#999999', linestyle=(0, (7, 4)), linewidth=.8)
        ax.set_title(label, loc='right', fontweight='bold')
        low = max(0, ordered[metric].min() - .06) if metric == 'balanced_accuracy' else min(0, ordered[metric].min() - .025)
        ax.set_ylim(low, max(ordered[metric].max() * 1.14, baseline + .05))
    figure.suptitle(title + f'\nComplete common cohort n = {int(metrics.n.iloc[0]):,}', fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .92), h_pad=1.5)
    save(figure, name, output)


def write_panel_dependencies(output: Path) -> None:
    pd.DataFrame([
        {'panel': 'a', 'status': 'schematic_reference_only', 'inputs': 'Illustrator artwork', 'missing': 'No numerical analysis applies'},
        {'panel': 'b', 'status': 'blocked', 'inputs': 'Internal training data available', 'missing': 'Fold membership, baseline predictions, training configurations, and fold evaluation outputs'},
        {'panel': 'c', 'status': 'blocked', 'inputs': 'External experimental and some SafetyNet predictions available', 'missing': 'Cohort, labels, and model prediction sets for each fold'},
        {'panel': 'd', 'status': 'recomputed_from_rows', 'inputs': 'ECBD restricted labels, scores and stored similarity', 'missing': 'Evaluation membership and confidence-band construction'},
        {'panel': 'e', 'status': 'recomputed_from_rows_and_structures', 'inputs': 'Same ECBD cohort; RDKit structural catalogs', 'missing': 'Confirmation of catalog versions and final evaluation cohort'},
        {'panel': 'f', 'status': 'recomputed_from_rows', 'inputs': 'Tox21 assay labels, three task predictions, stored similarity', 'missing': 'Similarity reference, fingerprint settings and assay-category confirmation'},
        {'panel': 'g', 'status': 'blocked', 'inputs': 'Only precomputed animal endpoint summaries supplied', 'missing': 'Compound-level LD50/LDLo/TDLo observations, species/routes, prediction scores and transformations'},
        {'panel': 'h', 'status': 'recomputed_from_rows_and_structures', 'inputs': 'ClinTox labels, scores, molecular structures, 100K reference and RDKit catalogs', 'missing': 'Primary policy uses achiral Morgan similarity<=0.9'},
    ]).to_csv(output / 'panel_dependencies.csv', index=False)
    pd.DataFrame([
        {'choice': 'primary_ecbd_cohort', 'value': 'Restricted table with native toxicity label and maxsim_100K <= 0.9', 'basis': 'Only ECBD table with both measured labels and explicit training-overlap scores', 'uncertainty': 'Name-matched restricted source table'},
        {'choice': 'ecbd_alternative_labels', 'value': 'Native labels, measured growth inhibition >20 and >50 percent', 'basis': 'Audit label sensitivity using explicit interpretable rules', 'uncertainty': 'Alternative label definitions are evaluated separately'},
        {'choice': 'tox21_primary_similarity', 'value': '<=0.9', 'basis': 'Methods exclude compounds above 0.9', 'uncertainty': 'Reference identity/settings missing; <0.9 and <1 variants also exported'},
        {'choice': 'tox21_score', 'value': 'Maximum of three supplied per-task ensemble means', 'basis': 'Manuscript maximum-across-cell-types convention', 'uncertainty': 'Analysis uses saved predictions'},
        {'choice': 'tox21_assay_inclusion', 'value': 'Both native outcome classes available after cohort filter', 'basis': 'auROC is undefined for a single class', 'uncertainty': 'Four supplied assay columns contain only positives'},
        {'choice': 'safetynet_threshold', 'value': 'Score >=0.2 is toxicity-positive', 'basis': 'Nontoxic selection score <0.2', 'uncertainty': 'Exact cutoff ties are counted in metric tables'},
        {'choice': 'structural_alerts', 'value': 'Eight RDKit catalogs applied to whole submitted molecules', 'basis': 'Brenk and pharmaceutical catalog comparison reproducible from structures', 'uncertainty': 'Configured catalogs use full submitted molecules; stored flags are checked separately'},
        {'choice': 'clintox_primary_cohort', 'value': 'Computed achiral Morgan radius2/2048-bit maximum similarity<=0.9', 'basis': 'Novelty exclusion uses the stated Morgan fingerprint policy', 'uncertainty': 'Full-cohort and chiral sensitivity variants are included'},
        {'choice': 'clintox_novelty_sensitivity', 'value': 'Morgan radius2,2048bits,chirality on/off; supplied raw100K reference', 'basis': 'Reproducible molecular overlap analysis with settings explicitly recorded', 'uncertainty': 'Configured fingerprints and reference cohort are recorded with the analysis'},
        {'choice': 'random_f1_baseline', 'value': 'Prevalence', 'basis': 'Independent Bernoulli predictor with positive rate equal to prevalence', 'uncertainty': 'F1 has no single baseline independent of classifier positive rate'},
        {'choice': 'confidence_intervals', 'value': 'Not generated', 'basis': 'Exact fold/bootstrap provenance absent', 'uncertainty': 'Curves are single supplied ensemble-score curves'},
    ]).to_csv(output / 'assumption_audit.csv', index=False)


def configure_style() -> None:
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': .65,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white'})


def output_for(letter: str) -> Path:
    output = FIGURE / f'panel_{letter}'
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_status(letter: str, report: dict) -> dict:
    status = {'panel': letter, **report}
    for filename in ('status.json', 'validation.json'):
        (output_for(letter) / filename).write_text(json.dumps(status, indent=2) + '\n')
    return status


def prepare_ecbd() -> pd.DataFrame:
    """Prepare the d/e cohort once and invalidate it when inputs or code change."""
    ECBD.mkdir(parents=True, exist_ok=True)
    sources = ('ecbd_restricted', 'ecbd_merged', 'ecbd_assay')
    identity = {'sources': {key: analysis.sha256(analysis.input_path(key)) for key in sources},
                'calculation_source_sha256': analysis.sha256(SHARED / 'analysis.py'),
                'cohort_rule': 'native toxicity labels and maxsim_100K <=0.9'}
    metadata = ECBD / 'preparation_cache.json'
    expected = ['primary_ecbd_cohort.csv', 'ecbd_restricted_molecular_cohort.csv',
                'ecbd_merged_molecular_cohort.csv', 'ecbd_assay_molecular_cohort.csv',
                'ecbd_cohort_score_label_sensitivity.csv', 'ecbd_source_provenance.csv',
                'ecbd_label_definition_audit.csv', 'ecbd_input_version_comparison.csv',
                'ecbd_task_score_aggregation_audit.csv']
    if (metadata.is_file() and json.loads(metadata.read_text()) == identity
            and all((ECBD / name).is_file() for name in expected)):
        return pd.read_csv(ECBD / 'primary_ecbd_cohort.csv')
    cohort, _ = analysis.analyze_ecbd(ECBD)
    cohort.to_csv(ECBD / 'primary_ecbd_cohort.csv', index=False)
    metadata.write_text(json.dumps(identity, indent=2) + '\n')
    return cohort


def prepare_member_audit() -> dict:
    """Reuse an unchanged saved-member audit without reparsing the large TSV."""
    ECBD.mkdir(parents=True, exist_ok=True)
    source = analysis.input_path('ecbd_members')
    identity = {'source_sha256': analysis.sha256(source),
                'calculation_source_sha256': analysis.sha256(SHARED / 'analysis.py')}
    cache = ECBD / 'member_audit_cache.json'
    report_path = ECBD / 'ecbd_ensemble_aggregation_provenance.json'
    expected = [report_path, ECBD / 'ecbd_saved_member_and_aggregation_metrics.csv',
                ECBD / 'ecbd_recomputed_ensemble_scores.csv']
    if (cache.is_file() and json.loads(cache.read_text()) == identity
            and all(path.is_file() for path in expected)):
        return json.loads(report_path.read_text())
    report = analysis.audit_ecbd_members(ECBD)
    cache.write_text(json.dumps(identity, indent=2) + '\n')
    return report


def write_figure_validation(results: dict) -> dict:
    """Retain figure-wide audit information outside individual panel results."""
    write_panel_dependencies(SHARED)
    result = {
        'workflow': 'Panel-specific analysis from molecular observations and saved prediction scores',
        'source_figure_or_summary_values_used': False,
        'ecbd_primary': results['d']['metrics'],
        'saved_ensemble': results['d']['saved_ensemble'],
        'tox21_evaluable_assays': results['f']['evaluable_assays'],
        'tox21_primary_auroc_range': results['f']['primary_auroc_range'],
        'clintox_input_rows': results['h']['input_rows'],
        'clintox_filter_complete_rows': results['h']['primary_rows'],
        'clintox_primary_policy': results['h']['primary_policy'],
        'clintox_primary_safetynet_metrics': results['h']['safetynet_metrics'],
        'clintox_assumed_overlap_rows': results['h']['chiral_sensitivity_rows'],
        'clintox_assumed_overlap_positive_count': results['h']['chiral_sensitivity_positive_count'],
        'clintox_achiral_overlap_rows': results['h']['primary_rows'],
        'clintox_achiral_overlap_positive_count': results['h']['primary_positive_count'],
        'recomputed_panels': ['d', 'e', 'f', 'h'],
        'blocked_computational_panels': ['b', 'c', 'g'],
        'panel_directories': {letter: str(FIGURE / f'panel_{letter}') for letter in results},
    }
    (SHARED / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
