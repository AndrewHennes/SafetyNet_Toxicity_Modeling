"""Calculate cytotoxicity metrics for Figure 2 and Supplementary Figure 2.

Functions define cohort eligibility, score aggregation, structural alerts,
molecular similarity, classification metrics and sensitivity analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, roc_auc_score)

REPOSITORY = Path('/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private')
SOURCE_FILES = {
    'tox21': ('datasets/external_validation/tox21/tox21_labels_and_safetynet_predictions.csv',
              'original_merged_tox21_results_with_minimol_representations_with_tox_predictions_only_tox_related_columns_with_maxsim_to_reference.csv'),
    'ecbd_restricted': ('datasets/external_validation/ecbd/ecbd_safetynet_predictions_with_structural_alerts.csv',
                       'ecbd_mtl_predictions_with_sim_with_structural_alerts.csv'),
    'ecbd_merged': ('datasets/external_validation/ecbd/ecbd_merged_panel_predictions.csv',
                   'ecbd_all_panels_merged_by_eos_with_minimol_representation_with_activity_and_tox_just_mean_of_ensemble_members.csv'),
    'ecbd_assay': ('datasets/external_validation/ecbd/ecbd_assay_labels_predictions_and_filters.csv',
                  'formatted_ecbd_antibiotic_toxicity_data_combined_with_tox_and_antibiotic_scores_and_filters.csv'),
    'ecbd_members': ('datasets/external_validation/ecbd/ecbd_embeddings_and_ensemble_predictions.tsv',
                    'formatted_ecbd_antibiotic_toxicity_data_combined_with_minimol_fingerprints_with_tox_and_antibiotic_scores.tsv'),
    'clintox': ('datasets/external_validation/clintox/clintox_labels_predictions_and_filters.csv',
                'clintox_combined_with_minimol_representations_with_tox_and_antibiotic_scores_with_tox_filters_with_metrics.csv'),
}
ALERTS = {
    'glaxo': 'CHEMBL_Glaxo', 'dundee': 'CHEMBL_Dundee', 'bms': 'CHEMBL_BMS',
    'brenk': 'BRENK', 'surechembl': 'CHEMBL_SureChEMBL', 'mlsmr': 'CHEMBL_MLSMR',
    'inpharmatica': 'CHEMBL_Inpharmatica', 'lint': 'CHEMBL_LINT',
}


def normalized_cxsmiles(value):
    """Repair only the missing space before a CXSMILES extension."""
    if isinstance(value, str) and '|' in value:
        graph, separator, extension = value.partition('|')
        if graph and not graph[-1].isspace():
            return graph + ' ' + separator + extension
    return value


def input_path(key: str) -> Path:
    """Resolve the canonical organized dataset location."""
    final, _legacy = SOURCE_FILES[key]
    target = REPOSITORY / final
    if target.is_file():
        return target
    raise FileNotFoundError(f'Missing {key} input. Expected {target}')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = input_path(key)
    delimiter = '\t' if path.suffix == '.tsv' else ','
    # Embeddings and redundant precomputed filter metrics are never loaded.
    selected = (lambda c: c in columns) if columns is not None else (
        lambda c: 'minimol' not in c.lower() and not c.startswith('Unnamed:'))
    data = pd.read_csv(path, sep=delimiter, usecols=selected, low_memory=False)
    data.insert(0, 'source_row', np.arange(len(data), dtype=int))
    data.insert(0, 'row_id', [f'{key}:{i}' for i in range(len(data))])
    return data


def evaluated_metrics(labels, scores, cutoff: float = .2) -> dict:
    """Compute discrimination and classification metrics with explicit conventions."""
    y = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels and scores must be aligned 1D arrays with binary labels.')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite probabilities in [0, 1].')
    y = y.astype(int)
    if not len(y):
        return {'n': 0, 'positive_count': 0, 'status': 'empty'}
    result = {'n': int(len(y)), 'positive_count': int(y.sum()),
              'prevalence': float(y.mean()), 'cutoff': cutoff,
              'status': 'evaluated' if len(np.unique(y)) == 2 else 'single_class'}
    if len(np.unique(y)) < 2:
        result.update({key: float('nan') for key in ['auroc', 'average_precision', 'pr_trapezoid_auc']})
    else:
        precision, recall, _ = precision_recall_curve(y, scores)
        result.update(auroc=float(roc_auc_score(y, scores)),
                      average_precision=float(average_precision_score(y, scores)),
                      pr_trapezoid_auc=float(auc(recall, precision)))
    prediction = scores >= cutoff
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                  mcc=(float(matthews_corrcoef(y, prediction)) if len(np.unique(y)) == 2 else float('nan')),
                  f1=float(f1_score(y, prediction, zero_division=0)),
                  precision=float(precision_score(y, prediction, zero_division=0)),
                  balanced_accuracy=(float(balanced_accuracy_score(y, prediction))
                                     if len(np.unique(y)) == 2 else float('nan')),
                  exact_score_cutoff_count=int(np.count_nonzero(scores == cutoff)))
    return result


def eligibility(frame: pd.DataFrame, label: str, score: str) -> pd.Series:
    y = pd.to_numeric(frame[label], errors='coerce')
    s = pd.to_numeric(frame[score], errors='coerce')
    return y.isin([0, 1]) & np.isfinite(s) & s.between(0, 1)


def similarity_mask(values: pd.Series, policy: str) -> pd.Series:
    """Missing similarities remain excluded from every novelty-filtered cohort."""
    if policy == 'unfiltered':
        return pd.Series(True, index=values.index)
    if policy == 'similarity_le_0_9':
        return values.between(0, .9)
    if policy == 'similarity_lt_0_9':
        return values.between(0, 1) & values.lt(.9)
    if policy == 'similarity_lt_1':
        return values.between(0, 1) & values.lt(1)
    raise ValueError(f'Unrecognized similarity policy: {policy}')


def tox21_category(name: str) -> tuple[int, str]:
    if any(token in name for token in ['-casp3-', '-herg-', '-rt-viability-']):
        return 5, 'Functional cytotoxicity / ion channels'
    if any(token in name for token in ['-ache-', '-p450-', '-aromatase-', '-hdac-']):
        return 1, 'Enzyme activities'
    if any(token in name for token in ['-p53-', '-elg1-', '-are-', '-dt40-']):
        return 2, 'Cellular stress responses'
    if any(token in name for token in ['-ap1-', '-nfkb-', '-sbe-', '-shh-']):
        return 3, 'Signaling pathways'
    return 4, 'Receptors / hormone signaling'


def analyze_tox21(output: Path) -> pd.DataFrame:
    """Compute every evaluable assay from labels and molecular prediction scores."""
    output.mkdir(parents=True, exist_ok=True)
    frame = read_rows('tox21')
    assay_columns = [c for c in frame if c.startswith('tox21-') and '_Activity__' in c]
    similarity = pd.to_numeric(frame.max_tanimoto_to_reference, errors='coerce')
    task_scores = ['Mean_HepG2_Toxicity', 'Mean_HSkMC_Toxicity', 'Mean_IMR90_Toxicity']
    frame['recomputed_max_toxicity'] = frame[task_scores].max(axis=1, skipna=False)
    score_difference = (frame.Max_Tox_Score - frame.recomputed_max_toxicity).abs()
    policies = ['unfiltered', 'similarity_le_0_9', 'similarity_lt_0_9', 'similarity_lt_1']
    metrics, exclusions = [], []
    # The 42-assay set is recovered by requiring both classes, never by reading
    # the precomputed 42-row performance table.
    for policy in policies:
        cohort = similarity_mask(similarity, policy)
        for assay in assay_columns:
            valid = eligibility(frame, assay, 'recomputed_max_toxicity') & cohort
            category, category_name = tox21_category(assay)
            for score_name in ['recomputed_max_toxicity', *task_scores]:
                row = evaluated_metrics(frame.loc[valid, assay], frame.loc[valid, score_name])
                row.update(assay=assay, assay_id=assay.split('_Activity__')[0].removeprefix('tox21-'),
                           policy=policy, score=score_name, category=category,
                           category_name=category_name,
                           cohort_rows=int(cohort.sum()),
                           missing_or_invalid_label_score=int((cohort & ~valid).sum()))
                metrics.append(row)
            if policy == 'similarity_le_0_9':
                missing = cohort & ~valid
                exclusions.append({'assay': assay, 'cohort_rows': int(cohort.sum()),
                                   'evaluated_rows': int(valid.sum()),
                                   'excluded_label_or_score': int(missing.sum()),
                                   'positive_count': int(frame.loc[valid, assay].sum()),
                                   'negative_count': int(valid.sum() - frame.loc[valid, assay].sum()),
                                   'single_class': frame.loc[valid, assay].nunique() < 2})
    result = pd.DataFrame(metrics)
    result.to_csv(output / 'tox21_metrics_all_policies_and_scores.csv', index=False)
    selected = result.loc[result.policy.eq('similarity_le_0_9') &
                          result.score.eq('recomputed_max_toxicity') &
                          result.status.eq('evaluated')].copy()
    selected.to_csv(output / 'tox21_assay_metrics.csv', index=False)
    pd.DataFrame(exclusions).to_csv(output / 'tox21_assay_exclusions.csv', index=False)
    frame['similarity_le_0_9'] = similarity.le(.9)
    frame['similarity_lt_0_9'] = similarity.lt(.9)
    frame['similarity_lt_1'] = similarity.lt(1)
    frame['similarity_missing'] = similarity.isna()
    # Wide cohort table preserves per-assay missing labels without conflating
    # missingness with inactive compounds or duplicating molecular identities.
    frame.to_csv(output / 'tox21_molecular_cohort_and_labels.csv', index=False)
    provenance = {
        'input': str(input_path('tox21')), 'sha256': sha256(input_path('tox21')),
        'source_rows': len(frame), 'assay_columns': len(assay_columns),
        'evaluable_primary_assays': len(selected),
        'primary_cohort_rows': int(similarity.le(.9).sum()),
        'exact_similarity_0_9': int(similarity.eq(.9).sum()),
        'missing_similarity': int(similarity.isna().sum()),
        'invalid_similarity': int((similarity.notna() & ~similarity.between(0, 1)).sum()),
        'primary_similarity_rule': 'max_tanimoto_to_reference <= 0.9',
        'source_score_check_max_absolute_difference': float(score_difference.max()),
        'score_rule': 'max of three supplied per-task ensemble mean scores',
        'class_rule': 'native binary assay labels; per-assay missing labels excluded',
        'single_class_assays': pd.DataFrame(exclusions).query('single_class').assay.tolist(),
        'duplicate_smiles_rows_retained': int(frame.Standardised_SMILES.duplicated().sum()),
        'metric_definitions': {'auroc': 'sklearn roc_auc_score',
                              'average_precision': 'sklearn average_precision_score',
                              'pr_trapezoid_auc': 'sklearn auc(recall, precision)'},
        'limitations': ['Analysis uses the supplied similarity column.',
                       'This analysis evaluates saved prediction scores. Model training and fold assignment are separate inputs.'],
    }
    (output / 'tox21_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return selected


def compute_alerts(frame: pd.DataFrame, smiles_column: str, output: Path,
                   prefix: str) -> pd.DataFrame:
    """Apply eight actual RDKit structural alert catalogs to full source molecules."""
    from rdkit import Chem, rdBase, RDLogger
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDLogger.DisableLog('rdApp.error')
    catalogs = {name: FilterCatalog(FilterCatalogParams(getattr(FilterCatalogParams.FilterCatalogs, catalog)))
                for name, catalog in ALERTS.items()}
    valid, rows = [], []
    cache = {}
    for value in frame[smiles_column]:
        text = value if isinstance(value, str) else ''
        if text not in cache:
            molecule = Chem.MolFromSmiles(normalized_cxsmiles(text)) if text else None
            if molecule is None or molecule.GetNumAtoms() == 0:
                cache[text] = (False, {name: np.nan for name in ALERTS})
            else:
                cache[text] = (True, {name: int(catalog.HasMatch(molecule))
                                     for name, catalog in catalogs.items()})
        ok, flags = cache[text]
        valid.append(ok)
        rows.append(flags)
    result = frame.copy()
    result['structure_valid_for_alerts'] = valid
    computed = pd.DataFrame(rows, index=frame.index)
    comparisons = []
    for name in ALERTS:
        result[f'recomputed_{name}'] = computed[name]
        if name in frame:
            matched = pd.to_numeric(frame[name], errors='coerce').isin([0, 1]) & np.asarray(valid)
            comparisons.append({'method': name, 'compared_rows': int(matched.sum()),
                                'disagreements': int((frame.loc[matched, name] != computed.loc[matched, name]).sum()),
                                'supplied_positive': int(frame.loc[matched, name].sum()),
                                'recomputed_positive': int(computed.loc[matched, name].sum())})
    result.to_csv(output / f'{prefix}_cohort_with_recomputed_alerts.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / f'{prefix}_provided_vs_recomputed_alerts.csv', index=False)
    (output / f'{prefix}_alert_provenance.json').write_text(json.dumps({
        'rdkit_version': rdBase.rdkitVersion, 'catalogs': ALERTS,
        'structure_policy': 'Full molecule including disconnected fragments; missing CXSMILES separator repaired, no salt stripping or neutralization.',
        'input_rows': len(frame), 'invalid_structure_rows': int((~np.asarray(valid)).sum()),
        'deduplicated_for_computation_only': len(cache),
        'classification_rule': 'At least one catalog match means a positive structural alert.',
        'invalid_structure_policy': 'Excluded from all methods in the corresponding filter comparison.'}, indent=2) + '\n')
    return result


def filter_metrics(frame: pd.DataFrame, label: str, score: str, cohort: str) -> pd.DataFrame:
    valid = frame.structure_valid_for_alerts & eligibility(frame, label, score)
    evaluated = frame.loc[valid]
    rows = []
    for name in ['safetynet', *ALERTS]:
        values = evaluated[score] if name == 'safetynet' else evaluated[f'recomputed_{name}']
        metrics = evaluated_metrics(evaluated[label], values, cutoff=.2 if name == 'safetynet' else .5)
        metrics.update(cohort=cohort, method=name,
                       score_or_flag=score if name == 'safetynet' else f'recomputed_{name}',
                       excluded_invalid_structure_or_label_score=int((~valid).sum()))
        rows.append(metrics)
    return pd.DataFrame(rows)


def analyze_ecbd(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit each supplied ECBD cohort independently, without optimizing a fit."""
    output.mkdir(parents=True, exist_ok=True)
    versions = {
        'ecbd_restricted': {
            'label': 'label_hepg2', 'value': 'value_hepg2', 'score': 'MaxTox',
            'tasks': ['HepG2_MTL', 'HSkMC_MTL', 'IMR90_MTL'], 'similarity': 'maxsim_100K'},
        'ecbd_merged': {
            'label': 'activity_HepG2', 'value': 'value_HepG2', 'score': 'Max_Tox_Score',
            'tasks': ['Mean_HepG2_Toxicity', 'Mean_HSkMC_Toxicity', 'Mean_IMR90_Toxicity']},
        'ecbd_assay': {
            'label': 'HepG2_Binarized_Toxicity', 'value': 'HepG2_Percent_Growth_Inhibition',
            'score': 'Max_Tox_Score', 'tasks': []},
    }
    all_metrics, label_comparison, score_checks, provenance = [], [], [], []
    frames = {}
    primary = None
    for key, spec in versions.items():
        needed = ['SMILES', spec['label'], spec['value'], spec['score'], *spec['tasks'], *ALERTS]
        if 'similarity' in spec:
            needed += [spec['similarity'], 'MatchedBy', 'Name']
        frame = read_rows(key, needed)
        native = frame[spec['label']]
        frame['native_toxicity_label'] = (native.map({'active': 1, 'inactive': 0})
                                           if key == 'ecbd_merged' else pd.to_numeric(native, errors='coerce'))
        measurement = pd.to_numeric(frame[spec['value']], errors='coerce')
        frame['toxicity_gi_gt_20'] = measurement.gt(20).where(measurement.notna())
        frame['toxicity_gi_gt_50'] = measurement.gt(50).where(measurement.notna())
        scores = [spec['score']]
        if spec['tasks']:
            frame['max_of_three_task_scores'] = frame[spec['tasks']].max(axis=1, skipna=False)
            frame['mean_of_three_task_scores'] = frame[spec['tasks']].mean(axis=1, skipna=False)
            scores += [*spec['tasks'], 'max_of_three_task_scores', 'mean_of_three_task_scores']
            difference = (frame[spec['score']] - frame.max_of_three_task_scores).abs()
            score_checks.append({'dataset': key, 'comparison': 'stored max minus max of task means',
                                 'max_absolute_difference': float(difference.max()),
                                 'mean_absolute_difference': float(difference.mean())})
        policies = ['unfiltered']
        if 'similarity' in spec:
            policies += ['similarity_le_0_9', 'similarity_lt_0_9', 'similarity_lt_1']
            similarity = pd.to_numeric(frame[spec['similarity']], errors='coerce')
        else:
            similarity = pd.Series(np.nan, index=frame.index)
        for policy in policies:
            include = similarity_mask(similarity, policy)
            for label in ['native_toxicity_label', 'toxicity_gi_gt_20', 'toxicity_gi_gt_50']:
                for score in scores:
                    eligible = include & eligibility(frame, label, score)
                    metrics = evaluated_metrics(frame.loc[eligible, label], frame.loc[eligible, score])
                    metrics.update(dataset=key, policy=policy, label=label, score=score,
                                   cohort_rows=int(include.sum()),
                                   missing_or_invalid_label_score=int((include & ~eligible).sum()))
                    all_metrics.append(metrics)
        for label in ['toxicity_gi_gt_20', 'toxicity_gi_gt_50']:
            shared = frame.native_toxicity_label.isin([0, 1]) & frame[label].isin([0, 1])
            label_comparison.append({'dataset': key, 'alternative': label,
                                     'shared_rows': int(shared.sum()),
                                     'disagreements_with_native_label': int((frame.loc[shared, label] != frame.loc[shared, 'native_toxicity_label']).sum())})
        frame['included_primary_panel_d_e'] = False
        if key == 'ecbd_restricted':
            keep = similarity_mask(similarity, 'similarity_le_0_9') & eligibility(frame, 'native_toxicity_label', spec['score'])
            frame['included_primary_panel_d_e'] = keep
            primary = frame.loc[keep].copy()
        frame.to_csv(output / f'{key}_molecular_cohort.csv', index=False)
        frames[key] = frame
        provenance.append({'dataset': key, 'input': str(input_path(key)), 'sha256': sha256(input_path(key)),
                           'rows': len(frame), 'native_label': spec['label'], 'native_score': spec['score'],
                           'duplicate_smiles_rows': int(frame.SMILES.duplicated().sum()),
                           'similarity_available': 'similarity' in spec})
    comparisons = []
    restricted = frames['ecbd_restricted']
    unique_restricted = restricted.loc[~restricted.SMILES.duplicated(keep=False)]
    for key in ['ecbd_merged', 'ecbd_assay']:
        other = frames[key]
        unique_other = other.loc[~other.SMILES.duplicated(keep=False)]
        shared = unique_restricted.merge(unique_other, on='SMILES', suffixes=('_restricted', '_other'), validate='one_to_one')
        score_diff = shared.MaxTox - shared.Max_Tox_Score
        labels = shared.native_toxicity_label_restricted.isin([0, 1]) & shared.native_toxicity_label_other.isin([0, 1])
        comparisons.append({'first': 'ecbd_restricted', 'second': key,
                            'matching_rule': 'Exact raw SMILES, unique in both input tables',
                            'matched_rows': len(shared), 'max_absolute_score_difference': float(score_diff.abs().max()),
                            'mean_absolute_score_difference': float(score_diff.abs().mean()),
                            'fraction_scores_within_1e_6': float(score_diff.abs().le(1e-6).mean()),
                            'paired_known_labels': int(labels.sum()),
                            'label_disagreements': int((shared.loc[labels, 'native_toxicity_label_restricted'] != shared.loc[labels, 'native_toxicity_label_other']).sum())})
    pd.DataFrame(comparisons).to_csv(output / 'ecbd_input_version_comparison.csv', index=False)
    pd.DataFrame(label_comparison).to_csv(output / 'ecbd_label_definition_audit.csv', index=False)
    pd.DataFrame(score_checks).to_csv(output / 'ecbd_task_score_aggregation_audit.csv', index=False)
    pd.DataFrame(provenance).to_csv(output / 'ecbd_source_provenance.csv', index=False)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output / 'ecbd_cohort_score_label_sensitivity.csv', index=False)
    if primary is None:
        raise RuntimeError('ECBD primary cohort was not created.')
    return primary, metrics


def audit_ecbd_members(output: Path) -> dict:
    """Compute score aggregation alternatives from the actual six saved members."""
    path = input_path('ecbd_members')
    header = pd.read_csv(path, sep='\t', nrows=0).columns
    member_columns = [c for c in header if c.startswith('Ensemble_Member_') and '_prediction_task_' in c]
    needed = ['SMILES', 'HepG2_Binarized_Toxicity', 'HepG2_Percent_Growth_Inhibition',
              'Max_Tox_Score', 'Mean_HepG2_Toxicity', 'Mean_HSkMC_Toxicity', 'Mean_IMR90_Toxicity', *member_columns]
    frame = read_rows('ecbd_members', needed)
    member_ids = sorted({int(c.split('_')[2]) for c in member_columns})
    expected = [f'Ensemble_Member_{member}_prediction_task_{task}' for member in member_ids for task in range(3)]
    if set(expected) != set(member_columns):
        raise ValueError('Every ensemble member must supply all three task scores.')
    tensor = frame[expected].to_numpy(float).reshape(len(frame), len(member_ids), 3)
    if not np.isfinite(tensor).all() or ((tensor < 0) | (tensor > 1)).any():
        raise ValueError('Invalid per-member probabilities.')
    frame['max_of_member_mean_tasks'] = tensor.mean(axis=1).max(axis=1)
    frame['mean_of_member_max_tasks'] = tensor.max(axis=2).mean(axis=1)
    frame['hepg2_member_mean'] = tensor[:, :, 0].mean(axis=1)
    frame['label_gi_gt_20'] = frame.HepG2_Percent_Growth_Inhibition.gt(20).where(frame.HepG2_Percent_Growth_Inhibition.notna())
    frame['label_gi_gt_50'] = frame.HepG2_Percent_Growth_Inhibition.gt(50).where(frame.HepG2_Percent_Growth_Inhibition.notna())
    rows = []
    for label in ['HepG2_Binarized_Toxicity', 'label_gi_gt_20', 'label_gi_gt_50']:
        for score in ['Max_Tox_Score', 'max_of_member_mean_tasks', 'mean_of_member_max_tasks', 'hepg2_member_mean']:
            valid = eligibility(frame, label, score)
            row = evaluated_metrics(frame.loc[valid, label], frame.loc[valid, score])
            row.update(label=label, score=score, level='aggregate')
            rows.append(row)
        for index, member in enumerate(member_ids):
            for score_name, scores in [('hepg2', tensor[:, index, 0]), ('max_three_tasks', tensor[:, index].max(axis=1))]:
                valid = frame[label].isin([0, 1]).to_numpy()
                row = evaluated_metrics(frame.loc[valid, label], scores[valid])
                row.update(label=label, score=f'member_{member}_{score_name}', level='saved_member')
                rows.append(row)
    pd.DataFrame(rows).to_csv(output / 'ecbd_saved_member_and_aggregation_metrics.csv', index=False)
    retained = ['row_id', 'source_row', 'SMILES', 'HepG2_Binarized_Toxicity',
                'HepG2_Percent_Growth_Inhibition', 'Max_Tox_Score', 'max_of_member_mean_tasks',
                'mean_of_member_max_tasks', 'hepg2_member_mean']
    frame[retained].to_csv(output / 'ecbd_recomputed_ensemble_scores.csv', index=False)
    report = {'input': str(path), 'sha256': sha256(path), 'rows': len(frame),
              'members': member_ids, 'member_count': len(member_ids), 'task_count': 3,
              'stored_max_vs_max_of_task_means_max_abs_diff': float((frame.Max_Tox_Score - frame.max_of_member_mean_tasks).abs().max()),
              'stored_max_vs_mean_of_task_maxima_max_abs_diff': float((frame.Max_Tox_Score - frame.mean_of_member_max_tasks).abs().max()),
              'interpretation': 'Analysis uses saved ensemble members.'}
    (output / 'ecbd_ensemble_aggregation_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def analyze_clintox(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = ['SMILES', 'target', 'original_split', 'Max_Tox_Score',
              'Mean_HepG2_Toxicity', 'Mean_HSkMC_Toxicity', 'Mean_IMR90_Toxicity', *ALERTS]
    frame = read_rows('clintox', needed)
    frame['max_of_three_task_scores'] = frame[['Mean_HepG2_Toxicity', 'Mean_HSkMC_Toxicity', 'Mean_IMR90_Toxicity']].max(axis=1, skipna=False)
    frame['valid_label_score'] = eligibility(frame, 'target', 'max_of_three_task_scores')
    metrics = []
    for split in ['all', 'train', 'validation', 'test']:
        eligible = frame.valid_label_score & (True if split == 'all' else frame.original_split.eq(split))
        for score in ['Max_Tox_Score', 'max_of_three_task_scores', 'Mean_HepG2_Toxicity', 'Mean_HSkMC_Toxicity', 'Mean_IMR90_Toxicity']:
            row = evaluated_metrics(frame.loc[eligible, 'target'], frame.loc[eligible, score])
            row.update(original_split=split, score=score)
            metrics.append(row)
    result = pd.DataFrame(metrics)
    result.to_csv(output / 'clintox_score_and_historical_split_audit.csv', index=False)
    frame.to_csv(output / 'clintox_molecular_cohort.csv', index=False)
    (output / 'clintox_source_provenance.json').write_text(json.dumps({
        'input': str(input_path('clintox')), 'sha256': sha256(input_path('clintox')),
        'rows': len(frame), 'valid_label_score': int(frame.valid_label_score.sum()),
        'stored_max_vs_max_task_difference': float((frame.Max_Tox_Score - frame.max_of_three_task_scores).abs().max()),
        'base_cohort': 'All supplied labeled compounds with valid scores; novelty selection is computed separately',
        'split_policy': 'Historical split labels are sensitivity strata, not SafetyNet train/test assignments.',
        'similarity_filter': 'Computed from molecules using the fingerprint and reference-cohort parameters in clintox_overlap_sensitivity.',
        'duplicate_raw_smiles_rows_retained': int(frame.SMILES.duplicated().sum())}, indent=2) + '\n')
    return frame.loc[frame.valid_label_score].copy(), result


def clintox_overlap_sensitivity(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Calculate training-library overlap with Morgan fingerprints.

    The reference contains 100,352 rows. Radius and width follow the repository
    defaults, and both achiral and chiral variants are evaluated.
    """
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.error')
    reference_path = REPOSITORY / 'datasets/toxicity_datasets/safetynet_100k_cytotoxicity_dataset.csv'
    reference = pd.read_csv(reference_path, usecols=['SMILES'])
    settings = {'reference_sha256': sha256(reference_path),
                'query_sha256': sha256(input_path('clintox')),
                'reference_column': 'SMILES', 'query_column': 'SMILES',
                'radius': 2, 'fingerprint_bits': 2048, 'chirality_variants': [True, False],
                'rdkit_version': rdBase.rdkitVersion,
                'molecule_policy': 'Full submitted molecules; no salt stripping or neutralization.',
                'format_repair': 'Insert missing space before first CXSMILES pipe; preserve graph and extension metadata.'}
    cache = output / 'clintox_recomputed_training_similarity.csv'
    metadata_file = output / 'clintox_overlap_provenance.json'
    if cache.is_file() and metadata_file.is_file() and json.loads(metadata_file.read_text()).get('settings') == settings:
        similarity = pd.read_csv(cache)
    else:
        def parse(value):
            mol = Chem.MolFromSmiles(normalized_cxsmiles(value)) if isinstance(value, str) and value.strip() else None
            if mol is None or mol.GetNumAtoms() == 0 or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                return None
            return mol
        reference_molecules = []
        excluded_reference_rows = []
        repaired_reference_rows = []
        seen = set()
        for row, value in enumerate(reference.SMILES):
            if value in seen:
                continue
            seen.add(value)
            molecule = parse(value)
            if molecule is None:
                excluded_reference_rows.append({'source_row': row, 'smiles': value, 'reason': 'Invalid or unspecified molecular structure'})
            else:
                reference_molecules.append(molecule)
                if normalized_cxsmiles(value) != value:
                    repaired_reference_rows.append({'source_row': row, 'smiles': value, 'repair': 'Insert missing space before first CXSMILES pipe'})
        if not reference_molecules:
            raise ValueError('No valid reference molecules for similarity calculation.')
        query_molecules = [parse(value) for value in frame.SMILES]
        similarity = frame[['row_id', 'source_row', 'SMILES']].copy()
        for chirality in [True, False]:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=chirality)
            fingerprints = list(generator.GetFingerprints(reference_molecules, numThreads=8))
            values = []
            for mol in query_molecules:
                values.append(max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol), fingerprints))
                              if mol is not None else np.nan)
            similarity[f'max_training_tanimoto_chirality_{str(chirality).lower()}'] = values
        similarity.to_csv(cache, index=False)
        pd.DataFrame(excluded_reference_rows, columns=['source_row', 'smiles', 'reason']).to_csv(output / 'clintox_overlap_excluded_reference_structures.csv', index=False)
        pd.DataFrame(repaired_reference_rows, columns=['source_row', 'smiles', 'repair']).to_csv(output / 'clintox_overlap_cxsmiles_repairs.csv', index=False)
        metadata_file.write_text(json.dumps({
            'settings': settings, 'reference_path': str(reference_path), 'reference_rows': len(reference),
            'unique_reference_strings': len(seen), 'valid_unique_reference_molecules': len(reference_molecules),
            'invalid_unique_reference_structures': len(excluded_reference_rows),
            'repaired_unique_reference_structures': len(repaired_reference_rows),
            'invalid_query_structures': sum(m is None for m in query_molecules),
            'caveat': 'Achiral and chiral Morgan fingerprints define alternative overlap policies.'}, indent=2) + '\n')
    joined = frame.merge(similarity[['row_id', *[c for c in similarity if c.startswith('max_training_')]]], on='row_id', validate='one_to_one')
    rows = []
    for chirality in [True, False]:
        column = f'max_training_tanimoto_chirality_{str(chirality).lower()}'
        for policy in ['similarity_le_0_9', 'similarity_lt_0_9', 'similarity_lt_1']:
            mask = similarity_mask(joined[column], policy)
            metrics = filter_metrics(joined.loc[mask], 'target', 'max_of_three_task_scores', f'clintox_proxy_{policy}_chiral_{chirality}')
            metrics['chirality'] = chirality
            metrics['similarity_policy'] = policy
            metrics['excluded_by_overlap_or_invalid_similarity'] = int((~mask).sum())
            rows.append(metrics)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output / 'clintox_recomputed_overlap_filter_metrics.csv', index=False)
    joined['included_primary_panel_h'] = (similarity_mask(joined.max_training_tanimoto_chirality_false, 'similarity_le_0_9')
                                          & joined.structure_valid_for_alerts
                                          & eligibility(joined, 'target', 'max_of_three_task_scores'))
    joined.to_csv(output / 'clintox_cohort_alerts_and_overlap.csv', index=False)
    joined.loc[joined.included_primary_panel_h].to_csv(output / 'clintox_primary_molecular_cohort.csv', index=False)
    return result
