"""Prepare compound-level DrugBank scores for Supplementary Figure 3.

The input schema specifies the required drug classes and cell-line scores.
Missing inputs are recorded in a status report."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
ANALYSIS=Path(__file__).resolve().parent
REPOSITORY=Path('/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private')
INPUT=REPOSITORY/'datasets/external_validation/drugbank/approved_systemic_antibiotics_and_antineoplastics.csv'
SCORES=['hepg2_score','hskmc_score','imr90_score']
GROUPS=['systemic_antibiotic','antineoplastic']
THRESHOLD=.2


def prepare():
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    schema={'file':str(INPUT),'required_columns':{'compound_id':'Unique DrugBank identifier within each class',
             'smiles':'Compound structure used for the predictions',
             'drug_class':'systemic_antibiotic or antineoplastic',
             **{c:'SafetyNet probability in [0,1]' for c in SCORES}},
            'required_companion_metadata':['DrugBank release and approval/systemic-use/category curation rules',
                'SafetyNet checkpoint identities, embedding version and score aggregation',
                'Any compound exclusions or duplicate-resolution rules'],
            'analysis_rule':'Maximum of three cell-line scores; safe if score <0.2; predicted toxic if score >=0.2.',
            'histogram_rule':'Probability density; 20 equal bins over [0,1]. KDE uses Scott bandwidth if scores vary.'}
    (ANALYSIS/'required_input_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    if not INPUT.is_file():
        status={'status':'blocked_missing_compound_level_data','required_input':str(INPUT),
                'panels_recomputed':[],'reason':'Inputs: DrugBank compound lists and per-compound SafetyNet scores.',
                'artwork_counts_used_as_analysis_inputs':False,'plots_generated':False}
        (ANALYSIS/'validation.json').write_text(json.dumps(status,indent=2)+'\n')
        return None,status
    import numpy as np
    import pandas as pd
    data=pd.read_csv(INPUT)
    expected=['compound_id','smiles','drug_class',*SCORES]
    if missing:=set(expected)-set(data.columns):raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data[expected].isna().any().any():raise ValueError('Missing identifiers, structures, categories or scores require explicit curation.')
    if data.duplicated(['compound_id','drug_class']).any():raise ValueError('Duplicate compound/category rows would bias counts and densities.')
    if set(data.drug_class)!=set(GROUPS):raise ValueError('Expected both documented drug classes and no unclassified records.')
    if not np.isfinite(data[SCORES].to_numpy(dtype=float)).all():raise ValueError('Prediction scores must be finite.')
    if not data[SCORES].ge(0).all().all() or not data[SCORES].le(1).all().all():raise ValueError('Prediction scores must lie in [0,1].')
    data['max_toxicity_score']=data[SCORES].max(axis=1)
    data[expected+['max_toxicity_score']].to_csv(ANALYSIS/'analysis_cohort.csv',index=False)
    status={'status':'ready_for_panel_analysis','input':str(INPUT),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            'cohort_sizes':data.drug_class.value_counts().to_dict(),'threshold':THRESHOLD,
            'artwork_counts_used_as_analysis_inputs':False,
            'limitation':'Histogram and kernel-density settings are documented in required_input_schema.json.'}
    (ANALYSIS/'validation.json').write_text(json.dumps(status,indent=2)+'\n')
    return data,status


if __name__=='__main__':
    print(json.dumps(prepare()[1],indent=2))
