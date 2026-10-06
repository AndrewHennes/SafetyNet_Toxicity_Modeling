"""Prepare the antibacterial assay cohorts shared by Figure 4b and Figure 4c.

Run directly or import prepare(). Cohort rules, exclusions, source hashes and
validation results are recorded alongside this module."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parent / '.matplotlib_cache'))
import numpy as np
import pandas as pd

REPOSITORY = Path('/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private')
INPUT = REPOSITORY / 'datasets/antibiotic_activity_datasets/broad_39k.csv'
SPECIES = ['AB', 'EC', 'KP', 'PA']
DISPLAY = {'AB':'A. baumannii','EC':'E. coli','KP':'K. pneumoniae','PA':'P. aeruginosa'}
ORDER = ['AB','PA','EC','KP']
MEASUREMENTS = [f'{s}_Percent_Growth_Inhibition_Mean' for s in SPECIES]
FLAGS = [f'{s}_Hit_Binarized_at_80_Percent_Growth_Inhibition' for s in SPECIES]
THRESHOLD = 80.0
COLORS = ['#d7eff4','#a2dce5','#54bfd1','#008fac']
def hit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[MEASUREMENTS].isna().any().any():
        raise ValueError('Missing assay values must be excluded explicitly before binarization.')
    flags = frame[MEASUREMENTS].gt(THRESHOLD).astype(int)
    flags.columns = SPECIES
    return flags

def aggregate_compounds(frame: pd.DataFrame, identity: str) -> pd.DataFrame:
    """Give each compound one contribution; repeated distinct assay values are averaged.

    The source has exact duplicate records and one raw-SMILES compound with
    multiple distinct measurements. Averaging each endpoint's distinct supplied
    measurements avoids reweighting a value through duplicated/join-expanded rows.
    """
    numeric = frame.groupby(identity,sort=False)[MEASUREMENTS].agg(lambda x: x.drop_duplicates().mean())
    numeric = numeric.reset_index()
    if identity == 'SMILES':
        standard = frame.groupby(identity,sort=False).Standardized_SMILES.first()
        numeric['Standardized_SMILES'] = numeric.SMILES.map(standard)
    counts = frame.groupby(identity,sort=False).size()
    numeric['source_record_count'] = numeric[identity].map(counts).astype(int)
    numeric['source_rows'] = numeric[identity].map(frame.groupby(identity,sort=False).source_row.agg(lambda x: ';'.join(map(str,x))))
    for species,column in zip(SPECIES,MEASUREMENTS):
        numeric[f'{species}_hit'] = numeric[column].gt(THRESHOLD).astype(int)
    return numeric

def prepare() -> dict:
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    input_hash=hashlib.sha256(INPUT.read_bytes()).hexdigest()
    validation=ANALYSIS/'validation.json'
    required=[ANALYSIS/'analysis_cohort.csv',ANALYSIS/'excluded_missing_assays.csv',
              ANALYSIS/'repeated_compounds_with_distinct_measurements.csv',ANALYSIS/'invalid_structure_records.csv']
    if validation.is_file() and all(path.is_file() for path in required):
        previous=json.loads(validation.read_text())
        if previous.get('preparation_code_sha256')==code_hash and previous.get('source_sha256')==input_hash:
            return previous
    raw=pd.read_csv(INPUT)
    raw.insert(0,'source_row',np.arange(len(raw)))
    if set(raw.Concentration.unique()) != {50} or set(raw['Concentration Units'].unique()) != {'uM'}:
        raise ValueError('Expected only the documented 50 µM antibacterial assays.')
    missing=raw[MEASUREMENTS].isna().any(axis=1)
    complete=raw.loc[~missing].copy()
    source_flags=complete[FLAGS].to_numpy()
    derived_flags=hit_frame(complete).to_numpy()
    if not np.array_equal(source_flags,derived_flags):
        raise ValueError('Supplied hit flags disagree with >80% inhibition; resolve before plotting.')
    from rdkit import Chem, rdBase
    canonical={}
    with rdBase.BlockLogs():
        for text in raw.SMILES.unique():
            repaired=text
            if "|" in text:
                graph,pipe,extension=text.partition("|")
                repaired=graph.rstrip()+" "+pipe+extension
            molecule=Chem.MolFromSmiles(repaired)
            canonical[text]=Chem.MolToSmiles(molecule,isomericSmiles=True) if molecule is not None else None
    complete['canonical_smiles']=complete.SMILES.map(canonical)
    cohort=aggregate_compounds(complete,'SMILES')
    cohort['canonical_smiles']=cohort.SMILES.map(canonical)
    complete.loc[complete.canonical_smiles.isna(),['source_row','SMILES']].to_csv(ANALYSIS/'invalid_structure_records.csv',index=False)
    cohort.to_csv(ANALYSIS/'analysis_cohort.csv',index=False)
    raw.loc[missing,['source_row','SMILES','Standardized_SMILES']+MEASUREMENTS].to_csv(ANALYSIS/'excluded_missing_assays.csv',index=False)
    multi=complete.groupby('SMILES')[MEASUREMENTS].nunique().max(axis=1)
    conflicting=multi.index[multi.gt(1)]
    complete.loc[complete.SMILES.isin(conflicting)].to_csv(ANALYSIS/'repeated_compounds_with_distinct_measurements.csv',index=False)
    mean_errors={}
    for s in SPECIES:
        expected=(complete[f'{s}_OD600_Rep1']+complete[f'{s}_OD600_Rep2'])/2
        mean_errors[s]=float((expected-complete[f'{s}_OD600_Mean']).abs().max())
    report={'source_csv':str(INPUT),'source_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            'source_rows':len(raw),'missing_assay_rows_excluded':int(missing.sum()),'missing_unique_smiles':int(raw.loc[missing,'SMILES'].nunique()),
            'complete_source_rows':len(complete),'complete_unique_raw_smiles':len(cohort),'paper_compound_count_reference_only':39267,
            'raw_unique_smiles_including_missing':int(raw.SMILES.nunique()),
            'canonical_unique_smiles_including_missing':len(set(x for x in canonical.values() if x is not None)),
            'invalid_unique_raw_smiles':sum(x is None for x in canonical.values()),'rdkit_version':rdBase.rdkitVersion,'repeated_compounds_with_distinct_measurements':len(conflicting),
            'source_flags_equal_recomputed_80_percent_rule':True,'replicate_od_mean_max_absolute_errors':mean_errors,
            'definition':'One raw-SMILES compound per row; exclude missing assays; average distinct reported inhibition measurements per endpoint; hit iff >80%.',
            'normalization':'Supplied percent inhibition.',
            'correlation_definition':'Primary: Pearson on continuous percent inhibition, as manuscript prose states. Binary Pearson is a separately labeled diagnostic of the figure title.',
            'blocked_panels':{'d':'Conceptual diagram, preserved only as source artwork.',
                              'e':'Virtual-library selection and ranking records.',
                              'f_g_h_i':'Prospective compound-level antibacterial and cytotoxicity measurements.'},
            'panel_a':'Computed by the shared chemical-space pipeline using its recorded parameters.',
            'no_artwork_values_used_in_plots':True}
    report['preparation_code_sha256']=code_hash
    (ANALYSIS/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def load_cohorts():
    """Load the shared complete-record and compound-level cohorts with exact float round trips."""
    report=prepare()
    raw=pd.read_csv(INPUT)
    raw.insert(0,'source_row',np.arange(len(raw)))
    complete=raw.loc[~raw[MEASUREMENTS].isna().any(axis=1)].copy()
    cohort=pd.read_csv(ANALYSIS/'analysis_cohort.csv',float_precision='round_trip')
    complete['canonical_smiles']=complete.SMILES.map(cohort.set_index('SMILES').canonical_smiles)
    return complete,cohort,report


def cohort_scenarios(complete,cohort):
    """Shared sensitivity selections; each panel calculates its own statistics."""
    return {'initial_all_complete_records':complete,
            'exact_record_deduplication':complete.drop_duplicates(subset=[c for c in complete.columns if c!='source_row']),
            'raw_smiles_first_record':complete.drop_duplicates('SMILES',keep='first'),
            'raw_smiles_last_record':complete.drop_duplicates('SMILES',keep='last'),
            'refined_raw_smiles_distinct_measurement_mean':cohort,
            'standardized_smiles_distinct_measurement_mean':aggregate_compounds(complete,'Standardized_SMILES'),
            'canonical_raw_smiles_distinct_measurement_mean':aggregate_compounds(complete.dropna(subset=['canonical_smiles']),'canonical_smiles')}


if __name__=='__main__':
    print(json.dumps(prepare(),indent=2))
