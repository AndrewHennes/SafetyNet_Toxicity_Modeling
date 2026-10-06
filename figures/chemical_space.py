"""Compute molecular chemical-space projections from molecular tables.

Fingerprint, dimensionality-reduction and t-SNE parameters are recorded in
the analysis manifest alongside molecule identifiers and source hashes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = Path('/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private')
DERIVED = REPOSITORY/'datasets/derived/chemical_space'
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib_cache'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(HERE/'.numba_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import rdFingerprintGenerator

SOURCES = {
 'internal_100k':(REPOSITORY/'datasets/toxicity_datasets/safetynet_100k_cytotoxicity_dataset.csv','SMILES',','),
 'tox21':(REPOSITORY/'datasets/external_validation/tox21/tox21_labels_and_safetynet_predictions.csv','Standardised_SMILES',','),
 'internal_39k':(REPOSITORY/'datasets/antibiotic_activity_datasets/broad_39k.csv','SMILES',','),
 'ecbd':(REPOSITORY/'datasets/antibiotic_activity_datasets/ecbd.csv','SMILES',','),
 'coadd':(REPOSITORY/'datasets/antibiotic_activity_datasets/coadd.csv','SMILES',','),
 'drug_repurposing_hub':(REPOSITORY/'datasets/external_validation/drug_repurposing_hub/drug_repurposing_hub_ec_hepg2_labels_and_predictions.tsv','SMILES','\t'),
}
GROUPS = {'cytotoxicity':['internal_100k','tox21'],
          'antibacterial':['internal_39k','ecbd','coadd','drug_repurposing_hub']}
COLORS={'internal_100k':'#00a9e0','tox21':'#8052a2','internal_39k':'#00a9e0',
        'ecbd':'#d64ca1','coadd':'#8052a2','drug_repurposing_hub':'#354185'}
LABELS={'internal_100k':'Internal 100K','tox21':'Tox21','internal_39k':'Internal 39K',
        'ecbd':'ECBD','coadd':'CO-ADD','drug_repurposing_hub':'Drug Repurposing Hub'}
PARAMETERS={'fingerprint':'RDKit Morgan binary','radius':2,'bits':2048,'include_chirality':False,
            'molecule_policy':'First source row per RDKit canonical isomeric SMILES within each library; all valid unique molecules, no sampling',
            'reduction':'TruncatedSVD, uncentered binary fingerprints','svd_components':50,'svd_iterations':7,
            'tsne_perplexity':30,'tsne_neighbors':'pynndescent','tsne_metric':'euclidean on SVD components',
            'tsne_initialization':'pca','tsne_early_exaggeration_iterations':250,'tsne_iterations':750,
            'tsne_negative_gradient':'fft','seed':42,'threads':12}


def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def parse_smiles(value):
    if not isinstance(value,str):
        return None
    if '|' in value:
        graph,separator,extension=value.partition('|')
        value=graph.rstrip()+' '+separator+extension
    return Chem.MolFromSmiles(value)


def source_fingerprints(names):
    """Return sparse fingerprints, row traceability, and exclusions for each set."""
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=PARAMETERS['radius'],
                    fpSize=PARAMETERS['bits'],includeChirality=PARAMETERS['include_chirality'])
    indices,indptr,records,excluded,counts=[],[0],[],[],{}
    RDLogger.DisableLog('rdApp.error')
    try:
        for name in names:
            path,column,separator=SOURCES[name]
            source=pd.read_csv(path,sep=separator,usecols=[column])[column]
            seen=set(); invalid=0; duplicates=0; first_count=len(records)
            for source_row,value in enumerate(source):
                molecule=parse_smiles(value)
                if molecule is None:
                    invalid+=1
                    excluded.append({'dataset':name,'source_row':source_row,'reason':'invalid_smiles','smiles':value})
                    continue
                canonical=Chem.MolToSmiles(molecule,isomericSmiles=True)
                if canonical in seen:
                    duplicates+=1
                    continue
                seen.add(canonical)
                bits=list(generator.GetFingerprint(molecule).GetOnBits())
                indices.extend(bits);indptr.append(len(indices))
                records.append({'dataset':name,'source_row':source_row,'canonical_smiles':canonical})
            counts[name]={'input_rows':len(source),'valid_unique_molecules':len(records)-first_count,
                          'invalid_smiles':invalid,'duplicate_valid_molecules':duplicates}
            print(name,counts[name],flush=True)
    finally:
        RDLogger.EnableLog('rdApp.error')
    matrix=sparse.csr_matrix((np.ones(len(indices),dtype=np.float32),
                             np.asarray(indices,dtype=np.int32),np.asarray(indptr,dtype=np.int64)),
                            shape=(len(records),PARAMETERS['bits']))
    return matrix,pd.DataFrame(records),pd.DataFrame(excluded),counts


def compute_group(group):
    """Compute coordinates when any molecule source or numerical setting changes."""
    import openTSNE
    import sklearn
    DERIVED.mkdir(parents=True,exist_ok=True)
    names=GROUPS[group]
    signature={name:sha256(SOURCES[name][0]) for name in names}
    versions={'rdkit':rdBase.rdkitVersion,'open_tsne':openTSNE.__version__,
              'scikit_learn':sklearn.__version__,'numpy':np.__version__}
    manifest_path=DERIVED/f'{group}_analysis_manifest.json'
    coordinates_path=DERIVED/f'{group}_tsne_coordinates.csv'
    if manifest_path.exists() and coordinates_path.exists():
        previous=json.loads(manifest_path.read_text())
        if (previous['source_sha256']==signature and previous['parameters']==PARAMETERS
                and previous['versions']==versions and previous.get('coordinates_sha256')==sha256(coordinates_path)):
            previous["source_files"]={name:str(SOURCES[name][0]) for name in names}
            previous["source_row_index_base"]=0
            serialized = json.dumps(previous, indent=2) + "\n"
            if manifest_path.read_text() != serialized:
                manifest_path.write_text(serialized)
            return pd.read_csv(coordinates_path),previous
    started=time.monotonic()
    matrix,rows,excluded,counts=source_fingerprints(names)
    excluded.to_csv(DERIVED/f'{group}_excluded_molecules.csv',index=False)
    print(f'{group}: reducing {matrix.shape[0]} fingerprints',flush=True)
    with threadpool_limits(limits=PARAMETERS['threads']):
        model=TruncatedSVD(n_components=PARAMETERS['svd_components'],n_iter=PARAMETERS['svd_iterations'],random_state=PARAMETERS['seed'])
        reduced=model.fit_transform(matrix).astype(np.float32)
    if not np.isfinite(reduced).all():
        raise ValueError("Non-finite molecular components; projection cannot proceed.")
    del matrix
    rows['svd_1']=reduced[:,0];rows['svd_2']=reduced[:,1]
    initial_path=DERIVED/f'{group}_svd_initial_coordinates.csv'
    rows.to_csv(initial_path,index=False)
    print(f'{group}: optimizing t-SNE for every included molecule',flush=True)
    tsne=openTSNE.TSNE(perplexity=PARAMETERS['tsne_perplexity'],
                      n_jobs=PARAMETERS['threads'],neighbors=PARAMETERS['tsne_neighbors'],
                      negative_gradient_method=PARAMETERS['tsne_negative_gradient'],
                      initialization=PARAMETERS['tsne_initialization'],random_state=PARAMETERS['seed'],
                      early_exaggeration_iter=PARAMETERS['tsne_early_exaggeration_iterations'],
                      n_iter=PARAMETERS['tsne_iterations'],metric='euclidean',verbose=True)
    embedded=tsne.fit(reduced)
    if not np.isfinite(np.asarray(embedded)).all():
        raise ValueError("Non-finite t-SNE coordinates.")
    rows['tsne_1']=np.asarray(embedded)[:,0];rows['tsne_2']=np.asarray(embedded)[:,1]
    rows.to_csv(coordinates_path,index=False)
    report={'source_sha256':signature,'source_files':{name:str(SOURCES[name][0]) for name in names},
            'parameters':PARAMETERS,'versions':versions,'cohorts':counts,'source_row_index_base':0,
            'svd_explained_variance_sum':float(model.explained_variance_ratio_.sum()),
            'tsne_kl_divergence':float(embedded.kl_divergence),
            'coordinates_sha256':sha256(coordinates_path),'elapsed_seconds':time.monotonic()-started,
            'limitations':['t-SNE geometry depends on the fingerprint, preprocessing, initialization and optimization settings.',
                           'Projection parameters and the random seed are recorded in this manifest.',
                           'Drug Repurposing Hub uses the supplied matched subset.',
                           'Figures 3a and 4a share the four-library fit; Figure 4 hides the Drug Repurposing Hub points without refitting.'],
            'implementation_reference':'https://opentsne.readthedocs.io/en/stable/api/index.html'}
    manifest_path.write_text(json.dumps(report,indent=2)+'\n')
    return rows,report


def export_panel(figure_name,output=None):
    group='cytotoxicity' if figure_name=='figure_1' else 'antibacterial'
    data,report=compute_group(group)
    names=GROUPS[group] if figure_name!='figure_4' else GROUPS[group][:-1]
    panel='b' if figure_name=='figure_1' else 'a'
    output=Path(output).resolve() if output else HERE/figure_name/f'panel_{panel}'
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none'})
    panel='b' if figure_name=='figure_1' else 'a'
    for dimensions,stem,title in [(('svd_1','svd_2'),'chemical_space_initial_svd','Initial fingerprint projection'),
                                  (('tsne_1','tsne_2'),'chemical_space_tsne','Morgan fingerprint t-SNE')]:
        fig,ax=plt.subplots(figsize=(6.7,5.0),layout='constrained')
        for name in names:
            points=data.loc[data.dataset.eq(name)]
            ax.scatter(points[dimensions[0]],points[dimensions[1]],s=.5,alpha=.55,c=COLORS[name],
                       label=f'{LABELS[name]} (n={len(points):,})',linewidths=0,rasterized=True)
        ax.legend(markerscale=6,frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1))
        ax.set_title(f'{panel}  {title}',loc='left',fontsize=11)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        caption='Morgan radius 2; t-SNE perplexity 30; random seed 42.'
        if figure_name=='figure_1':caption+=' FDA cohort missing.'
        fig.supxlabel(caption,fontsize=7)
        fig.savefig(output/f'{stem}.svg',dpi=180)
        fig.savefig(output/f'{stem}.png',dpi=180)
        plt.close(fig)
    (output/'chemical_space_status.json').write_text(json.dumps({
        'status':'recomputed_with_explicit_assumptions','panel':panel,'libraries':names,
        'coordinates':str(DERIVED/f'{group}_tsne_coordinates.csv'),'analysis_manifest':str(DERIVED/f'{group}_analysis_manifest.json'),
        'source_artwork_used':False},indent=2)+'\n')
    return report


def main():
    for name in ('figure_1','figure_3','figure_4'):
        export_panel(name)


if __name__=='__main__':
    main()
