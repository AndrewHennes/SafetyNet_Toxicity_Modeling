"""Compute Supplementary Figure 3b only when actual DrugBank scores are supplied."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
ANALYSIS=HERE
spec=importlib.util.spec_from_file_location('supplementary_figure_3_shared',HERE.parent/'shared'/'prepare.py')
shared=importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main() -> dict:
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    data,preparation_status=shared.prepare()
    status={**preparation_status,'panel':'b','shared_validation':str(shared.ANALYSIS/'validation.json')}
    if data is None:
        (ANALYSIS/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        return status
    import numpy as np
    import pandas as pd
    os.environ.setdefault('MPLCONFIGDIR',str(HERE.parents[1]/'.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','svg.hashsalt':'safetynet_drugbank_analysis'})
    labels={'systemic_antibiotic':'Systemic antibiotics','antineoplastic':'Antineoplastic agents'}
    from scipy.stats import gaussian_kde
    fig,ax=plt.subplots(figsize=(4.6,3.7))
    edges=np.linspace(0,1,21);density_rows=[];histogram_rows=[]
    for group,color in zip(shared.GROUPS,['#d4509c','#8050a5']):
        scores=data.loc[data.drug_class.eq(group),'max_toxicity_score'].to_numpy()
        heights,_=np.histogram(scores,bins=edges,density=True)
        ax.hist(scores,bins=edges,density=True,alpha=.4,color=color,edgecolor='#444444',label=f'{labels[group]} ({len(scores)})')
        for lo,hi,height in zip(edges[:-1],edges[1:],heights):
            histogram_rows.append({'drug_class':group,'bin_left':lo,'bin_right':hi,'density':height})
        if len(scores)>1 and np.std(scores)>0:
            x=np.linspace(0,1,400);density=gaussian_kde(scores)(x)
            ax.plot(x,density,color=color,lw=1)
            density_rows.extend({'drug_class':group,'score':xx,'density':yy} for xx,yy in zip(x,density))
    ax.set(xlim=(0,1),xlabel='SafetyNet toxicity score',ylabel='Density',title='Supplementary Figure 3b: calculated distributions')
    ax.legend(frameon=False,fontsize=8);fig.tight_layout()
    fig.savefig(ANALYSIS/'panel_b_toxicity_distribution.svg',bbox_inches='tight')
    fig.savefig(ANALYSIS/'panel_b_toxicity_distribution.png',bbox_inches='tight',dpi=180)
    plt.close(fig)
    pd.DataFrame(histogram_rows).to_csv(ANALYSIS/'histogram_bins.csv',index=False)
    pd.DataFrame(density_rows,columns=['drug_class','score','density']).to_csv(ANALYSIS/'density_values.csv',index=False)
    status.update(status='computed_from_supplied_compound_scores',plots_generated=True,panels_recomputed=['b'])
    (ANALYSIS/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
