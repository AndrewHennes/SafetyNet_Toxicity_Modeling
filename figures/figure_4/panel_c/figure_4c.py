"""Compute Figure 4c; scientific preparation is shared across panels."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parents[1] / '.matplotlib_cache'))
spec = importlib.util.spec_from_file_location('figure_4_shared_preparation', HERE.parent / 'shared' / 'prepare.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

SPECIES=shared.SPECIES
DISPLAY=shared.DISPLAY
ORDER=shared.ORDER
COLORS=shared.COLORS
hit_frame=shared.hit_frame

MEASUREMENTS=shared.MEASUREMENTS
REFERENCE_CORRELATION = np.array([[1,.59,.67,.65],[.59,1,.66,.67],[.67,.66,1,.85],[.65,.67,.85,1]])

def correlation(frame: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    values = hit_frame(frame) if binary else frame[MEASUREMENTS].rename(columns=dict(zip(MEASUREMENTS,SPECIES)))
    return values[ORDER].corr(method='pearson')

def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ANALYSIS/f'{stem}.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(ANALYSIS/f'{stem}.png',bbox_inches='tight',facecolor='white',dpi=180)
    plt.close(fig)

def plot_correlation(values: pd.DataFrame, stem: str, definition: str) -> None:
    array=values.to_numpy()
    masked=np.ma.array(array,mask=np.triu(np.ones((4,4),dtype=bool),1))
    cmap=LinearSegmentedColormap.from_list('source_palette',['#72cbe8','#ffffff','#a68abb'])
    cmap.set_bad('white')
    fig,ax=plt.subplots(figsize=(4.7,4.3))
    im=ax.imshow(masked,cmap=cmap,vmin=0,vmax=1)
    labels=[DISPLAY[s] for s in ORDER]
    ax.set_xticks(range(4),labels,rotation=50,ha='right',fontstyle='italic',fontsize=9)
    ax.set_yticks(range(4),labels,fontstyle='italic',fontsize=9)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='#333333',lw=.5);ax.tick_params(which='minor',bottom=False,left=False)
    for row in range(4):
        for col in range(row+1):ax.text(col,row,f'{array[row,col]:.2f}',ha='center',va='center',color='white' if row==col else 'black')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.07,label='Pearson r')
    ax.set_title('Figure 4c: '+definition,fontsize=11,pad=12)
    fig.subplots_adjust(left=.27,right=.90,bottom=.27,top=.84)
    fig.text(.05,.015,'Pearson correlation across complete, unique-compound assay records.',fontsize=6.5,color='#555555')
    save(fig,stem)

def main() -> dict:
    complete,cohort,report=shared.load_cohorts()
    scenarios=shared.cohort_scenarios(complete,cohort)
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.linewidth':.6,'svg.fonttype':'none','svg.hashsalt':'safetynet_figure4_analysis'})
    comparisons=[]
    reference_pairs=np.tril_indices(4,-1)
    for name,frame in scenarios.items():
        for kind,binary in [('continuous_inhibition',False),('binary_hits',True)]:
            matrix=correlation(frame,binary)
            for row,col in zip(*reference_pairs):
                value=float(matrix.iloc[row,col]);reference=float(REFERENCE_CORRELATION[row,col])
                comparisons.append({'scenario':name,'definition':kind,'species_1':ORDER[row],'species_2':ORDER[col],
                                    'calculated':value,'artwork_reference_only':reference,'difference':value-reference})
    pd.DataFrame(comparisons).to_csv(ANALYSIS/'correlation_comparison.csv',index=False)
    continuous=correlation(cohort);binary=correlation(cohort,True)
    continuous.to_csv(ANALYSIS/'continuous_inhibition_pearson.csv')
    binary.to_csv(ANALYSIS/'binary_hit_pearson.csv')
    clipped=cohort[MEASUREMENTS].clip(0,100).rename(columns=dict(zip(MEASUREMENTS,SPECIES)))[ORDER].corr()
    clipped.to_csv(ANALYSIS/'diagnostic_correlation_clipped_0_100.csv')
    plot_correlation(continuous,'panel_c_continuous_inhibition','continuous inhibition')
    plot_correlation(binary,'panel_c_binary_hits_diagnostic','binary hit labels (diagnostic)')
    status={'primary_definition':'Pearson correlation of continuous growth inhibition',
            'diagnostic_definition':'Pearson correlation of binary >80% inhibition hit labels'}
    status.update(panel='c',status='recomputed_from_assay_measurements',
                  shared_validation=str(shared.ANALYSIS/'validation.json'),
                  cohort=str(shared.ANALYSIS/'analysis_cohort.csv'),source_sha256=report['source_sha256'])
    (ANALYSIS/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
