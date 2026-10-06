"""Compute Figure 4b; scientific preparation is shared across panels."""
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

REFERENCE_INTERSECTIONS = {'1111':89,'1000':74,'1110':43,'0110':37,'0100':21,'1100':14,
                           '0010':9,'0111':8,'0001':5,'1101':4,'1010':3,'0101':2,'1011':2,'0011':1}
REFERENCE_TOTALS = {'AB':229,'EC':218,'KP':192,'PA':111}

def intersection_counts(frame: pd.DataFrame) -> pd.Series:
    patterns = hit_frame(frame).astype(str).agg(''.join,axis=1)
    counts = patterns.value_counts().drop('0000',errors='ignore')
    return counts.sort_values(ascending=False,kind='stable')

def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ANALYSIS/f'{stem}.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(ANALYSIS/f'{stem}.png',bbox_inches='tight',facecolor='white',dpi=180)
    plt.close(fig)

def plot_intersections(frame: pd.DataFrame, stem: str, subtitle: str) -> None:
    counts = intersection_counts(frame)
    flags = hit_frame(frame)
    fig=plt.figure(figsize=(9,4.3))
    grid=fig.add_gridspec(2,2,width_ratios=[4,1],height_ratios=[3,1],wspace=.24,hspace=.06)
    bars=fig.add_subplot(grid[0,0]); matrix=fig.add_subplot(grid[1,0],sharex=bars)
    totals=fig.add_subplot(grid[0,1]); x=np.arange(len(counts))
    bars.bar(x,counts.to_numpy(),color='#b2b2b2',width=.8)
    for i,value in enumerate(counts):bars.text(i,value+1.5,str(value),ha='center',fontsize=9)
    bars.set(ylabel='Number of antibacterial hits',ylim=(0,max(counts)*1.17))
    bars.tick_params(axis='x',bottom=False,labelbottom=False)
    matrix.set(ylim=(3.55,-.55),xticks=[],yticks=range(4),yticklabels=[DISPLAY[s] for s in SPECIES])
    for row in range(4):matrix.axhline(row,color='#dddddd',lw=.5,ls=':')
    for col,pattern in enumerate(counts.index):
        active=[row for row,bit in enumerate(pattern) if bit=='1']
        if len(active)>1:matrix.plot([col,col],[min(active),max(active)],color='#777777',lw=.8)
        for row in active:matrix.scatter(col,row,s=26,facecolor=COLORS[row],edgecolor='#222222',lw=.5,zorder=3)
    heights=flags.sum()
    totals.bar(np.arange(4),heights,color=COLORS)
    totals.set_xticks(np.arange(4),SPECIES)
    totals.set_ylim(0,max(heights)*1.2)
    totals.set_ylabel('Number of hits')
    for i,value in enumerate(heights):totals.text(i,value+3,f'{value}\n({value/len(frame):.1%})',ha='center',fontsize=8)
    fig.suptitle(f'Figure 4b: inhibition >80% at 50 µM\n{subtitle}',fontsize=11)
    fig.subplots_adjust(left=.18,right=.98,top=.79,bottom=.13)
    fig.text(.18,.018,'Complete assay records; missing assays excluded.',fontsize=7,color='#555555')
    save(fig,stem)

def main() -> dict:
    complete,cohort,report=shared.load_cohorts()
    scenarios=shared.cohort_scenarios(complete,cohort)
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.linewidth':.6,'svg.fonttype':'none','svg.hashsalt':'safetynet_figure4_analysis'})
    audit=[];comparisons=[]
    for name,frame in scenarios.items():
        flags=hit_frame(frame);counts=intersection_counts(frame)
        audit.append({'scenario':name,'compound_or_record_count':len(frame),'any_hit':int(flags.any(axis=1).sum()),
                      **{f'{species}_hits':int(flags[species].sum()) for species in SPECIES}})
        for pattern in sorted(set(counts.index)|set(REFERENCE_INTERSECTIONS)):
            observed=int(counts.get(pattern,0));reference=REFERENCE_INTERSECTIONS.get(pattern,0)
            comparisons.append({'scenario':name,'pattern_ab_ec_kp_pa':pattern,'calculated':observed,
                                'artwork_reference_only':reference,'difference':observed-reference})
    pd.DataFrame(audit).to_csv(ANALYSIS/'cohort_sensitivity.csv',index=False)
    pd.DataFrame(comparisons).to_csv(ANALYSIS/'intersection_comparison.csv',index=False)
    counts=intersection_counts(cohort)
    counts.rename_axis('pattern_ab_ec_kp_pa').rename('count').to_csv(ANALYSIS/'intersection_counts.csv')
    totals=hit_frame(cohort).sum()
    totals.rename_axis('species').rename('hits').to_csv(ANALYSIS/'species_totals.csv')
    plot_intersections(complete,'iteration_1_all_complete_records',f'Initial: {len(complete):,} complete source rows')
    plot_intersections(cohort,'panel_b_compound_intersections',f'Refined: {len(cohort):,} complete unique raw-SMILES compounds')
    status={'computed_species_totals':totals.astype(int).to_dict(),'artwork_totals_reference_only':REFERENCE_TOTALS,
            'matched_artwork_intersections':int(sum(int(counts.get(k,0))==v for k,v in REFERENCE_INTERSECTIONS.items()))}
    status.update(panel='b',status='recomputed_from_assay_measurements',
                  shared_validation=str(shared.ANALYSIS/'validation.json'),
                  cohort=str(shared.ANALYSIS/'analysis_cohort.csv'),source_sha256=report['source_sha256'])
    (ANALYSIS/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
