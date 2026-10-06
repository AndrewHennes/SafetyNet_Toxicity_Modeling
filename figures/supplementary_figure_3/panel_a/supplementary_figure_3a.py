"""Compute Supplementary Figure 3a only when actual DrugBank scores are supplied."""
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
    status={**preparation_status,'panel':'a','shared_validation':str(shared.ANALYSIS/'validation.json')}
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
    rows=[]
    for group in shared.GROUPS:
        selected=data.loc[data.drug_class.eq(group)]
        rows.append({'drug_class':group,'compounds':len(selected),
                     **{score:int(selected[score].ge(shared.THRESHOLD).sum()) for score in shared.SCORES},
                     'any_cell_line':int(selected.max_toxicity_score.ge(shared.THRESHOLD).sum())})
    table=pd.DataFrame(rows);table['any_cell_fraction']=table.any_cell_line/table.compounds
    table.to_csv(ANALYSIS/'calculated_toxicity_counts.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.2,2.6));ax.axis('off')
    cells=[[labels[row.drug_class],int(row.compounds),int(row.hepg2_score),int(row.hskmc_score),int(row.imr90_score),
            f'{int(row.any_cell_line)} ({row.any_cell_fraction:.0%})'] for row in table.itertuples()]
    artist=ax.table(cellText=cells,colLabels=['Cohort','N','HepG2','HSkMC','IMR-90','Any'],loc='center',
                    colWidths=[.41,.07,.10,.11,.11,.16])
    artist.auto_set_font_size(False);artist.set_fontsize(8);artist.scale(1,2)
    ax.set_title('Supplementary Figure 3a: calculated toxicity counts',fontsize=11)
    fig.tight_layout()
    fig.savefig(ANALYSIS/'panel_a_toxicity_counts.svg',bbox_inches='tight')
    fig.savefig(ANALYSIS/'panel_a_toxicity_counts.png',bbox_inches='tight',dpi=180)
    plt.close(fig)
    status.update(status='computed_from_supplied_compound_scores',plots_generated=True,panels_recomputed=['a'])
    (ANALYSIS/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
