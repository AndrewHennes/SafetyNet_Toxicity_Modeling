"""Generate figure panels from molecular records and prediction scores.

Each figure runner calculates its supported analyses and records any missing
input requirements in the corresponding panel status files."""
from pathlib import Path
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
FIGURES=['figure_1','figure_2','figure_3','figure_4','supplementary_figure_1',
         'supplementary_figure_2','supplementary_figure_3']


def main():
    environment=os.environ.copy()
    environment.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib_cache'))
    environment.setdefault('NUMBA_CACHE_DIR',str(HERE/'.numba_cache'))
    for name in FIGURES:
        print(f'Analyzing source data for {name}',flush=True)
        subprocess.run([sys.executable,str(HERE/name/f'{name}.py')],cwd=HERE,
                       env=environment,check=True)
    print('All supported analyses completed. Panel status files identify missing source data.')


if __name__=='__main__':
    main()
