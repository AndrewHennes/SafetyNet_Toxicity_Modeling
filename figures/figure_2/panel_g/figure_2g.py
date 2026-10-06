import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
STATUS={'panel': 'g', 'status': 'blocked_missing_raw_inputs', 'details': 'Inputs: compound-level animal dose observations, species/routes, units, transformations, and SafetyNet predictions.', 'reference_artwork': str(HERE.parent / 'figure.svg'), 'synthetic_or_summary_replay': False}


def main() -> dict:
    (HERE/'status.json').write_text(json.dumps(STATUS,indent=2)+'\n')
    return dict(STATUS)


if __name__=='__main__':
    main()
