import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
STATUS={'panel': 'b', 'status': 'blocked_missing_raw_inputs', 'details': 'Inputs: fold membership, baseline predictions, training configurations, and paired evaluation results.', 'reference_artwork': str(HERE.parent / 'figure.svg'), 'synthetic_or_summary_replay': False}


def main() -> dict:
    (HERE/'status.json').write_text(json.dumps(STATUS,indent=2)+'\n')
    return dict(STATUS)


if __name__=='__main__':
    main()
