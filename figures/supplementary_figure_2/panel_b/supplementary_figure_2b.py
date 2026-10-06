import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
STATUS={'panel': 'b', 'status': 'blocked_missing_raw_inputs', 'details': 'Inputs: animal toxicity observations and matched predictions, compound identities, species/routes, endpoints, units, transformations, and inclusion policy.', 'reference_artwork': str(HERE.parent / 'figure.svg'), 'synthetic_or_summary_replay': False}


def main() -> dict:
    (HERE/'status.json').write_text(json.dumps(STATUS,indent=2)+'\n')
    return dict(STATUS)


if __name__=='__main__':
    main()
