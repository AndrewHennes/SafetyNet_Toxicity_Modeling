import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
STATUS={'panel': 'a', 'status': 'schematic_reference_only', 'details': 'SafetyNet architecture schematic. The architecture diagram is available in figure.svg.', 'reference_artwork': str(HERE.parent / 'figure.svg'), 'synthetic_or_summary_replay': False}


def main() -> dict:
    (HERE/'status.json').write_text(json.dumps(STATUS,indent=2)+'\n')
    return dict(STATUS)


if __name__=='__main__':
    main()
