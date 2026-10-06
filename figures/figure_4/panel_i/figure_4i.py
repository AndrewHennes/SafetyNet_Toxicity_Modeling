"""Record the input requirements and analysis status of Figure 4i."""
from __future__ import annotations
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main() -> dict:
    output=HERE
    output.mkdir(parents=True,exist_ok=True)
    status={'panel': 'i', 'status': 'blocked_missing_inputs', 'reason': 'K. pneumoniae inputs: compound-level antibacterial hits, measured toxicity labels, and SafetyNet scores.', 'plots_generated': False, 'artwork_values_used_as_analysis_inputs': False}
    status['reference_figure']=str(HERE.parent/'figure.svg')
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
