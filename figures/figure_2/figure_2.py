"""Generate this figure one panel at a time from any working directory."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PANELS=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def main() -> dict:
    results={}
    for letter in PANELS:
        script=HERE/f'panel_{letter}'/f'figure_2{letter}.py'
        spec=importlib.util.spec_from_file_location(f'{HERE.name}_panel_{letter}',script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f'Generating {HERE.name} panel {letter}.',flush=True)
        results[letter]=module.main()
    (HERE/'shared'/'panel_run_status.json').write_text(json.dumps(results,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('figure2_summary',HERE/'shared/workflow.py')
    workflow=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    workflow.write_figure_validation(results)
    return results


if __name__=='__main__':
    main()
