"""Generate Figure 4 panel-by-panel, independent of the working directory."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main() -> dict:
    results={}
    for letter in 'abcdefghi':
        path=HERE/f'panel_{letter}'/f'figure_4{letter}.py'
        spec=importlib.util.spec_from_file_location(f'figure_4_panel_{letter}',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        results[f'panel_{letter}']=module.main()
    return results


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
