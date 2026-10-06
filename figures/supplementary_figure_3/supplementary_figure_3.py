"""Run the two Supplementary Figure 3 panel analyses from any directory."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main() -> dict:
    results={}
    for letter in 'ab':
        spec=importlib.util.spec_from_file_location(f'supplementary_figure_3_panel_{letter}',HERE/f'panel_{letter}'/f'supplementary_figure_3{letter}.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        results[f'panel_{letter}']=module.main()
    return results


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
