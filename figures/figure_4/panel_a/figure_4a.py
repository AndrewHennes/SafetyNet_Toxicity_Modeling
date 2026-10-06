"""Generate Figure 4a using the shared molecular chemical-space analysis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main() -> dict:
    spec=importlib.util.spec_from_file_location('shared_chemical_space',HERE.parents[1]/'chemical_space.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_panel('figure_4',output=HERE)


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
