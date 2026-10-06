"""Generate only Figure 3a with the shared chemical-space calculation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    specification = spec_from_file_location("safetynet_chemical_space", HERE.parents[1] / "chemical_space.py")
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.export_panel("figure_3", HERE)


if __name__ == "__main__":
    main()
