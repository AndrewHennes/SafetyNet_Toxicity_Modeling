"""Generate Figure 3 panels and shared audits without command-line arguments."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(path, name):
    specification = spec_from_file_location(name, path.resolve())
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    analysis = load_module(HERE / "shared" / "analysis.py", "figure_3_analysis")
    cohort = analysis.prepare_cohort()
    load_module(HERE / "panel_a" / "figure_3a.py", "figure_3_panel_a").main()
    for letter in "bcdefg":
        panel = load_module(HERE / f"panel_{letter}" / f"figure_3{letter}.py", f"figure_3_panel_{letter}")
        panel.main(cohort=cohort)
    load_module(HERE / "shared" / "audit_ecbd_versions.py", "figure_3_ecbd_audit").main()
    load_module(HERE / "shared" / "validate_analysis.py", "figure_3_validation").main()


if __name__ == "__main__":
    main()
