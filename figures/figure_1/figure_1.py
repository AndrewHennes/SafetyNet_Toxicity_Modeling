"""Generate each Figure 1 panel from any working directory."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    for panel in "abcdef":
        subprocess.run([sys.executable, str(HERE / f"panel_{panel}" / f"figure_1{panel}.py")], check=True)
    sys.path.insert(0, str(HERE / "shared"))
    from analysis import viability_statistics
    import numpy as np
    viability, counts, correlation = viability_statistics()
    metadata = json.loads((HERE / "panel_c/validation.json").read_text())["descriptor_provenance"]
    report = {
        "screen_rows": len(viability), "source_screen_rows": 100390,
        "toxicity_rule": "Each raw relative viability < 0.8; no deduplication",
        "toxic_union": sum(counts.values()), "source_toxic_union": 20177,
        "all_three": counts[7], "source_all_three": 1992,
        "correlations_round_as_in_artwork": bool(np.allclose(np.round(correlation.to_numpy(), 2),
            [[1, .44, .33], [.44, 1, .29], [.33, .29, 1]])),
        "panels_recomputed": ["b (independent projection)", "c (internal and Tox21 only)", "d", "e", "f"],
        "missing_panels": ["a is schematic artwork", "b projection depends on the recorded molecular and optimization settings",
                           "c Uses supplied molecular cohorts and configured density settings"],
        "descriptor_provenance": metadata}
    (HERE / "shared/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "descriptor_provenance"}, indent=2))


if __name__ == "__main__":
    main()
