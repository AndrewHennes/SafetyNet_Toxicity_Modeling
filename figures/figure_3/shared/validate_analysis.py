"""Check censoring examples and molecular-cohort/output invariants without a CLI."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    spec = spec_from_file_location("figure_3_analysis", HERE / "analysis.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = [("10", "uM", 10, "uM", 1), ("<10", "uM", 10, "uM", 1),
             ("<=10", "uM", 10, "uM", 1), (">10", "uM", 10, "uM", 0),
             (">=10", "uM", 10, "uM", None), ("<20", "uM", 10, "uM", None),
             ("<=20", "uM", 10, "uM", None), (">5", "uM", 10, "uM", None),
             (">=5", "uM", 10, "uM", None), ("<=5", "uM", 10, "uM", 1),
             (">=20", "uM", 10, "uM", 0), ("20", "uM", 10, "uM", 0),
             ("5", "uM", 10, "uM", 1), ("1", "ug/mL", 10, "uM", 0),
             ("0.25", "ug/mL", 10, "uM", 1), ("10", "uM", 1, "ug/mL", 1),
             ("32", "uM", 1, "ug/mL", 0), ("unknown", "uM", 10, "uM", None),
             ("10", "unknown", 10, "uM", None), ("10", "uM", 0, "uM", None)]
    for text, unit, threshold, threshold_unit, expected in cases:
        actual, _, _ = module.cc50_label(text, unit, threshold, threshold_unit, "CCO")
        assert np.isnan(actual) if expected is None else actual == expected, (text, unit, threshold, threshold_unit, actual, expected)
    assert module.canonical_structure("") is None
    assert module.cc50_label("1", "ug/mL", 10, "uM", "")[1] == "invalid_structure_for_unit_conversion"
    assert module.repair_cxsmiles_spacing("C[C@H](F)O|&1:1|") == "C[C@H](F)O |&1:1|"
    assert module.repair_cxsmiles_spacing("C[C@H](F)O |&1:1|") == "C[C@H](F)O |&1:1|"
    assert module.canonical_structure("C[C@H](F)O|&1:1|") is not None
    directory = HERE
    cohort = pd.read_csv(directory / "reconstructed_cohort.csv")
    thresholds = pd.concat([pd.read_csv(HERE.parent / f"panel_{letter}" / "threshold_metrics.csv")
                            for letter in "de"], ignore_index=True)
    histograms = pd.concat([pd.read_csv(HERE.parent / f"panel_{letter}" / "histogram_counts.csv")
                            for letter in "bc"], ignore_index=True)
    assert not cohort.duplicated(["organism", "canonical_isomeric_smiles"]).any()
    assert cohort.toxicity_label.isin([0, 1]).all()
    assert cohort.safetynet_score.between(0, 1).all()
    assert cohort.contributing_source_ids.notna().all()
    for organism, frame in cohort.groupby("organism"):
        safe, toxic = int(frame.toxicity_label.eq(0).sum()), int(frame.toxicity_label.eq(1).sum())
        table = thresholds[thresholds.organism.eq(organism)]
        assert len(table) == 9
        assert ((table.tn + table.fp) == safe).all()
        assert ((table.tp + table.fn) == toxic).all()
        assert table.tn.is_monotonic_increasing and table.tp.is_monotonic_decreasing
        for label, count in [(0, safe), (1, toxic)]:
            bins = histograms[histograms.organism.eq(organism) & histograms.toxicity_label.eq(label)]
            assert bins['count'].sum() == count and len(bins) == 20
            assert np.isclose(np.sum(bins.density * (bins.bin_right - bins.bin_left)), 1)
    expected_frames = module.analyze(cohort)
    table_panels = {"histogram_counts": "bc", "kde_values": "bc",
                    "threshold_metrics": "de", "classification_metrics": "fg"}
    for name, letters in table_panels.items():
        actual = pd.concat([pd.read_csv(HERE.parent / f"panel_{letter}" / (name + ".csv"))
                            for letter in letters], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected_frames[name], check_dtype=False,
                                      rtol=1e-12, atol=1e-12)
    for letter, organism in zip("bcdefg", ["e_coli", "k_pneumoniae"] * 3):
        panel = HERE.parent / f"panel_{letter}"
        summary = pd.read_csv(panel / "cohort_summary.csv")
        assert summary.organism.tolist() == [organism]
        assert int(summary.n.iloc[0]) == int(cohort.organism.eq(organism).sum())
        assert (panel / f"figure_3{letter}.svg").is_file() and (panel / f"figure_3{letter}.png").is_file()
    alert_comparison = pd.read_csv(directory / "supplied_vs_recomputed_alerts.csv")
    assert alert_comparison.agrees.all()
    sensitivity = pd.read_csv(directory / "ecbd_version_label_novelty_sensitivity.csv")
    assert (sensitivity.n == sensitivity.nontoxic + sensitivity.toxic).all()
    similarities = pd.read_csv(directory / "ecbd_version_hit_candidates_with_similarity.csv")
    for column in ["computed_maxsim_100k", "computed_maxsim_39k"]:
        assert similarities[column].dropna().between(0, 1).all()
    report = {"cc50_cases_passed": len(cases), "empty_structure_cases_passed": 2,
              "cxsmiles_spacing_cases_passed": 3,
              "cohort_rows_checked": len(cohort), "panel_numeric_tables_match_molecular_analysis": True,
              "independent_panel_outputs_checked": 6, "cohort_unique_molecule_and_count_invariants": "passed",
              "supplied_alert_flags_checked": len(alert_comparison), "all_supplied_alert_flags_agree": True,
              "ecbd_sensitivity_rows_checked": len(sensitivity), "similarity_values_bounded": True}
    (directory / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
