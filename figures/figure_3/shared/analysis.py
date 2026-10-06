"""Analyze Figure 3 compound-level assay records and prediction scores.

Cohort selection rules and sensitivity analyses are recorded alongside the
results. Call prepare_cohort() or run this file from any working directory."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from rdkit.Chem import FilterCatalog
from scipy.stats import gaussian_kde
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, roc_auc_score)

REPOSITORY = Path("/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private")
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE
INPUTS = {
    "ecbd": ("datasets/external_validation/ecbd/ecbd_assay_labels_predictions_and_filters.csv", "formatted_ecbd_antibiotic_toxicity_data_combined_with_tox_and_antibiotic_scores_and_filters.csv"),
    "coadd_tox": ("datasets/external_validation/coadd/coadd_hek293_labels_and_predictions.tsv", "co_add_hek_with_standardized_smiles_with_minimol_fingerprints_activity_scores_and_tox_scores.csv"),
    "drh": ("datasets/external_validation/drug_repurposing_hub/drug_repurposing_hub_ec_hepg2_labels_and_predictions.tsv", "hepg2_ec_tanimoto_matches_with_standardized_smiles_with_minimol_fingerprints_activity_scores_and_tox_scores.csv"),
    "coadd_activity": ("datasets/antibiotic_activity_datasets/coadd.csv", None),
}
CATALOG_NAMES = {"Brenk": "BRENK", "Glaxo": "CHEMBL_Glaxo", "Dundee": "CHEMBL_Dundee", "BMS": "CHEMBL_BMS",
    "SureCHEMBL": "CHEMBL_SureChEMBL", "MLSMR": "CHEMBL_MLSMR", "Inpharmatica": "CHEMBL_Inpharmatica", "LINT": "CHEMBL_LINT"}
COLORS = {0: "#eca4c8", 1: "#8b6fad"}
SPECIES = {"e_coli": "E. coli", "k_pneumoniae": "K. pneumoniae"}
METRIC_NAMES = {"mcc": "MCC", "f1": "F1 score", "balanced_accuracy": "Balanced accuracy", "precision": "Precision"}
MANIFEST, AUDIT, EXCLUDED = [], [], []


def repair_cxsmiles_spacing(text):
    """Preserve CXSMILES annotations while repairing their missing separator."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    position = text.find("|")
    if position > 0 and not text[position - 1].isspace():
        text = text[:position] + " " + text[position:]
    return text


def input_path(key):
    canonical, _legacy = INPUTS[key]
    path = REPOSITORY / canonical
    if not path.is_file():
        raise FileNotFoundError(f"Missing source dataset: {REPOSITORY / canonical}")
    return path


def read_input(key, columns, separator=","):
    path = input_path(key)
    data = pd.read_csv(path, usecols=columns, sep=separator)
    data.insert(0, "source_row", np.arange(len(data)) + 2)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    MANIFEST.append({"key": key, "path": str(path), "sha256": digest, "rows": len(data), "delimiter": separator})
    return data


@lru_cache(maxsize=32768)
def canonical_structure(smiles, isomeric=True):
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(repair_cxsmiles_spacing(smiles))
    return Chem.MolToSmiles(molecule, isomericSmiles=isomeric) if molecule is not None and molecule.GetNumAtoms() > 0 else None


def standard_row(source, source_id, row_numbers, smiles, label, score, **extra):
    structure = canonical_structure(smiles)
    return {"source": source, "source_id": str(source_id), "source_rows": str(row_numbers),
            "smiles": smiles, "canonical_isomeric_smiles": structure, "toxicity_label": label,
            "safetynet_score": score, "organism": "e_coli", **extra}


def summarize(name, frame, source, rule):
    if frame.empty:
        AUDIT.append({"variant": name, "source": source, "rule": rule, "n": 0, "nontoxic": 0, "toxic": 0})
        return
    for organism, group in frame.groupby("organism"):
        y = pd.to_numeric(group.toxicity_label, errors="coerce")
        score = pd.to_numeric(group.safetynet_score, errors="coerce")
        valid = y.isin([0, 1]) & score.between(0, 1) & group.canonical_isomeric_smiles.notna()
        y, score = y[valid].astype(int), score[valid]
        AUDIT.append({"variant": name, "source": source, "organism": organism, "rule": rule,
                      "n": len(y), "nontoxic": int((y == 0).sum()), "toxic": int((y == 1).sum()),
                      "auroc": roc_auc_score(y, score) if y.nunique() == 2 else np.nan,
                      "average_precision": average_precision_score(y, score) if y.nunique() == 2 else np.nan})


def build_ecbd():
    hit_columns = {"e_coli": "Esherichia_coli", "k_pneumoniae": "Klebsiella_pneumoniae"}
    flags = [name.lower() for name in CATALOG_NAMES if name != "Brenk"]
    columns = ["SMILES", "HepG2_Binarized_Toxicity", "HepG2_Percent_Growth_Inhibition", "Max_Tox_Score", *flags]
    for prefix in hit_columns.values():
        columns += [prefix + "_Binarized_Hit", prefix + "_Percent_Growth_Inhibition"]
    data = read_input("ecbd", columns)
    frames = []
    for organism, prefix in hit_columns.items():
        for policy in ["supplied_binary_hit", "growth_inhibition_gt_50"]:
            selected = data[data[prefix + "_Binarized_Hit"].eq(1) if policy == "supplied_binary_hit" else data[prefix + "_Percent_Growth_Inhibition"].gt(50)]
            records = []
            for row in selected.to_dict("records"):
                records.append(standard_row("ecbd", f"ecbd_row_{row['source_row']}", row["source_row"], row["SMILES"],
                    row["HepG2_Binarized_Toxicity"], row["Max_Tox_Score"], organism=organism,
                    antibacterial_inhibition=row[prefix + "_Percent_Growth_Inhibition"], toxicity_endpoint="HepG2 supplied binary label",
                    toxicity_rule="HepG2_Binarized_Toxicity", antibacterial_rule=policy,
                    **{f"supplied_{flag}": row[flag] for flag in flags}))
            frame = pd.DataFrame(records)
            summarize("ecbd_" + policy, frame, "ecbd", "Existing HepG2 binary labels; no novelty exclusion")
            if policy == "supplied_binary_hit":
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parse_value(text):
    match = re.fullmatch(r"\s*(<=|>=|<|>|=)?\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*", str(text))
    return (match.group(1) or "=", float(match.group(2))) if match else (None, np.nan)


def parse_dose(text):
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]*)?)\s*(ug/mL|uM)\s*", str(text))
    return (float(match.group(1)), match.group(2)) if match else (np.nan, None)


def cc50_label(value, unit, threshold, threshold_unit, smiles):
    """Classify only when the complete CC50 censoring interval lies on one side.

    Positive means CC50 <= threshold. Whole submitted-form molecular mass is
    used only for a cross-unit comparison; salts are not silently stripped.
    """
    operator, bound = parse_value(value)
    if operator is None or unit not in {"ug/mL", "uM"} or threshold_unit not in {"ug/mL", "uM"} or not np.isfinite(threshold) or threshold <= 0:
        return np.nan, "missing_or_unparseable_value_or_unit", np.nan
    molecular_weight = np.nan
    if unit != threshold_unit:
        molecule = Chem.MolFromSmiles(repair_cxsmiles_spacing(smiles)) if isinstance(smiles, str) else None
        if molecule is None or molecule.GetNumAtoms() == 0:
            return np.nan, "invalid_structure_for_unit_conversion", np.nan
        molecular_weight = Descriptors.MolWt(molecule)
        if not np.isfinite(molecular_weight) or molecular_weight <= 0:
            return np.nan, "invalid_molecular_weight_for_unit_conversion", molecular_weight
        bound = bound * 1000 / molecular_weight if unit == "ug/mL" else bound * molecular_weight / 1000
    if operator == "=":
        return int(bound <= threshold), "exact_cc50", molecular_weight
    if operator in {"<", "<="} and bound <= threshold:
        return 1, "upper_bound_proves_toxic", molecular_weight
    if operator == ">" and bound >= threshold:
        return 0, "lower_bound_proves_nontoxic", molecular_weight
    if operator == ">=" and bound > threshold:
        return 0, "lower_bound_proves_nontoxic", molecular_weight
    return np.nan, "censoring_interval_straddles_threshold", molecular_weight


def build_coadd():
    tox_columns = ["COADD_ID", "SMILES", "Standardized_SMILES", "DRVAL_MEDIAN", "DRVAL_UNIT", "DMAX_AVE", "Max_Tox_Score"]
    toxicity = read_input("coadd_tox", tox_columns, "\t")
    if toxicity.COADD_ID.duplicated().any():
        raise ValueError("CO-ADD toxicity compound IDs are not unique; review dose-response replication before joining.")
    activity_columns = ["COADD_Compounds_ID", "SMILES", "Standardized_SMILES", "EC_Growth_Inhibition", "EC_Binarized_at_80_Percent_Growth_Inhibition", "EC_Drug_Concentration", "EC_Strain"]
    activity = read_input("coadd_activity", activity_columns)
    activity = activity[activity.COADD_Compounds_ID.isin(toxicity.COADD_ID)]
    # The merged source repeats the same E. coli fact across other organism rows.
    # Deduplicate E. coli facts before counting replicates or hit disagreements.
    facts = activity.drop_duplicates(subset=activity_columns).copy()
    facts.to_csv(OUTPUT_DIR / "coadd_joined_antibacterial_facts.csv", index=False)
    groups = {key: frame for key, frame in facts.groupby("COADD_Compounds_ID")}
    candidates = []
    for row in toxicity.to_dict("records"):
        compound_id = row["COADD_ID"]
        if compound_id not in groups:
            EXCLUDED.append({"source": "coadd", "source_id": compound_id, "reason": "no_exact_antibacterial_compound_id_join"})
            continue
        facts_for_compound = groups[compound_id]
        labels = pd.to_numeric(facts_for_compound.EC_Binarized_at_80_Percent_Growth_Inhibition, errors="coerce").dropna()
        if not labels.eq(1).any():
            continue
        structures = facts_for_compound.Standardized_SMILES.dropna().map(canonical_structure).dropna().unique()
        toxicity_structure = canonical_structure(row["Standardized_SMILES"])
        identity_ok = len(structures) == 1 and structures[0] == toxicity_structure
        concentrations = facts_for_compound.EC_Drug_Concentration.dropna().unique()
        dose, dose_unit = parse_dose(concentrations[0]) if len(concentrations) == 1 else (np.nan, None)
        for policy in ["matched_antibacterial_dose", "fixed_10_um", "conventional_32_ug_ml_or_10_um"]:
            target, target_unit = (dose, dose_unit)
            if policy == "fixed_10_um":
                target, target_unit = 10, "uM"
            elif policy == "conventional_32_ug_ml_or_10_um":
                target, target_unit = (32, "ug/mL") if row["DRVAL_UNIT"] == "ug/mL" else (10, "uM")
            label, reason, molecular_weight = cc50_label(row["DRVAL_MEDIAN"], row["DRVAL_UNIT"], target, target_unit, row["SMILES"])
            record = standard_row("coadd", compound_id, row["source_row"], row["SMILES"], label, row["Max_Tox_Score"],
                antibacterial_source_rows=";".join(facts_for_compound.source_row.astype(str)),
                antibacterial_distinct_facts=len(facts_for_compound), antibacterial_hit_consensus=bool(labels.eq(1).all()),
                standardized_structure_join_agrees=bool(identity_ok), antibacterial_concentration=";".join(map(str, concentrations)),
                cc50_reported=row["DRVAL_MEDIAN"], cc50_unit=row["DRVAL_UNIT"], toxicity_rule=policy,
                toxicity_endpoint="HEK293 CC50", toxicity_target_concentration=target, toxicity_target_unit=target_unit,
                cc50_classification_reason=reason, whole_form_molecular_weight_for_conversion=molecular_weight,
                dmax_reported=row["DMAX_AVE"], antibacterial_rule="supplied EC wild-type 80% inhibition hit")
            candidates.append(record)
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_csv(OUTPUT_DIR / "coadd_candidate_classification_audit.csv", index=False)
    valid = candidate_frame.toxicity_label.isin([0, 1]) & candidate_frame.standardized_structure_join_agrees
    for policy in candidate_frame.toxicity_rule.unique():
        for consensus in [True, False]:
            mask = valid & candidate_frame.toxicity_rule.eq(policy)
            if consensus:
                mask &= candidate_frame.antibacterial_hit_consensus
            summarize(f"coadd_{policy}_{'consensus_hits' if consensus else 'any_hit'}", candidate_frame[mask], "coadd", "CC50 interval-aware; exact compound ID and standardized structure join")
    primary = candidate_frame[candidate_frame.toxicity_rule.eq("matched_antibacterial_dose")].copy()
    for row in primary.to_dict("records"):
        reasons = []
        if not row["standardized_structure_join_agrees"]:
            reasons.append("structure_disagrees_across_compound_id_join")
        if not row["antibacterial_hit_consensus"]:
            reasons.append("discordant_antibacterial_hit_labels")
        if row["toxicity_label"] not in (0, 1):
            reasons.append(row["cc50_classification_reason"])
        if reasons:
            EXCLUDED.append({"source": "coadd", "source_id": row["source_id"], "reason": ";".join(reasons)})
    return primary[primary.toxicity_label.isin([0, 1]) & primary.antibacterial_hit_consensus & primary.standardized_structure_join_agrees].copy()


def collapse_consistent_records(frame, name, allow_any_toxic=False, audit_exclusions=False):
    """One chemical per organism; do not average conflicting saved predictions."""
    kept = []
    for (_, structure), group in frame.groupby(["organism", "canonical_isomeric_smiles"], dropna=False):
        label = pd.to_numeric(group.toxicity_label, errors="coerce")
        score = pd.to_numeric(group.safetynet_score, errors="coerce")
        reasons = []
        if pd.isna(structure):
            reasons.append("invalid_molecular_structure")
        if not label.isin([0, 1]).all() or not score.between(0, 1).all():
            reasons.append("missing_invalid_label_or_score")
        if label.nunique() > 1 and not allow_any_toxic:
            reasons.append("discordant_observed_toxicity_labels")
        if score.max() - score.min() > 1e-6:
            reasons.append("different_saved_prediction_versions")
        if reasons:
            if audit_exclusions:
                EXCLUDED.append({"source": name, "source_id": ";".join(group.source_id.astype(str)), "reason": ";".join(reasons)})
            continue
        row = group.iloc[0].to_dict()
        row["toxicity_label"] = int(label.max())
        row["contributing_source_ids"] = ";".join(group.source_id.astype(str).unique())
        row["contributing_sources"] = ";".join(group.source.astype(str).unique())
        row["contributing_source_rows"] = ";".join(f"{source}:{line}" for source, line in zip(group.source, group.source_rows))
        row["merged_record_count"] = len(group)
        kept.append(row)
    return pd.DataFrame(kept, columns=list(frame.columns) + [x for x in ["contributing_source_ids", "contributing_sources", "contributing_source_rows", "merged_record_count"] if x not in frame.columns])


def build_drh():
    columns = ["SMILES", "Standardized_SMILES", "HepG2_smiles", "EC_Broad ID", "EC__orig_index", "HepG2_ID", "HepG2_broad_id", "HepG2__orig_index", "Tanimoto", "EC_hit_inh", "EC_hit_kill", "HepG2_dose", "HepG2_screen_id", "HepG2_label_HepG2", "Max_Tox_Score"]
    data = read_input("drh", columns, "\t")
    data = data[data.EC_hit_inh.eq(1) | data.EC_hit_kill.eq(1)].copy()
    data["identity_isomeric"] = [canonical_structure(a) == canonical_structure(b) and canonical_structure(a) is not None for a, b in zip(data.SMILES, data.HepG2_smiles)]
    data["identity_connectivity"] = [canonical_structure(a, False) == canonical_structure(b, False) and canonical_structure(a, False) is not None for a, b in zip(data.SMILES, data.HepG2_smiles)]
    data.to_csv(OUTPUT_DIR / "drh_candidate_identity_audit.csv", index=False)
    results = {}
    for hit_kind in ["EC_hit_inh", "EC_hit_kill"]:
        for matching in ["identity_isomeric", "identity_connectivity", "reported_tanimoto_1", "all_supplied_matches"]:
            selected = data[data[hit_kind].eq(1)]
            if matching.startswith("identity_"):
                selected = selected[selected[matching]]
            elif matching == "reported_tanimoto_1":
                selected = selected[selected.Tanimoto.eq(1)]
            rows = []
            for row in selected.to_dict("records"):
                rows.append(standard_row("drug_repurposing_hub", f"{row['EC_Broad ID']}|{row['HepG2_ID']}", row["source_row"], row["SMILES"],
                    row["HepG2_label_HepG2"], row["Max_Tox_Score"], antibacterial_rule=hit_kind,
                    toxicity_rule="supplied HepG2 label; consensus across same-molecule paired records", toxicity_endpoint="HepG2 supplied binary label",
                    hepg2_broad_id=row["HepG2_broad_id"], hepg2_dose_reported=row["HepG2_dose"], hepg2_screen=row["HepG2_screen_id"],
                    reported_pair_tanimoto=row["Tanimoto"], pair_matching_rule=matching))
            frame = pd.DataFrame(rows)
            if frame.empty:
                continue
            for any_toxic in [False, True]:
                variant = f"drh_{hit_kind}_{matching}_{'any_toxic' if any_toxic else 'consensus_toxicity'}"
                collapsed = collapse_consistent_records(frame, "drug_repurposing_hub", allow_any_toxic=any_toxic,
                    audit_exclusions=hit_kind == "EC_hit_inh" and matching == "identity_isomeric" and not any_toxic)
                summarize(variant, collapsed, "drug_repurposing_hub", "Unique canonical isomeric EC structures; inspect identity policy")
                results[variant] = collapsed
    for row in data[data.EC_hit_inh.eq(1) & ~data.identity_isomeric].to_dict("records"):
        EXCLUDED.append({"source": "drug_repurposing_hub", "source_id": f"{row['EC_Broad ID']}|{row['HepG2_ID']}", "reason": "paired_assays_do_not_have_identical_isomeric_structures"})
    return results["drh_EC_hit_inh_identity_isomeric_consensus_toxicity"]


def calculate_alerts(cohort):
    catalogs = {name: FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, enum))) for name, enum in CATALOG_NAMES.items()}
    comparison = []
    for index, row in cohort.iterrows():
        molecule = Chem.MolFromSmiles(repair_cxsmiles_spacing(row.smiles))
        if molecule is None:
            raise ValueError(f"Invalid structure survived cohort validation: {row.source_id}")
        for name, catalog in catalogs.items():
            flag = int(catalog.HasMatch(molecule))
            cohort.loc[index, "alert_" + name.lower()] = flag
            supplied = row.get("supplied_" + name.lower())
            if pd.notna(supplied):
                comparison.append({"source_id": row.source_id, "organism": row.organism, "method": name,
                                   "supplied_flag": int(supplied), "recomputed_flag": flag, "agrees": int(supplied) == flag})
    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / "supplied_vs_recomputed_alerts.csv", index=False)
    return cohort, {name: {"rdkit_enum": CATALOG_NAMES[name], "entries": catalog.GetNumEntries()} for name, catalog in catalogs.items()}


def metrics_for(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mcc": matthews_corrcoef(y, predicted), "f1": f1_score(y, predicted, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, predicted), "precision": precision_score(y, predicted, zero_division=0)}


def analyze(cohort):
    metric_rows, threshold_rows, histogram_rows, kde_rows, summaries = [], [], [], [], []
    for organism, group in cohort.groupby("organism"):
        y, score = group.toxicity_label.astype(int).to_numpy(), group.safetynet_score.to_numpy(float)
        if len(np.unique(y)) != 2:
            raise ValueError(f"Both toxicity classes are needed for {organism}.")
        summaries.append({"organism": organism, "n": len(group), "nontoxic": int((y == 0).sum()), "toxic": int((y == 1).sum()),
            "auroc": roc_auc_score(y, score), "average_precision": average_precision_score(y, score),
            "median_nontoxic": float(np.median(score[y == 0])), "median_toxic": float(np.median(score[y == 1])),
            "scores_exactly_0_2": int((score == .2).sum())})
        for threshold in np.arange(1, 10) / 10:
            counts = metrics_for(y, score > threshold)
            safe = counts["tn"] / (counts["tn"] + counts["fp"])
            toxic = counts["tp"] / (counts["tp"] + counts["fn"])
            threshold_rows.append({"organism": organism, "threshold": threshold, **counts,
                "nontoxic_retained_fraction": safe, "toxic_removed_fraction": toxic, "g_mean": np.sqrt(safe * toxic),
                "h_mean": 2 * safe * toxic / (safe + toxic) if safe + toxic else 0.0})
        for name in ["SafetyNet", *CATALOG_NAMES]:
            predicted = score > .2 if name == "SafetyNet" else group["alert_" + name.lower()].astype(bool).to_numpy()
            metric_rows.append({"organism": organism, "method": name, **metrics_for(y, predicted)})
        for label in [0, 1]:
            values = score[y == label]
            counts, edges = np.histogram(values, bins=np.linspace(0, 1, 21))
            histogram_rows += [{"organism": organism, "toxicity_label": label, "bin_left": edges[i], "bin_right": edges[i + 1], "count": int(count), "density": count / len(values) / .05} for i, count in enumerate(counts)]
            x = np.linspace(-.05, 1.05, 400)
            if len(values) > 1 and np.std(values) > 0:
                density = gaussian_kde(values, bw_method="scott")(x)
                kde_rows += [{"organism": organism, "toxicity_label": label, "score": xx, "density": yy} for xx, yy in zip(x, density)]
    tables = {"classification_metrics": metric_rows, "threshold_metrics": threshold_rows, "histogram_counts": histogram_rows, "kde_values": kde_rows, "cohort_summary": summaries}
    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    return frames


def prepare_cohort():
    """Rebuild shared molecular rows from the repository datasets, without drawing panels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.clear(); AUDIT.clear(); EXCLUDED.clear()
    ecbd, coadd, drh = build_ecbd(), build_coadd(), build_drh()
    source_records = pd.concat([ecbd, coadd, drh], ignore_index=True)
    source_records.to_csv(OUTPUT_DIR / "eligible_source_records.csv", index=False)
    summarize("pooled_source_records_before_cross_source_deduplication", source_records, "combined", "One eligible source record before chemical deduplication")
    cohort = collapse_consistent_records(source_records, "combined", audit_exclusions=True)
    summarize("primary_unique_consistent_molecules", cohort, "combined", "Exact canonical isomeric identity; exclude conflicting labels or score versions")
    cohort, catalogs = calculate_alerts(cohort)
    cohort.to_csv(OUTPUT_DIR / "reconstructed_cohort.csv", index=False)
    for source, frame in cohort.groupby("source"):
        summarize("primary_source_contribution", frame, source, "After within- and between-source chemical deduplication")
    frames = analyze(cohort)
    frames["cohort_summary"].to_csv(OUTPUT_DIR / "cohort_summary.csv", index=False)
    pd.DataFrame(AUDIT).to_csv(OUTPUT_DIR / "cohort_variant_audit.csv", index=False)
    pd.DataFrame(EXCLUDED).to_csv(OUTPUT_DIR / "excluded_records.csv", index=False)
    manifest = {"inputs": MANIFEST, "rdkit_version": rdBase.rdkitVersion, "catalogs": catalogs,
        "primary_rules": {"ecbd": "supplied binarized activity and HepG2 labels", "coadd": "exact ID and supplied-standardized structure agreement; unanimous E. coli hit observations; CC50 interval classified at matched antibacterial dose", "drh": "growth-inhibition hits; exact isomeric molecular identity between paired assays; unanimous observed toxicity", "duplicate_policy": "one canonical isomeric molecule per organism; conflicts in labels or saved scores excluded", "score": "provided Max_Tox_Score, not newly inferred", "threshold": "predicted toxic if score strictly greater than cutoff", "kde": "SciPy gaussian_kde with Scott bandwidth; no boundary correction", "chemical_space": "shared chemical_space.py analysis supplied by repository figure infrastructure"},
        "limits": ["Cohort selection and prediction-version assumptions are recorded in primary_rules.", "CO-ADD CC50, ECBD HepG2 labels, and DRH HepG2 labels have different assay definitions and exposure conditions.", "Training-set similarity filtering requires a specified exclusion policy and matching reference cohort.", "DMAX_AVE is retained for audit, not interpreted as a toxicity label without a source data dictionary.", "Statistics are calculated from the selected compound-level records."],
        "documentation": ["https://www.co-add.org/sites/co-add.org/files/CO-ADD_ScreeningWorkflow_May2016.pdf", "https://db.co-add.org/organism", "https://rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html"]}
    (OUTPUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cohort


if __name__ == "__main__":
    cohort = prepare_cohort()
    print(analyze(cohort)["cohort_summary"].to_string(index=False))
