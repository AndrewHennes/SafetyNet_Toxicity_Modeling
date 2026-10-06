"""Audit all supplied ECBD releases and plausible novelty rules from source rows.

Morgan parameters are explicit assumptions; provided similarities are separate.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score

REPOSITORY = Path("/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private")
OUTPUT_DIR = Path(__file__).resolve().parent
SOURCES = {
    "assay_101097": ("datasets/external_validation/ecbd/ecbd_assay_labels_predictions_and_filters.csv", "formatted_ecbd_antibiotic_toxicity_data_combined_with_tox_and_antibiotic_scores_and_filters.csv"),
    "merged_106373": ("datasets/external_validation/ecbd/ecbd_merged_panel_predictions.csv", "ecbd_all_panels_merged_by_eos_with_minimol_representation_with_activity_and_tox_just_mean_of_ensemble_members.csv"),
    "restricted_43993": ("datasets/external_validation/ecbd/ecbd_safetynet_predictions_with_structural_alerts.csv", "ecbd_mtl_predictions_with_sim_with_structural_alerts.csv"),
}
REFERENCES = {
    "100k": REPOSITORY / "datasets/toxicity_datasets/safetynet_100k_cytotoxicity_dataset.csv",
    "39k": REPOSITORY / "datasets/antibiotic_activity_datasets/broad_39k.csv",
}
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def repair_cxsmiles_spacing(text):
    """Insert the required space before a CXSMILES extension, preserving it."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    position = text.find("|")
    if position > 0 and not text[position - 1].isspace():
        text = text[:position] + " " + text[position:]
    return text


@lru_cache(maxsize=160000)
def canonical(smiles):
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(repair_cxsmiles_spacing(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_path(version):
    canonical_path, _legacy = SOURCES[version]
    path = REPOSITORY / canonical_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical ECBD input: {path}")
    return path


def load_sources():
    result, provenance = {}, []
    for version in SOURCES:
        path = source_path(version)
        if version == "assay_101097":
            columns = ["SMILES", "HepG2_Binarized_Toxicity", "HepG2_Percent_Growth_Inhibition", "Max_Tox_Score"]
            columns += [f"{name}_{suffix}" for name in ["Esherichia_coli", "Klebsiella_pneumoniae"] for suffix in ["Binarized_Hit", "Percent_Growth_Inhibition"]]
        elif version == "merged_106373":
            columns = ["eos", "SMILES", "activity_EC", "activity_KP", "activity_HepG2", "value_EC", "value_KP", "value_HepG2", "Max_Tox_Score"]
        else:
            columns = ["SMILES", "Name", "label_ec", "label_kp", "label_hepg2", "value_ec", "value_kp", "value_hepg2", "MaxTox", "maxsim_100K", "maxsim_39K", "maxsim_abx", "MatchedBy"]
        data = pd.read_csv(path, usecols=columns)
        data.insert(0, "source_row", np.arange(len(data)) + 2)
        result[version] = data
        provenance.append({"version": version, "path": str(path), "sha256": digest(path), "rows": len(data)})
    return result, provenance


def normalized_version(version, data, organism):
    code = "ec" if organism == "e_coli" else "kp"
    if version == "assay_101097":
        prefix = "Esherichia_coli" if code == "ec" else "Klebsiella_pneumoniae"
        activity, inhibition = data[prefix + "_Binarized_Hit"], data[prefix + "_Percent_Growth_Inhibition"]
        toxicity, toxicity_value, score = data.HepG2_Binarized_Toxicity, data.HepG2_Percent_Growth_Inhibition, data.Max_Tox_Score
        ids = data.source_row.map(lambda value: f"ecbd_assay_row_{value}")
    elif version == "merged_106373":
        activity, inhibition = data["activity_" + code.upper()].map({"active": 1, "inactive": 0}), data["value_" + code.upper()]
        toxicity = data.activity_HepG2.map({"active": 1, "inactive": 0})
        toxicity_value, score, ids = data.value_HepG2, data.Max_Tox_Score, data.eos
    else:
        activity, inhibition = data["label_" + code], data["value_" + code]
        toxicity, toxicity_value, score = data.label_hepg2, data.value_hepg2, data.MaxTox
        ids = data.source_row.map(lambda value: f"ecbd_restricted_row_{value}")
    frame = pd.DataFrame({"version": version, "organism": organism, "source_id": ids, "source_row": data.source_row,
                          "smiles": data.SMILES, "native_activity_label": pd.to_numeric(activity, errors="coerce"),
                          "antibacterial_growth_inhibition": pd.to_numeric(inhibition, errors="coerce"),
                          "native_toxicity_label": pd.to_numeric(toxicity, errors="coerce"),
                          "hepg2_growth_inhibition": pd.to_numeric(toxicity_value, errors="coerce"),
                          "safetynet_score": pd.to_numeric(score, errors="coerce")})
    # Only molecules eligible under at least one prespecified hit rule need costly
    # similarity calculations. Nonhits cannot enter any audited hit cohort.
    return frame[frame.native_activity_label.eq(1) | frame.antibacterial_growth_inhibition.gt(50)].copy()


def reference_fingerprints():
    arrays, metadata = {}, []
    for name, path in REFERENCES.items():
        column = "Standardized_SMILES"
        data = pd.read_csv(path, usecols=[column])
        seen, fingerprints, invalid = set(), [], 0
        for text in data[column]:
            structure = canonical(text)
            if structure is None:
                invalid += 1
                continue
            if structure in seen:
                continue
            seen.add(structure)
            fingerprints.append(GENERATOR.GetFingerprint(Chem.MolFromSmiles(structure)))
        if not fingerprints:
            raise ValueError(f"No valid fingerprint references in {path}")
        arrays[name] = fingerprints
        metadata.append({"reference": name, "path": str(path), "sha256": digest(path), "input_column": column,
                         "source_rows": len(data), "unique_valid_isomeric_structures": len(fingerprints), "invalid_rows": invalid,
                         "cxsmiles_spacing_repairs": sum(isinstance(text, str) and repair_cxsmiles_spacing(text) != text.strip() for text in data[column])})
        print(f"Figure3 similarity audit: {name} reference has {len(fingerprints)} unique structures", flush=True)
    return arrays, metadata


def metric_summary(frame):
    if frame.empty:
        return {"n": 0, "nontoxic": 0, "toxic": 0, "auroc": np.nan, "average_precision": np.nan}
    y, score = frame.toxicity_label.astype(int), frame.safetynet_score
    return {"n": len(frame), "nontoxic": int(y.eq(0).sum()), "toxic": int(y.eq(1).sum()),
            "auroc": roc_auc_score(y, score) if y.nunique() == 2 else np.nan,
            "average_precision": average_precision_score(y, score) if y.nunique() == 2 else np.nan}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources, provenance = load_sources()
    candidates = pd.concat([normalized_version(version, data, organism) for version, data in sources.items() for organism in ["e_coli", "k_pneumoniae"]], ignore_index=True)
    candidates["canonical_isomeric_smiles"] = candidates.smiles.map(canonical)
    references, reference_metadata = reference_fingerprints()
    query_structures = candidates.canonical_isomeric_smiles.dropna().unique()
    similarity_rows = []
    for structure in query_structures:
        fingerprint = GENERATOR.GetFingerprint(Chem.MolFromSmiles(structure))
        similarity_rows.append({"canonical_isomeric_smiles": structure,
            **{"computed_maxsim_" + name: max(DataStructs.BulkTanimotoSimilarity(fingerprint, reference)) for name, reference in references.items()}})
    similarity = pd.DataFrame(similarity_rows)
    candidates = candidates.merge(similarity, on="canonical_isomeric_smiles", how="left", validate="many_to_one")
    # Transfer only similarity metadata, never labels or prediction scores, across
    # an exact isomeric identity join. Keep min/max when duplicate annotations differ.
    restricted = sources["restricted_43993"]
    restricted = restricted.assign(canonical_isomeric_smiles=restricted.SMILES.map(canonical))
    provided = restricted.groupby("canonical_isomeric_smiles").agg(
        provided_maxsim_100k_min=("maxsim_100K", "min"), provided_maxsim_100k_max=("maxsim_100K", "max"),
        provided_maxsim_39k_min=("maxsim_39K", "min"), provided_maxsim_39k_max=("maxsim_39K", "max"),
        provided_maxsim_abx_min=("maxsim_abx", "min"), provided_maxsim_abx_max=("maxsim_abx", "max"),
        provided_similarity_source_rows=("source_row", lambda values: ";".join(values.astype(str))))
    candidates = candidates.merge(provided, on="canonical_isomeric_smiles", how="left", validate="many_to_one")
    candidates.to_csv(OUTPUT_DIR / "ecbd_version_hit_candidates_with_similarity.csv", index=False)
    summary, memberships = [], []
    for (version, organism), group in candidates.groupby(["version", "organism"]):
        for activity_rule in ["native", "growth_inhibition_gt_50", "growth_inhibition_gt_80"]:
            hits = group.native_activity_label.eq(1) if activity_rule == "native" else group.antibacterial_growth_inhibition.gt(50 if activity_rule.endswith("50") else 80)
            for toxicity_rule in ["native", "growth_inhibition_gt_20", "growth_inhibition_gt_50"]:
                frame = group[hits].copy()
                if toxicity_rule == "native":
                    frame["toxicity_label"] = frame.native_toxicity_label
                else:
                    threshold = 20 if toxicity_rule.endswith("20") else 50
                    frame["toxicity_label"] = np.where(frame.hepg2_growth_inhibition.notna(), frame.hepg2_growth_inhibition.gt(threshold).astype(float), np.nan)
                valid = frame.toxicity_label.isin([0, 1]) & frame.safetynet_score.between(0, 1) & frame.canonical_isomeric_smiles.notna()
                missing_count = int((~valid).sum())
                frame = frame[valid].copy()
                frame["activity_rule"] = activity_rule
                frame["toxicity_rule"] = toxicity_rule
                memberships.append(frame)
                conditions = {"none": pd.Series(True, index=frame.index),
                    "computed_100k_lt_0_9": frame.computed_maxsim_100k.lt(.9),
                    "computed_39k_lt_0_9": frame.computed_maxsim_39k.lt(.9),
                    "computed_both_lt_0_9": frame.computed_maxsim_100k.lt(.9) & frame.computed_maxsim_39k.lt(.9),
                    "computed_100k_lt_1": frame.computed_maxsim_100k.lt(1),
                    "provided_100k_lt_0_9_known_matches_only": frame.provided_maxsim_100k_max.lt(.9),
                    "provided_39k_lt_0_9_known_matches_only": frame.provided_maxsim_39k_max.lt(.9),
                    "provided_100k_39k_lt_0_9_abx_lt_0_7_known_matches_only": frame.provided_maxsim_100k_max.lt(.9) & frame.provided_maxsim_39k_max.lt(.9) & frame.provided_maxsim_abx_max.lt(.7)}
                for novelty_rule, keep in conditions.items():
                    selected = frame[keep].copy()
                    # Report the native source-row cohort and a strict molecule-
                    # consensus version separately so duplicates are visible.
                    for duplication_rule in ["source_rows", "unique_consistent_isomeric_molecules"]:
                        chosen = selected
                        if duplication_rule != "source_rows":
                            groups = selected.groupby("canonical_isomeric_smiles")
                            acceptable = groups.toxicity_label.nunique().eq(1) & groups.safetynet_score.agg(lambda values: values.max() - values.min() <= 1e-6)
                            chosen = selected[selected.canonical_isomeric_smiles.isin(acceptable[acceptable].index)].drop_duplicates("canonical_isomeric_smiles")
                        summary.append({"version": version, "organism": organism, "activity_rule": activity_rule, "toxicity_rule": toxicity_rule,
                            "novelty_rule": novelty_rule, "duplication_rule": duplication_rule, "missing_label_score_or_structure": missing_count,
                            "native_hits_with_known_provided_similarity": int(frame.provided_maxsim_100k_max.notna().sum()), **metric_summary(chosen)})
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "ecbd_version_label_novelty_sensitivity.csv", index=False)
    pd.concat(memberships, ignore_index=True).to_csv(OUTPUT_DIR / "ecbd_version_variant_memberships.csv", index=False)
    manifest = {"source_versions": provenance, "references": reference_metadata, "rdkit_version": rdBase.rdkitVersion,
        "fingerprint_assumptions": {"type": "Morgan binary", "radius": 2, "bits": 2048, "include_chirality": False,
            "query_structure": "full source SMILES canonicalized without salt stripping or tautomer changes",
            "reference_structure": "provided Standardized_SMILES column", "similarity": "Tanimoto maximum over complete valid unique reference collection"},
        "unique_query_structures": len(query_structures), "candidate_source_rows": len(candidates),
        "interpretation": ["Native activity labels and numeric growth-inhibition thresholds are distinct policies, not interchangeable fields.",
            "Stored and calculated similarities are analyzed separately.",
            "Only provided similarity annotations are joined across identical structures; labels and model scores remain with their own source version.",
            "Known-antibiotic <0.7 and training-library <0.9 rules are a sensitivity analysis, not asserted retrospective cohort rules.",
            "Each cohort policy is evaluated separately."]}
    (OUTPUT_DIR / "ecbd_version_similarity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Figure3 ECBD audit: {len(summary)} label/version/novelty summaries from {len(query_structures)} unique hit candidates", flush=True)


if __name__ == "__main__":
    main()
