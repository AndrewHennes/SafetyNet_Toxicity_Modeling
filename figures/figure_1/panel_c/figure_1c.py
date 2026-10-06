"""Calculate Figure 1c molecular-property densities.

Descriptors are calculated from the internal and Tox21 molecular structures.
The FDA-approved comparator requires an additional molecular cohort."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE
DERIVED = Path("/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/datasets/derived/figure_1")
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parents[1] / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED

REPOSITORY = Path("/Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private")
SCREEN = REPOSITORY / "datasets/toxicity_datasets/safetynet_100k_cytotoxicity_dataset.csv"
TOX21 = REPOSITORY / "datasets/external_validation/tox21/tox21_labels_and_safetynet_predictions.csv"
PROPERTIES = ("MW", "logP", "TPSA", "QED")
COLORS = ("#50caff", "#9a70ba")


DESCRIPTOR_POLICY = "raw_smiles_preserved_cx_separator_v2"


def normalized_cxsmiles(value):
    """Insert only the missing separator before a CXSMILES extension.

    RDKit expects a space before the first pipe; the molecular graph text and all
    extension metadata are preserved. This is format repair, not standardization.
    """
    if isinstance(value, str) and "|" in value:
        graph, separator, extension = value.partition("|")
        if graph and not graph[-1].isspace():
            return graph + " " + separator + extension
    return value


def molecular_properties(smiles: pd.Series, dataset: str) -> tuple[pd.DataFrame, int]:
    """Calculate four descriptors, retaining all rows including duplicates.

    Invalid molecules are recorded as missing and excluded property-by-property.
    No unrecorded subsampling is used. RDKit's default sanitization is applied.
    """
    records = []
    invalid = 0
    for row_index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(normalized_cxsmiles(value)) if isinstance(value, str) else None
        if molecule is None:
            records.append((row_index, np.nan, np.nan, np.nan, np.nan))
            invalid += 1
            continue
        records.append((row_index, Descriptors.MolWt(molecule), Crippen.MolLogP(molecule),
                        rdMolDescriptors.CalcTPSA(molecule), QED.qed(molecule)))
    frame = pd.DataFrame(records, columns=["source_row", *PROPERTIES])
    frame.insert(0, "dataset", dataset)
    return frame, invalid


def get_properties() -> tuple[pd.DataFrame, dict]:
    """Reuse valid descriptors and repair rejected CXSMILES rows transparently."""
    from rdkit import rdBase
    signature = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                 for label,path in (("internal_100k",SCREEN),("tox21",TOX21))}
    DERIVED.mkdir(parents=True, exist_ok=True)
    cache, metadata = DERIVED / "molecular_properties.csv", DERIVED / "descriptor_provenance.json"
    previous = json.loads(metadata.read_text()) if metadata.is_file() else None
    same_inputs = bool(previous and list(previous["source_sha256"].values()) == list(signature.values())
                       and previous["rdkit_version"] == rdBase.rdkitVersion)
    if cache.is_file() and same_inputs and previous.get("descriptor_policy") == DESCRIPTOR_POLICY:
        previous["source_files"] = {"internal_100k":str(SCREEN),"tox21":str(TOX21)}
        metadata.write_text(json.dumps(previous,indent=2)+"\n")
        return pd.read_csv(cache), previous
    sources = {"internal_100k":pd.read_csv(SCREEN,usecols=["SMILES"])["SMILES"],
               "tox21":pd.read_csv(TOX21,usecols=["Standardised_SMILES"])["Standardised_SMILES"]}
    failures, repair_rows = {}, []
    if cache.is_file() and same_inputs:
        # Version 1 used the same descriptors on exactly the same valid rows.
        # Only rows previously rejected by RDKit need to be recalculated.
        result = pd.read_csv(cache)
        for dataset,values in sources.items():
            rejected = result.index[result.dataset.eq(dataset) & result.MW.isna()]
            for index in rejected:
                source_row = int(result.loc[index,"source_row"])
                original = values.iloc[source_row]
                repaired = normalized_cxsmiles(original)
                frame, invalid = molecular_properties(pd.Series([original]),dataset)
                if not invalid:
                    result.loc[index,list(PROPERTIES)] = frame.loc[0,list(PROPERTIES)].to_numpy()
                    repair_rows.append({"dataset":dataset,"source_row":source_row,
                                        "repair":"insert space before CXSMILES extension" if repaired != original else "reparsed"})
            failures[dataset] = {"input_rows":len(values),
                                 "invalid_smiles":int(result.loc[result.dataset.eq(dataset),"MW"].isna().sum()),
                                 "recovered_from_initial_cache":sum(r["dataset"] == dataset for r in repair_rows)}
    else:
        frames=[]
        for dataset,values in sources.items():
            frame, invalid = molecular_properties(values,dataset)
            frames.append(frame)
            failures[dataset] = {"input_rows":len(values),"invalid_smiles":invalid}
            for source_row, original in enumerate(values):
                if isinstance(original,str) and normalized_cxsmiles(original) != original:
                    repair_rows.append({"dataset":dataset,"source_row":source_row,
                                        "repair":"insert space before CXSMILES extension"})
        result=pd.concat(frames,ignore_index=True)
    result.to_csv(cache,index=False)
    pd.DataFrame(repair_rows,columns=["dataset","source_row","repair"]).to_csv(DERIVED/"cxsmiles_format_repairs.csv",index=False)
    info={"source_sha256":signature,"source_files":{"internal_100k":str(SCREEN),"tox21":str(TOX21)},
          "rdkit_version":rdBase.rdkitVersion,"descriptor_policy":DESCRIPTOR_POLICY,"cohorts":failures,
          "format_repair":"Insert a missing space before the first CXSMILES pipe, preserving graph and extension metadata.",
          "format_repair_rows":len(repair_rows),"kde":"scipy gaussian_kde, Scott bandwidth, all valid rows",
          "limitations":"Raw molecular forms with CXSMILES spacing repair; Scott bandwidth."}
    metadata.write_text(json.dumps(info,indent=2)+"\n")
    return result,info

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": .65,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    properties, metadata = get_properties()
    density_rows = []
    limits = [(60, 900), (-8, 8), (0, 260), (0, 1)]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), layout="constrained")
    for index, (ax, prop, bounds) in enumerate(zip(axes, PROPERTIES, limits)):
        x = np.linspace(*bounds, 400)
        for dataset, color, label in zip(("internal_100k", "tox21"), COLORS, ("Internal 100K", "Tox21")):
            values = properties.loc[properties.dataset.eq(dataset), prop].dropna().to_numpy()
            density = gaussian_kde(values)(x)
            ax.plot(x, density, color=color, lw=1)
            ax.fill_between(x, density, color=color, alpha=.3, label=label)
            density_rows.extend({"dataset": dataset, "property": prop, "x": v, "density": d}
                                for v, d in zip(x, density))
        ax.set(title=prop, xlim=bounds, ylim=(0, None))
        if index == 0:
            ax.set_ylabel("Density")
        if index == 3:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("c  Molecular properties", fontsize=11)
    fig.savefig(OUTPUT / "molecular_property_densities.svg")
    fig.savefig(OUTPUT / "molecular_property_densities.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(density_rows).to_csv(OUTPUT / "property_density_values.csv", index=False)
    (OUTPUT / "validation.json").write_text(json.dumps({
        "status": "partially_recomputed", "missing_input": "FDA-approved molecular cohort",
        "descriptor_provenance": metadata}, indent=2) + "\n")


if __name__ == "__main__":
    main()
