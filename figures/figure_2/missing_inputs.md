# Inputs still needed for full Figure 2 analysis

| Panel | Current analysis | Required to complete or verify it |
|---|---|---|
| a | Reference schematic retained | No statistical analysis applies |
| b | Internal benchmark | Train/validation/test or fold assignments, baseline model configurations, held-out predictions and labels, and Wilcoxon pairing/options |
| c | External benchmark | Assay/cohort/label policies and each baseline and SafetyNet fold prediction set |
| d | ROC/precision-recall analysis on the declared ECBD cohort | Compound list, toxicity definition, score/model version, training-overlap reference, and confidence-band construction |
| e | All eight alerts, including Brenk, computed from structures | Confirmation of original catalog release, structure handling and same final ECBD cohort as panel d |
| f | Forty-two assays computed from molecular labels and scores | Similarity-reference identity, fingerprint settings, cohort membership, and assay-category mapping |
| g | Animal toxicity correlation | Compound-level dose observations, identities, species/routes, units, target transform, prediction scores, and inclusion/exclusion policy |
| h | Achiral Morgan novelty-filtered analysis with full-cohort and chiral sensitivity analyses | Reference cohort, molecular preprocessing, Morgan parameters, and clinical cohort membership |

The workflow evaluates saved compound-level predictions. Training and benchmark-fold generation are separate workflows.

The 101,097-row ECBD table uses a >20% growth-inhibition toxicity definition and six saved ensemble members. The restricted 43,993-row table uses name-matched records, its native HepG2 labels, and its own predictions.
