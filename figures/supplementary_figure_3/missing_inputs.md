# Required inputs for Supplementary Figure 3

Both panels use the curated DrugBank cohorts for approved systemic antibiotics and antineoplastic compounds, with per-compound SafetyNet scores.

Required fields and metadata

- DrugBank compound identifiers and standardized structures for the approved systemic-antibiotic and approved antineoplastic cohorts.
- The DrugBank release and explicit approval, systemic-use and therapeutic-category inclusion rules.
- The HepG2, HSkMC and IMR-90 predictions for each compound, or exact model checkpoints and embeddings needed to reproduce those scores.
- Any exclusion/deduplication decisions and the original histogram edges and kernel-density settings.

The script records the precise expected input columns in `shared/required_input_schema.json`. It uses the maximum of the three cell scores and defines predicted safe as `<0.2`, matching the stated filtering rule.
