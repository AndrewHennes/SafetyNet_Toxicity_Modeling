# Figure 2e: ECBD structural-alert comparison

Recomputes all eight RDKit alert catalogs, including Brenk, and compares them with SafetyNet using the same shared ECBD cohort as panel d. This runner prepares that shared cohort if needed and does not generate panel d or any other panel.

Computed molecular flags, supplied-versus-computed flag checks, catalog provenance, filter metric CSV, SVG/PNG plots and panel status are saved here. Cohort and score-version audits shared with panel d remain in `../shared/ecbd/`.

Invalid structures are excluded consistently across the compared methods.

Run this panel independently from any working directory:

```sh
/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/python /Users/asselism/Desktop/Collins_Lab/Repositories/safetynet-private/figures/figure_2/panel_e/figure_2e.py
```

The runner accepts no command-line options and changes only this panel’s results and any required shared inputs. The scientific formulas and cohort policies are unchanged by the directory reorganization.
