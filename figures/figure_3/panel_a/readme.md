# Figure 3a · Antibacterial chemical space

Run `figure_3a.py` directly to draw only this panel using the common `chemical_space.py` calculation two directories above. Molecular inputs and derived coordinate caches use absolute repository paths. The runner writes its SVG, PNG, and status files beside itself regardless of the process working directory.

The shared calculation uses the 39K, ECBD, CO-ADD, and Drug Repurposing Hub molecular libraries, with Morgan fingerprints, singular-value decomposition, and t-distributed stochastic neighbor embedding. Settings and coordinate provenance are linked in `chemical_space_status.json`. `chemical_space_initial_svd` records the initial projection; `chemical_space_tsne` is the refined output. A validated coordinate cache is reused when its input hashes and settings match.
