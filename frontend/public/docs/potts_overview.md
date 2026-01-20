# Potts analysis overview

This page summarizes the reduced-state Potts pipeline used by the Potts analysis panel.

## Pipeline steps
1) Load a cluster NPZ file produced by the clustering workflow.
2) Build a Potts model over residue microstate labels.
3) Sample the model (Gibbs or replica exchange) and a QUBO proxy (SA).
4) Compare samples to MD marginals and pairwise edge statistics.
5) Save summary artifacts for visualization.

## Inputs (cluster NPZ)
- `merged__labels`: shape (T, N) integer labels per frame and residue.
- `merged__cluster_counts`: shape (N,) number of microstates per residue.
- `contact_edge_index`: shape (2, E) edges for pairwise couplings.
- `residue_keys`: optional labels for plotting.

## Outputs
- `run_summary.npz`: full sampling summaries, marginals, JS divergences, and metadata.
- `run_metadata.json`: human-readable settings and quick stats.
- `marginals.html`: interactive comparison plot.
- `beta_scan.html`: only if beta_eff calibration is enabled.

## Diagnostics
- Per-residue marginals and JS divergence versus MD.
- Pairwise edge joint distributions for additional validation.
- Optional beta_eff calibration for SA schedules.

## Related docs
- [Potts model fitting](doc:potts_model)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [SA/QUBO sampling](doc:potts_sa_qubo)
- [beta_eff calibration](doc:potts_beta_eff)
