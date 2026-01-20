# beta_eff calibration

SA schedules do not correspond to a physical temperature. We estimate an effective beta by comparing SA samples to Gibbs reference samples.

## Goal
Find beta_eff such that Gibbs samples at beta_eff best match the SA samples in reduced space.

## Distance metric
The pipeline computes a combined distance between two sample sets:
- mean per-residue JS divergence of marginals
- mean JS divergence of pairwise edge joints

Weights are controlled by `beta-eff-w-marg` and `beta-eff-w-pair`.

## How the scan is run
- Requires `gibbs-method rex` to provide Gibbs samples across a beta ladder.
- If `beta-eff-grid` is not specified, the ladder betas are used.
- If requested betas are missing, the pipeline runs an extra REX pass.

## Outputs
- `beta_scan.html`: D(beta) curves for each SA schedule.
- `run_summary.npz`: `beta_eff`, `beta_eff_by_schedule`, `beta_eff_grid`, and distances.
- `run_metadata.json`: quick access to the chosen beta_eff values.

## Related docs
- [Potts analysis overview](doc:potts_overview)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [SA/QUBO sampling](doc:potts_sa_qubo)
