# SA/QUBO sampling

This page describes how Potts sampling is mapped to a QUBO and sampled with simulated annealing (SA).

## QUBO mapping
Each residue uses one-hot binary variables z_{r,k} with a constraint sum_k z_{r,k} = 1.
The Potts energy is embedded into a QUBO with a quadratic penalty:

- Penalty term: lambda_r (sum_k z_{r,k} - 1)^2
- lambda_r is chosen from Potts energy bounds and scaled by `penalty-safety`.

The QUBO energy is scaled by beta before sampling.

## Simulated annealing
The implementation uses neal's `SimulatedAnnealingSampler` as a classical baseline.
Controls:
- `sa-reads`: number of SA reads (independent runs).
- `sa-sweeps`: sweeps per read.
- Optional beta schedule: `sa-beta-hot`, `sa-beta-cold`, or multiple schedules via `sa-beta-schedule`.

An "SA auto" schedule is always run using neal's defaults.

## Validity and repair
Samples can violate one-hot constraints. The pipeline reports invalid rates.
If `repair=argmax`, invalid samples are coerced to a valid label per residue.

## Related docs
- [Potts analysis overview](doc:potts_overview)
- [Potts model fitting](doc:potts_model)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [beta_eff calibration](doc:potts_beta_eff)
