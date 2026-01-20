# Gibbs sampling and replica exchange

This page describes how Gibbs and replica exchange sampling are run for the Potts model.

## Single-chain Gibbs
Each sweep updates every residue using the conditional distribution given its neighbors.
Controls:
- `gibbs-samples`: number of returned samples.
- `gibbs-burnin`: sweeps discarded before sampling.
- `gibbs-thin`: keep every Nth sweep after burn-in.

The inverse temperature beta scales the energy inside the conditional logits.

## Replica exchange (parallel tempering)
Replica exchange runs multiple Gibbs chains at different betas and swaps adjacent replicas.
Controls:
- `rex-betas`: explicit ladder of betas.
- Auto ladder: `rex-beta-min`, `rex-beta-max`, `rex-n-replicas`, `rex-spacing`.
- Sampling: `rex-rounds`, `rex-burnin-rounds`, `rex-sweeps-per-round`, `rex-thin-rounds`.

Outputs include samples per beta and swap acceptance rates.
These samples are also used for beta_eff calibration when enabled.

## Related docs
- [Potts analysis overview](doc:potts_overview)
- [Potts model fitting](doc:potts_model)
- [SA/QUBO sampling](doc:potts_sa_qubo)
- [beta_eff calibration](doc:potts_beta_eff)
