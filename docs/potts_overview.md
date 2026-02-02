# Potts analysis overview

This page summarizes the reduced-state Potts pipeline used by the Potts analysis panel.

## Pipeline steps
1) Load a cluster NPZ file produced by the clustering workflow.
2) Build a Potts model over residue microstate labels (or reuse a pre-fit model).
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
- `potts_model.npz`: fitted Potts model (h/J/edges/K) for reuse.
- `marginals.html`: interactive comparison plot.
- `beta_scan.html`: only if beta_eff calibration is enabled.

## Offline fitting (optional)
You can fit a Potts model offline (e.g., with CUDA) and then reuse it for sampling.

```bash
python -m phase.potts.main \\
  --npz path/to/cluster.npz \\
  --results-dir results/potts_fit \\
  --fit-only \\
  --plm-device cuda
```

Use `--model-npz` to skip fitting and use a pre-fit model in a later run. You can repeat
`--model-npz` (or pass a comma-separated list) to combine multiple models (they are summed).

If you use the webserver, you can upload a pre-fit model NPZ to the cluster via
`POST /api/v1/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/potts_model`.

### Quick setup (uv)
Run the setup script once to create the fitting environment and install dependencies:

```bash
./scripts/potts_setup.sh
source .venv-potts-fit/bin/activate
```

Then run the interactive fitting script (requires an active venv):

```bash
./scripts/potts_fit.sh
```

The fitting script prompts for the PLM device (auto/cuda/cpu) and hyperparameters.

## Delta Potts fits (optional)
Delta fits learn sparse patches on top of a frozen base model. You can select one or more
macro/metastable states to define the patch dataset. Run the interactive script:

```bash
./scripts/potts_delta_fit.sh
```

This produces delta models (patches) and combined models that can be sampled or combined with
other Potts models.

## Diagnostics
- Per-residue marginals and JS divergence versus MD.
- Pairwise edge joint distributions for additional validation.
- Optional beta_eff calibration for SA schedules.

## Related docs
- [Potts model fitting](doc:potts_model)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [SA/QUBO sampling](doc:potts_sa_qubo)
- [beta_eff calibration](doc:potts_beta_eff)
