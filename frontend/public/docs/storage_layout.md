# PHASE Storage Layout

This page summarizes where PHASE stores data under the configured data root.
All persistent artifacts live under `projects/` (e.g., `/app/data/projects`).

## Top-level

- `projects/<project_id>/`
  - `project.json`
  - `systems/<system_id>/`

## System

- `systems/<system_id>/system.json`
- `structures/` (uploaded PDBs)
- `trajectories/` (uploaded trajectories, if copied)
- `descriptors/` (descriptor NPZs per state)
- `clusters/<cluster_id>/` (cluster outputs)
- `results/` (job result JSON files)
  - `jobs/<job_id>.json`

## Cluster

- `clusters/<cluster_id>/cluster.npz` (cluster labels + metadata)
- `clusters/<cluster_id>/models/` (per-residue dp_data pickles)
- `clusters/<cluster_id>/potts_models/<model_id>/` (Potts model + metadata)
- `clusters/<cluster_id>/samples/<sample_id>/` (sampling outputs)
  - MD evaluations and Potts sampling results (summary NPZ, plots, reports)
- `clusters/<cluster_id>/backmapping.npz` (optional backmapping export)

## Notes

- Job metadata is always stored in `systems/<system_id>/results/jobs/`.
- Potts models and samples are **scoped to a cluster**.
- Sampling outputs (Gibbs, REX, SA, MD evaluation) are all stored under
  the corresponding `samples/<sample_id>/` directory.
