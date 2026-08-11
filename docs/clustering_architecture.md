# Clustering Fan-Out Architecture

This document describes the parallel clustering pipeline used for residue clustering in the webserver. The goal is to
preprocess once, then fan out per-residue clustering to multiple RQ workers, and finally reduce the outputs into the
standard cluster NPZ format.

## Overview

The clustering job is split into three phases:

1) **Preprocess (orchestrator job)**
   - Collect all frames for the selected metastable states.
   - Build a merged per-residue angle tensor with shape `(n_frames, n_residues, 5)` (`phi`, `psi`, `omega`, `chi1`, `chi2`).
   - Persist intermediate inputs to a workspace directory.

2) **Chunk jobs (fan-out)**
   - One RQ job per residue.
   - Each job loads the shared angles array (memmap), clusters its residue, and writes labels to disk.

3) **Reduce (orchestrator job)**
   - Load all chunk outputs.
   - Assemble `merged__labels` and `merged__cluster_counts`.
   - Write the final cluster NPZ and metadata JSON.

The orchestrator stays alive while chunk jobs run so the frontend can poll a single job ID for progress updates.

## Workspace Layout

The preprocess step creates a workspace under:

`data/projects/<project_id>/systems/<system_id>/metastable/clusters/<cluster_id>_work/`

Typical files:

- `angles.npy` (float32, shape `(n_frames, n_residues, 5)`)
- `frame_state_ids.npy` (state id for each frame)
- `frame_meta_ids.npy` (metastable id for each frame)
- `frame_indices.npy` (frame index inside each state trajectory)
- `contact_edge_index.npy` (2 x n_edges)
- `contact_mode.npy`, `contact_cutoff.npy`
- `manifest.json` (inputs + cluster parameters)
- `chunk_0000.npz`, `chunk_0001.npz`, ... (per-residue outputs)

## Chunk Job Output

Each chunk output (`chunk_XXXX.npz`) contains:

- `labels` (int32, length `n_frames`)
- `cluster_count` (int32, scalar)
- `diagnostics_json` (optional; only stored for residue 0 when using ToMATo)

## Progress Reporting

The orchestrator updates progress based on completed chunk jobs:

- 0–10%: initialization + preprocessing
- 10–80%: chunk completion ratio
- 80–90%: reduction and metadata write
- 90–100%: finalization and persistence

The frontend polls the orchestrator job ID, so progress is always tied to a single job.

## Failure Handling

- If any chunk job fails, the orchestrator raises and marks the cluster entry as failed.
- Workspaces are kept on disk for debugging unless explicitly cleaned.

## Assigning MD Frames to an Existing Cluster

When we need to label frames from other MD trajectories (states or metastable subsets) using an already-built cluster
NPZ, we do a per-residue k-nearest-neighbor (kNN) assignment in a periodic angle embedding:

- For each residue, we build a reference KD-tree from the original clustered frames.
- Each frame’s residue angles are embedded as `sin/cos` pairs (periodic, so the distance is meaningful across wrap-around).
- For every target frame, we query the `k` nearest neighbors in that embedding and assign the majority label.
- Default `k_neighbors` is 10. Missing residues or empty references yield label `-1` for that residue.

This is distance-based, but done in the sin/cos embedding rather than raw angles.
See `backend/services/metastable_clusters.py` (`assign_cluster_labels_to_states` and `_assign_labels_from_reference`).

## Symmetric Chi2 Folding

Some sidechains have equivalent torsional states that differ only by a symmetric flip. For DADApy density-peak clustering, PHASE folds `chi2` for selected symmetric residue types before fitting or predicting cluster labels:

- `PHE`, `TYR`, `ASP`

The rule maps `chi2` to a doubled-angle periodic coordinate, `2 * chi2 mod 2π`, so states separated by a 180 degree ring flip are treated as the same clustering coordinate. This covers cases such as `+90`/`-90` degrees and also `0`/`180` degrees. This is only applied to the clustering/prediction copy of the descriptor matrix. Stored descriptor NPZ files and descriptor visualizations keep the original physical `chi2` values.

If descriptor keys are numeric-only, for example `res_54`, clustering resolves residue names from a representative state structure before deciding whether the chi2 symmetry rule applies. The saved `residue_keys` remain unchanged for UI/API compatibility, while the internal clustering key becomes effectively `res_54_PHE` for symmetry detection and model metadata.

The cluster metadata stores `descriptor_symmetry` with the rule version, descriptor index, symmetric residue names, and candidate residue keys for reproducibility. A listed candidate with no available `chi2` column is unaffected.

## Worker Configuration

Parallel fan-out requires multiple RQ worker processes. If only one worker is available, the orchestrator
falls back to the single-process path to avoid deadlock.

Recommended:

- Run multiple RQ workers in the `phase-jobs` queue (e.g., 4–32 depending on CPU/RAM).
- Avoid over-subscribing BLAS/OpenMP threads when using many processes.

## Code References

- Orchestrator job: `backend/tasks.py` (`run_cluster_job`)
- Preprocess / chunk / reduce: `backend/services/metastable_clusters.py`
- Cluster job API endpoint: `backend/api/v1/routes/clusters.py`
