# Clustering Fan-Out Architecture

This document describes the parallel clustering pipeline used for residue clustering in the webserver. The goal is to
preprocess once, then fan out per-residue clustering to multiple RQ workers, and finally reduce the outputs into the
standard cluster NPZ format.

## Overview

The clustering job is split into three phases:

1) **Preprocess (orchestrator job)**
   - Collect all frames for the selected metastable states.
   - Build a merged per-residue angle tensor with shape `(n_frames, n_residues, 3)` (phi/psi/chi1).
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

- `angles.npy` (float32, shape `(n_frames, n_residues, 3)`)
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
