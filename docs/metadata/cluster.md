# Cluster Metadata (`cluster_metadata.json`)

Location
`projects/<project_id>/systems/<system_id>/clusters/<cluster_id>/cluster_metadata.json`

Purpose
Cluster-level metadata for a single cluster folder. This file replaces cluster entries previously stored in `system.json`.

Required fields
- `cluster_id` (string)
- `created_at` (ISO8601 string)

Common fields
- `name` (string or null)
- `status` (string) – `"queued" | "running" | "finished" | "failed"`
- `progress` (int)
- `status_message` (string or null)
- `job_id` (string or null)
- `path` (string) – relative path to `cluster.npz` (e.g. `clusters/<cluster_id>/cluster.npz`)
- `state_ids` (array of strings) – selected macro state ids
- `metastable_ids` (array of strings) – selected metastable state ids (often same as `state_ids`)
- `analysis_mode` (string or null) – `"macro" | "metastable"` (or null)
- `cluster_algorithm` (string) – e.g. `"density_peaks"`
- `cluster_params` (object) – algorithm params
- `max_cluster_frames` (int or null)
- `random_state` (int)
- `generated_at` (ISO8601 string or null)
- `contact_edge_count` (int or null)
- `assigned_state_paths` (object) – `{state_id: relative_path_to_md_eval_npz}`
- `assigned_metastable_paths` (object) – `{metastable_id: relative_path_to_md_eval_npz}`

Notes
- Potts models and samples are **not** stored in this file.
- Potts models live under `clusters/<cluster_id>/potts_models/<model_id>/model_metadata.json`.
- Samples live under `clusters/<cluster_id>/samples/<sample_id>/sample_metadata.json`.

