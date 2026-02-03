# System Metadata (`system.json`)

Location
`projects/<project_id>/systems/<system_id>/system.json`

Purpose
System-level metadata only. It should not contain cluster, Potts model, or sampling metadata. Those live in their respective subfolders.

Required fields
- `system_id` (string)
- `project_id` (string)
- `name` (string)
- `description` (string or null)
- `created_at` (ISO8601 string)

Common fields
- `status` (string) – `"processing" | "ready" | "single-ready" | "awaiting-descriptor" | "pdb-only" | "empty"`
- `macro_locked` (bool)
- `metastable_locked` (bool)
- `analysis_mode` (string or null) – `"macro" | "metastable"` (or null)
- `residue_selections` (object or array or null)
- `residue_selections_mapping` (object)
- `descriptor_metadata_file` (string or null)
- `metastable_model_dir` (string or null)
- `metastable_states` (array of objects)
- `states` (object mapping `state_id` -> state metadata)

State metadata (per `states[state_id]`)
- `state_id` (string)
- `name` (string)
- `pdb_file` (string or null)
- `trajectory_file` (string or null)
- `descriptor_file` (string or null)
- `descriptor_metadata_file` (string or null)
- `n_frames` (int)
- `stride` (int)
- `source_traj` (string or null)
- `slice_spec` (string or null)
- `residue_selection` (string or null)
- `residue_keys` (array of strings)
- `residue_mapping` (object)
- `metastable_labels_file` (string or null)
- `role` (string or null)
- `created_at` (ISO8601 string)

Derived fields (computed at runtime, not persisted)
- `descriptor_keys` (array of strings)
- `analysis_states` (array of objects)
- `metastable_clusters` (array of cluster entries; see `cluster.md`)

