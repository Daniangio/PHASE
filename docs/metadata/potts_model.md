# Potts Model Metadata (`model_metadata.json`)

Location
`projects/<project_id>/systems/<system_id>/clusters/<cluster_id>/potts_models/<model_id>/model_metadata.json`

Purpose
Metadata for a single Potts model stored in its own folder next to the `.npz` file.

Required fields
- `model_id` (string)
- `created_at` (ISO8601 string)

Common fields
- `name` (string)
- `path` (string) – relative path to the `.npz` model file (e.g. `clusters/<cluster_id>/potts_models/<model_id>/<name>.npz`)
- `source` (string) – e.g. `offline`, `potts_fit`, `potts_delta_fit`, `simulation`, `upload`
- `params` (object) – fit hyperparameters and provenance

Typical `params` keys
- `fit_mode` (string) – `"standard"` or `"delta"`
- `delta_kind` (string) – e.g. `delta_patch`, `model_patch`, `delta_active`, `delta_inactive`, `model_active`, `model_inactive`
- `state_ids` (array of strings, optional)
- `base_model_id` / `base_model_path` (string, optional)
- PLM fit settings (epochs, lr, batch_size, etc.)

Notes
- The model folder should contain exactly one `.npz` model artifact.
- If `path` is omitted, the loader will fall back to the single `.npz` file in the folder.

