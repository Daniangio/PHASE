# Sample Metadata (`sample_metadata.json`)

Location
`projects/<project_id>/systems/<system_id>/clusters/<cluster_id>/samples/<sample_id>/sample_metadata.json`

Purpose
Metadata for a single sample folder (MD evaluations, Gibbs samples, SA samples, etc.).

Required fields
- `sample_id` (string)
- `created_at` (ISO8601 string)

Common fields
- `name` (string)
- `type` (string) – e.g. `md_eval`, `potts_sampling`, `upload`
- `method` (string or null) – e.g. `gibbs`, `sa`, `upload`
- `state_id` (string or null) – for MD evaluations
- `metastable_id` (string or null) – for MD evaluations
- `model_id` (string or null) – single model reference
- `model_ids` (array of strings or null) – multiple models
- `model_names` (array of strings or null) – display names for models
- `paths` (object) – relative paths to artifacts in the sample folder
- `path` (string or null) – optional shortcut to the primary NPZ
- `params` (object, optional) – sampling hyperparameters
- `summary` (object, optional) – normalized run info (beta, counts, schedules, beta_eff)

Typical `paths` keys
- `summary_npz`
- `metadata_json`
- `marginals_plot`
- `sampling_report`
- `cross_likelihood_report`
- `beta_scan_plot`

Notes
- The sample folder should contain one primary `.npz` artifact (e.g. `run_summary.npz` or `md_eval.npz`).
- If `path` is omitted, the loader will fall back to the single `.npz` file in the folder.
