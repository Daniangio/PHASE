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
- `source` (string or null) – e.g. `offline`, `simulation`, `clustering`, `upload`
- `state_id` (string or null) – for MD evaluations
- `metastable_id` (string or null) – for MD evaluations
- `model_id` (string or null) – single model reference
- `model_ids` (array of strings or null) – multiple models
- `model_names` (array of strings or null) – display names for models
- `paths` (object) – relative paths to artifacts in the sample folder
- `path` (string or null) – optional shortcut to the primary NPZ
- `params` (object, optional) – sampling hyperparameters

Correlated samples (optional)
- `series_kind` (string) – e.g. `lambda_sweep`
- `series_id` (string) – correlation key shared by all samples in the series
- `series_label` (string) – display label for the series
- `lambda` (float) – for `lambda_sweep` samples, the interpolation value in `[0,1]`
- `lambda_index` (int) – index in the lambda grid (0-based)
- `lambda_count` (int) – total number of lambdas in the sweep
- `endpoint_model_a_id` / `endpoint_model_b_id` (string) – endpoint model references

Typical `paths` keys
- `summary_npz`

Notes
- The sample folder should contain one primary `.npz` artifact: `sample.npz`.
- If `path` is omitted, the loader will fall back to the single `.npz` file in the folder.
