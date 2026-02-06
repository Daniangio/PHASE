# Analysis Metadata (`analysis_metadata.json`)

Location
`projects/<project_id>/systems/<system_id>/clusters/<cluster_id>/analyses/<analysis_type>/<analysis_id>/analysis_metadata.json`

Purpose
Metadata for a derived Potts analysis artifact (computed after sampling, not during sampling).

Required fields
- `analysis_id` (string)
- `analysis_type` (string) – e.g. `md_vs_sample`, `model_energy`
- `created_at` (ISO8601 string)
- `project_id` (string)
- `system_id` (string)
- `cluster_id` (string)
- `paths.analysis_npz` (string) – relative path to the analysis `.npz`

Common fields
- `md_sample_id` / `md_sample_name` (string, for `md_vs_sample`)
- `sample_id` / `sample_name` (string)
- `sample_type` / `sample_method` (string or null)
- `model_id` / `model_name` (string or null, for `model_energy`)
- `drop_invalid` (bool) – whether invalid SA rows were dropped
- `md_label_mode` (string) – `assigned` or `halo`
- `summary` (object) – small scalar summary (means/medians/counts)

Analysis NPZ schema
- `md_vs_sample`:
  - `node_js` (N,) float
  - `edge_js` (E,) float (may be empty if the cluster has no edges)
- `model_energy`:
  - `energies` (T,) float
- `lambda_sweep`:
  - `lambdas` (M,) float
  - `node_js_mean` (3, M) float (vs 3 reference MD samples)
  - `edge_js_mean` (3, M) float
  - `combined_distance` (3, M) float (weighted by `alpha`)
  - `deltaE_mean` (M,) float (ΔE = E_A - E_B on λ-samples)
  - `deltaE_q25` / `deltaE_q75` (M,) float (IQR for ΔE)
  - `sample_ids` / `sample_names` (M,) str
  - `ref_md_sample_ids` / `ref_md_sample_names` (3,) str

Notes
- Analyses live under `clusters/<cluster_id>/analyses/` and can be safely deleted/regenerated without affecting the raw samples.
