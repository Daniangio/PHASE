# Descriptor Metadata (`*_descriptor_metadata.json`)

Location
`projects/<project_id>/systems/<system_id>/descriptors/<state_id>_descriptor_metadata.json`

Purpose
Metadata for descriptor NPZ files produced per macro state.

Fields
- `descriptor_keys` (array of strings) – residue keys present in the NPZ
- `residue_mapping` (object) – mapping from residue keys to selection strings
- `n_frames` (int)
- `residue_selection` (string) – selection used to build descriptors

Notes
- `system.json` should **not** store global `descriptor_keys`. This file is the source of truth.

