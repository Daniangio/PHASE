# Delta JS (A/B/Other): How To Read

This analysis compares each sample against two references (A and B) using Jensen-Shannon (JS) distances.

- `JS(A)`: distance of the sample distribution from reference A
- `JS(B)`: distance of the sample distribution from reference B
- lower JS means more similar

Potts models are optional:
- if provided, model pair is used to define the edge set and can auto-infer references from model state_ids
- if not provided, you must choose an edge mode:
  - `cluster`: use edges from `cluster.npz`
  - `all_vs_all`: complete graph over residues
  - `contact`: build edges from structure contacts (same contact logic used in Potts fitting)
and provide explicit reference sample sets A and B

## Exact Formulas (Current `upsert_delta_js_analysis`)

Notation:
- `X^s` = labels of sample `s`, shape `(T_s, N)` (frames × residues)
- `K_i` = number of clusters for residue `i`
- `E` = edge set used by analysis
- `JS(p,q)` = Jensen-Shannon divergence

Reference distributions are built by pooling frames from selected reference samples:

1. Node reference marginals
- For side `A` (similarly `B`):
`p_i^A(k) = (sum over ref samples r in A of count_r(i,k)) / (sum over k' counts total)`

2. Edge reference joints
- For edge `(i,j)`:
`p_{ij}^A(k,l) = pooled joint histogram over A references, normalized`

Discriminative weights (A vs B):
- Node: `D_i = JS(p_i^A, p_i^B)`  (`D_residue`)
- Edge: `D_{ij} = JS(p_{ij}^A, p_{ij}^B)`  (`D_edge`)

Top-K selection:
- Top nodes: largest `D_i` (saved in `top_residue_indices`)
- Top edges: largest `D_{ij}` (saved in `top_edge_indices`)

Per-sample distances:
- Node-to-A: `JS_i^A(s) = JS(p_i^s, p_i^A)`
- Node-to-B: `JS_i^B(s) = JS(p_i^s, p_i^B)`
- Edge-to-A: `JS_{ij}^A(s) = JS(p_{ij}^s, p_{ij}^A)` (top edges only)
- Edge-to-B: `JS_{ij}^B(s) = JS(p_{ij}^s, p_{ij}^B)` (top edges only)

Weighted aggregates (stored as scalar summaries in analysis NPZ):
- Node-weighted:
`JS_node_weighted^A(s) = sum_i D_i * JS_i^A(s) / sum_i D_i`
`JS_node_weighted^B(s) = sum_i D_i * JS_i^B(s) / sum_i D_i`

- Edge-weighted (top edges, weights = `D_{ij}`):
`JS_edge_weighted^A(s) = sum_(i,j in topE) D_{ij} * JS_{ij}^A(s) / sum_(i,j in topE) D_{ij}`
`JS_edge_weighted^B(s) = sum_(i,j in topE) D_{ij} * JS_{ij}^B(s) / sum_(i,j in topE) D_{ij}`

Final mixed score (`node_edge_alpha = alpha`):
- `JS_mixed^A(s) = (1 - alpha) * JS_node_weighted^A(s) + alpha * JS_edge_weighted^A(s)`
- `JS_mixed^B(s) = (1 - alpha) * JS_node_weighted^B(s) + alpha * JS_edge_weighted^B(s)`

If no edges are available, edge-weighted terms fall back to node-weighted values.

## Colors

- Red: more A-like (low `JS(A)`, high `JS(B)`)
- Blue: more B-like (low `JS(B)`, high `JS(A)`)
- Green: similar to both (both low)
- Purple: far from both (both high / novel)

Colors are blended from these four modes, not hard-thresholded.

## Node and Edge Weighting

Node and edge separability weights are computed from A-vs-B reference distance:

- node weight: `w_i = JS(p_i^A, p_i^B)`
- edge weight: `w_ij = JS(p_ij^A, p_ij^B)`

These weights are used to build weighted aggregate distances and to support edge-weighted residue blending in 3D.

### Intuition (Simple)

Think of each edge as a "vote" about whether a sample looks like A or B.

- If an edge is very different between A and B, it gets a **high weight** (`w_ij` large).
- If an edge is similar in A and B, it gets a **low weight** (`w_ij` small).

Then sample-to-A/B edge distances are averaged using those weights:

- high-information edges count more
- weak/non-discriminative edges count less

So the final edge score is driven mostly by edges that truly separate A from B.

## Hyperparameters

- `top_k_residues`: number of residues kept in top ranking (`D_residue`)
- `top_k_edges`: number of edges kept in top ranking (`D_edge`)
- `node_edge_alpha` in `[0,1]`: interpolation used only for scalar summary outputs (`js_mixed_a/b`), not for the default residue/edge heatmaps
- `ranking_method`: currently only `js_ab`
- `md_label_mode`: `assigned` or `halo`
- `drop_invalid`: drops frames flagged as invalid in sample metadata
- `edge_mode` (when Potts models are not used):
  - `cluster`, `all_vs_all`, `contact`
- contact hyperparameters (`edge_mode=contact`):
  - `contact_state_ids`, `contact_pdbs`, `contact_cutoff`, `contact_atom_mode`

## Practical Interpretation

- Red + low novelty: strong A-like signature
- Blue + low novelty: strong B-like signature
- Green: shared/common behavior (not discriminative)
- Purple: out-of-reference behavior (possible intermediate/other mechanism)

## About Edge-Weighted Blending In Visualizations

- Default heatmaps:
  - `Per-Residue JS` = node-only JS values
  - `Per-Edge JS` = edge-only JS values
- Optional `Edge-weighted node blending` adds a third residue heatmap:
  - for each residue, incident edge JS values are averaged with edge weights `D_edge`
  - result is mixed with node JS by visualization smoothing strength `alpha`:
  - `blended = (1 - alpha) * node + alpha * incident_edge_mean`
- The 3D page uses the same blending logic when the blending flag is enabled.
