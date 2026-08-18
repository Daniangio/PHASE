# Endpoint Frustration Analysis

This analysis compares one or more clustered trajectories against a fixed pair of endpoint Potts models:

- `model A`
- `model B`

It stores commitment and local frustration outputs for every selected sample. Delta-energy distributions can be run as a separate analysis from the same web page or from `phase_console`.

## Commitment

Commitment measures which endpoint is locally preferred by the delta parameters:

- residue commitment: `q_i = Pr(Δh_i(x_i) < 0)`
- edge commitment: `q_ij = Pr(ΔJ_ij(x_i, x_j) < 0)`

Interpretation:

- `q ≈ 1`: active-leaning under the chosen A/B ordering
- `q ≈ 0`: inactive-leaning
- `q ≈ 0.5`: locally neutral / mixed

The web page can display:

- raw `q`
- centered `q - 0.5`

## Frustration

Frustration is computed from endpoint-local energies on every frame.

For residues:

- compute residue-local energy under `model A`
- compute residue-local energy under `model B`
- compare each residue against its graph neighborhood in the same frame

For edges:

- compute the edge coupling contribution under `model A`
- compute the edge coupling contribution under `model B`
- compare each selected edge against incident selected edges in the same frame

Raw frustration values are then normalized **per sample** using all values from the whole trajectory:

- node A channel: median / robust scale over all residue raw frustrations in the sample
- node B channel: same
- edge A/B channels: same over selected edges

Saved normalization parameters:

- `node_norm_center_a`, `node_norm_scale_a`
- `node_norm_center_b`, `node_norm_scale_b`
- `edge_norm_center_a`, `edge_norm_scale_a`
- `edge_norm_center_b`, `edge_norm_scale_b`

Derived channels:

- symmetric frustration: `0.5 * (z_A + z_B)`
- polarity frustration: `z_A - z_B`

Interpretation:

- symmetric: hotspots strained under both endpoints
- polarity: direction of the local conflict
- webserver color convention for polarity: red means **more A-like** (less frustrated under model A), blue means more B-like

## Delta Energy

Delta energy is available as an independent analysis. It computes a global endpoint score for every selected frame:

- `ΔE = E_model_A - E_model_B`

Interpretation:

- `ΔE < 0`: the frame is energetically lower under model A
- `ΔE > 0`: the frame is energetically lower under model B
- distribution overlap between trajectories suggests that the endpoint pair does not separate those trajectories globally

The web page shows these values on the `Delta energy distributions` tab. Histogram bars are kept neutral gray; the fitted/smoothed distribution curves carry the sample colors. A display toggle can show either histograms plus fitted curves or fitted curves only. The trajectory selector also controls each curve's color and line style and can add a vertical marker at the fitted-curve maximum. Peak-marker dash and height are configurable; a blank height uses 110% of the tallest displayed curve. When launching this analysis, each selected trajectory can use all frames or a random subset with a per-trajectory frame limit.

## Saved outputs

Main analysis file:

- `clusters/<cluster_id>/analyses/endpoint_frustration/<analysis_id>/analysis.npz`

Per-sample framewise files:

- `clusters/<cluster_id>/analyses/endpoint_frustration/<analysis_id>/samples/<sample_id>.npz`

Main NPZ contains compact per-sample summaries used by the web UI:

- `q_residue_all`
- `q_edge`
- `frustration_node_sym_mean`, `frustration_node_sym_std`, `frustration_node_sym_median`
- `frustration_node_pol_mean`, `frustration_node_pol_std`, `frustration_node_pol_median`
- `frustration_edge_sym_mean`, `frustration_edge_sym_std`, `frustration_edge_sym_median`
- `frustration_edge_pol_mean`, `frustration_edge_pol_std`, `frustration_edge_pol_median`
- `D_residue`, `D_edge`
- `top_edge_indices`

Per-sample NPZ contains framewise arrays:

- `frustration_node_sym_framewise`
- `frustration_node_pol_framewise`
- `frustration_edge_sym_framewise`
- `frustration_edge_pol_framewise`
- `global_node_sym_framewise`
- `global_node_pol_framewise`
- `global_edge_sym_framewise`
- `global_edge_pol_framewise`

## Parameters

- `model A`, `model B`: endpoint Potts models to compare
- `sample_ids`: trajectories to analyze
- `md_label_mode`
  - `assigned`: use assigned residue labels
  - `halo`: use halo labels when present
- `keep_invalid`
  - off by default
  - when off, invalid frames are removed before commitment/frustration are computed
- `top_k_edges`
  - number of highest-`|ΔJ|` edges kept for per-edge frustration and edge-weighted residue blending
- `workers`
  - optional sample-level parallelism
  - `0` or omitted means automatic worker count
  - each worker processes one selected sample at a time
- `progress`
  - CLI and `phase_console` can show a sample-level progress bar
  - web jobs expose the same progress through the job status bar

## Web page

The endpoint page shows four bar plots for the selected sample:

- residue commitment
- edge commitment
- residue frustration
- edge frustration

It also shows a delta-energy distribution panel for all samples stored in the selected analysis.

Frustration can be shown either as:

- raw normalized values
- centered values relative to selected reference MD trajectories already included in the analysis

Color scale used in the web page:

- symmetric (raw): green = low frustration, red = high frustration
- symmetric (centered): green = lower than reference MD, red = higher than reference MD
- polarity: red = more A-like, blue = more B-like

Edge-weighted residue coloring:

- blends residue values with incident edge values
- weighted by `|ΔJ|`
- default weight: `0.75`
- default state on webserver: disabled
- UI applies a post-blend contrast normalization to avoid flat/blunted colors when edge blending is enabled

## 3D page

The 3D viewer can color residues by:

- commitment
- frustration (symmetric or polarity)

It uses the same edge-weighted residue blending toggle as the 2D page.

## Independent Delta-Energy Outputs

Standalone delta-energy analyses are stored under:

- `clusters/<cluster_id>/analyses/delta_energy/<analysis_id>/analysis.npz`

The NPZ contains:

- `sample_ids`, `sample_labels`, `sample_types`
- `sample_available_frame_counts`, `sample_frame_counts`, `sample_frame_limits`
- `delta_energy_bins`, `delta_energy_hist`
- `delta_energy_mean`, `delta_energy_std`, `delta_energy_median`, `delta_energy_min`, `delta_energy_max`

`sample_frame_limits` is `0` when all available frames were used; otherwise it records the requested random subset size.
