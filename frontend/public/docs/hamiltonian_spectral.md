# Hamiltonian Spectral Analysis

This analysis identifies PHASE dynamic sectors from fitted Potts Hamiltonians.

It implements the non-ligand parts of the PHASE spectral-sector pipeline:

- Single-state spectra: intrinsic sectors of one thermodynamic state.
- Pair spectra: rewiring sectors between two states.

Ligand empirical-flow projection is intentionally not included yet.

## Inputs

- Cluster: the cluster workspace containing Potts models and residue keys.
- States: one or more system states selected by the user.
- State Potts model: PHASE resolves one full Hamiltonian per selected state from Potts model metadata.
- Eigenvectors to store: number of leading eigenvectors saved in the result NPZ.

The state-to-model resolver uses, in order, explicit model metadata such as `params.state_ids`, state paths in `params.pdbs`, and exact model name/path matches such as `model_active` for state `active`.

Delta-patch-only models are skipped because they are not complete Hamiltonians. Combined endpoint models are allowed when they are the available full Hamiltonian for a state.

## Single-State Analysis

For each selected state with a resolvable model:

1. The Potts model is converted to zero-sum gauge.
2. Each coupling block `J_ij(s_i, s_j)` is compressed to a scalar Frobenius norm:

```text
F[i,j] = sqrt(sum_{s_i,s_j} J_ij(s_i,s_j)^2)
```

3. The symmetric matrix `F` is diagonalized.
4. Eigenvalues and leading eigenvectors are saved.

Large absolute residue loadings `|v_i|` indicate residues strongly participating in that Hamiltonian sector.

## Pair Analysis

After single analyses exist, PHASE computes missing pair analyses involving at least one selected state:

```text
Delta F = F_B - F_A
```

`Delta F` is diagonalized by largest absolute eigenvalue. Positive and negative eigenvalues are both meaningful:

- Positive matrix entries indicate couplings stronger in state B than state A.
- Negative matrix entries indicate couplings weaker in state B than state A.
- Signed eigenvector loadings identify residues participating in the rewiring mode.

Pairs are incremental: rerunning the analysis with a new state adds that state's single analysis and missing pairs between that state and states already analyzed.

## Web Interpretation

The page has two modes:

- Single-state sectors: displays spectra for `F`.
- Pair rewiring sectors: displays spectra for `Delta F`.

Common panels:

- Eigenvalue spectrum: ranks sector components.
- Residue loading bars: shows the selected eigenvector component.
- Matrix heatmap: shows `F` or `Delta F` over residue pairs.
- Strongest edges: table of largest matrix entries by absolute magnitude.

For single-state plots, residue bars use `|v_i|` because eigenvector sign is arbitrary.

For pair plots, residue bars are signed. Red and blue separate opposite sides of the differential sector; the sign depends on the stored order `state B - state A`.

## Output Files

Single analyses are saved under:

```text
clusters/<cluster_id>/analyses/hamiltonian_spectral_single/<analysis_id>/
```

Pair analyses are saved under:

```text
clusters/<cluster_id>/analyses/hamiltonian_spectral_pair/<analysis_id>/
```

Each folder contains:

- `analysis_metadata.json`
- `analysis.npz`

Important NPZ arrays:

- `matrix`: `F` for single mode or `Delta F` for pair mode.
- `residue_keys`: residue labels aligned to matrix axes.
- `eigenvalues`: complete eigenspectrum.
- `top_eigenvalues`: saved leading eigenvalues.
- `top_eigenvectors`: shape `(top_k, n_residues)`.
- `residue_strength`: row-sum strength for quick ranking.

## Why A State Can Be Skipped

A skipped state does not mean that MD frames were not assigned to clusters. Cluster assignment and Hamiltonian spectra are separate steps.

This analysis requires a full fitted Potts Hamiltonian for each state. A state is skipped when PHASE cannot find one unambiguous full model associated with that state. Common cases:

- The state exists and has MD/sample labels, but no Potts model was fitted on that state.
- The state was added after an older fit, so its labels exist but no state-specific Hamiltonian exists.
- Multiple equally plausible models match the same state name and PHASE refuses to guess.
- Only a delta-patch model exists; patch-only models are not complete Hamiltonians and are skipped.

To include a skipped state, fit or create a full/combined Potts model for that state, then rerun the spectral analysis. The run is incremental, so existing singles and pairs are reused unless overwrite is enabled.

## 3D Coloring

The 3D page loads a reference PDB/state and colors the cartoon representation by eigenvector participation.

Selected eigenvector mode:

- Single-state spectra: green intensity is proportional to `|v_i|`.
- Pair spectra: red and blue are opposite signed sides of the selected rewiring eigenvector of `Delta F`.

All-vectors mode:

- The first eight saved eigenvectors are considered.
- Each residue is assigned to the eigenvector with the largest contribution `|lambda_k| * |v_ki|`.
- The eigenvector identity defines hue; contribution strength defines intensity.

Residue numbering:

- `PDB/auth residue id` uses numbers parsed from stored residue keys such as `res_193`.
- `Sequential label_seq_id` uses residue index + 1 and is useful when the PDB starts at 1 while PHASE residue keys use a different numbering.

## Differential Laplacian Allostery Mode

PHASE also stores a normalized allostery view for every pair analysis.

For a pair ordered as `state A -> state B`, the raw pair matrix is:

```text
Delta F = F_B - F_A
```

The signed normalized Laplacian is then computed with absolute degree:

```text
D_i = sum_j |Delta F[i,j]|
L[i,i] = 1                            if D_i > 0
L[i,j] = -Delta F[i,j] / sqrt(D_i D_j) if i != j and D_i,D_j > 0
L[i,j] = 0                             otherwise
```

This normalization reduces the dominance of intrinsically high-degree flexible residues and makes sparse allosteric rewiring communities easier to detect.

Interpretation:

- `Delta F spectral rewiring` is the raw signed coupling-change view.
- `Differential Laplacian allostery` is the normalized graph view.
- Laplacian eigenvalues are shown ascending.
- The stored Laplacian vectors are the smallest non-zero modes, i.e. Fiedler-like sectors.
- Red and blue in pair views are opposite signed sides of the selected sector.

If a pair analysis was created before Laplacian support, rerun Hamiltonian spectral analysis. Existing pair folders are upgraded automatically because PHASE now checks for the Laplacian arrays before skipping a pair.

Important stored arrays for pair analyses:

- `laplacian_matrix`: signed normalized Laplacian `L`.
- `laplacian_degree`: absolute degree vector `D`.
- `laplacian_eigenvalues`: full ascending Laplacian spectrum.
- `laplacian_top_eigenvalues`: selected smallest non-zero eigenvalues.
- `laplacian_top_eigenvectors`: selected Fiedler-like eigenvectors, shape `(top_k, n_residues)`.
- `laplacian_top_indices`: indices of selected modes in the full ascending spectrum.
