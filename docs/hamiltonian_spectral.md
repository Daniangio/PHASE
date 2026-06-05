# Hamiltonian Spectral Analysis

This analysis identifies PHASE dynamic sectors from fitted Potts Hamiltonians.

It implements the non-ligand spectral-sector pipeline:

- Single-state Frobenius spectra: intrinsic sectors of one thermodynamic state.
- Single-state Laplacian communities: rigid structural modules inside one state.
- Pair DeltaF spectra: signed rewiring sectors between two states.
- Pair differential Laplacian communities: normalized allosteric pathways between two states.

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

3. The symmetric matrix `F` is diagonalized for the absolute Frobenius spectrum.
4. A normalized graph Laplacian is computed from `A = F`.
5. The smallest non-zero Laplacian modes are used as a spectral embedding.
6. Rows of the embedding are normalized onto the unit hypersphere.
7. DADApy density peak clustering is run on cosine distances in that embedding to assign residue communities.

Interpretation:

- Large absolute Frobenius loadings `|v_i|` indicate residues strongly participating in a Hamiltonian sector.
- Single-state Laplacian communities suppress hyper-flexible high-degree loops and identify rigid coupled architectural modules of that state.

## Pair Analysis

After single analyses exist, PHASE computes missing pair analyses involving at least one selected state:

```text
Delta F = F_B - F_A
```

`Delta F` is diagonalized by largest absolute eigenvalue. Positive and negative eigenvalues are both meaningful:

- Positive matrix entries indicate couplings stronger in state B than state A.
- Negative matrix entries indicate couplings weaker in state B than state A.
- Signed eigenvector loadings identify residues participating in the rewiring mode.

For differential Laplacian allostery, PHASE uses the non-negative adjacency:

```text
A = |Delta F|
```

This represents total coupling rewiring magnitude. The normalized Laplacian and DADApy community detection then identify functional pathways between states.

Pairs are incremental: rerunning the analysis with a new state adds that state's single analysis and missing pairs between that state and states already analyzed.

## Normalized Laplacian

For single-state communities, `A = F`. For pair communities, `A = |Delta F|`.

```text
D_i = sum_j A[i,j]
L[i,i] = 1.0                    if D_i > 0
L[i,j] = -A[i,j] / sqrt(D_i D_j) if i != j and D_i,D_j > 0
L[i,j] = 0.0                     otherwise
```

The complete Laplacian spectrum is sorted ascending. The stored `laplacian_top_*` vectors are the smallest non-zero modes. The embedding dimension is chosen by a simple eigengap rule over those non-zero modes.

## DADApy Communities

PHASE builds a spectral embedding from the selected non-zero Laplacian modes, row-normalizes it to the unit hypersphere, and computes cosine distances:

```text
distance(i,j) = 1 - dot(Y_i, Y_j)
```

DADApy density peak clustering is then applied to those distances. Cosine distance is used because the embedding lives on a sphere; treating rows as ordinary Euclidean coordinates can distort angular sectors.

Stored community IDs are integer labels. They are categorical, not ordered continuous values.

## Web Interpretation

The page has two modes:

- Single-state sectors: displays either `F` spectra or single-state Laplacian communities.
- Pair rewiring sectors: displays either signed `Delta F` spectra or differential Laplacian communities.

Common panels:

- Eigenvalue spectrum: ranks sector components.
- Residue loading bars: shows the selected eigenvector component.
- Matrix heatmap: shows `F`, `Delta F`, or the Laplacian source matrix reordered by community when community mode is active.
- Strongest edges: table of largest displayed matrix entries by absolute magnitude.
- Community summary: community sizes and inter-community coupling sums when Laplacian communities are available.

For single-state Frobenius plots, residue bars use `|v_i|` because eigenvector sign is arbitrary.

For pair `Delta F` plots, residue bars are signed. Red and blue separate opposite sides of the differential sector; the sign depends on the stored order `state B - state A`.

For community views, colors are categorical. Do not interpret community color intensity as a scalar score.

## 3D Coloring

The 3D page loads a reference PDB/state and colors the cartoon representation.

Color modes:

- Selected eigenvector: colors by the selected component loading.
- Dominant among first 8 vectors: each residue is colored by its strongest component.
- Communities: each residue is colored by its DADApy community ID using a categorical palette.

Single-state Laplacian communities map structural modules in one state. Pair differential Laplacian communities map functional allosteric pathways between two states.

Residue numbering:

- `PDB/auth residue id` uses numbers parsed from stored residue keys such as `res_193`.
- `Sequential label_seq_id` uses residue index + 1 and is useful when the PDB starts at 1 while PHASE residue keys use a different numbering.

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

- `matrix`: `F` for single mode or signed `Delta F` for pair mode.
- `residue_keys`: residue labels aligned to matrix axes.
- `eigenvalues`: complete absolute/Frobenius eigenspectrum.
- `top_eigenvalues`: saved leading absolute/Frobenius eigenvalues.
- `top_eigenvectors`: shape `(top_k, n_residues)`.
- `residue_strength`: row-sum strength for quick ranking.
- `laplacian_source_matrix`: `F` for single community mode or `|Delta F|` for pair community mode.
- `laplacian_matrix`: normalized Laplacian `L`.
- `laplacian_degree`: degree vector `D`.
- `laplacian_eigenvalues`: full ascending Laplacian spectrum.
- `laplacian_top_eigenvalues`: selected smallest non-zero eigenvalues.
- `laplacian_top_eigenvectors`: selected Fiedler-like eigenvectors, shape `(top_k, n_residues)`.
- `laplacian_top_indices`: indices of selected modes in the full ascending spectrum.
- `laplacian_embedding`: row-normalized spectral embedding used for clustering.
- `laplacian_embedding_indices`: eigenvector indices used in that embedding.
- `community_ids`: DADApy community ID for each residue.
- `community_halo_ids`: DADApy halo assignment for each residue.
- `community_sizes`: two-column array `(community_id, n_residues)`.
- `community_matrix_order`: residue order that groups the matrix by community.
- `community_interaction_matrix`: coarse-grained inter-community coupling sums.
- `community_diagnostics_json`: clustering method, distance metric, DADApy parameters, and fallback warnings if any.

## Why A State Can Be Skipped

A skipped state does not mean that MD frames were not assigned to clusters. Cluster assignment and Hamiltonian spectra are separate steps.

This analysis requires a full fitted Potts Hamiltonian for each state. A state is skipped when PHASE cannot find one unambiguous full model associated with that state. Common cases:

- The state exists and has MD/sample labels, but no Potts model was fitted on that state.
- The state was added after an older fit, so its labels exist but no state-specific Hamiltonian exists.
- Multiple equally plausible models match the same state name and PHASE refuses to guess.
- Only a delta-patch model exists; patch-only models are not complete Hamiltonians and are skipped.

To include a skipped state, fit or create a full/combined Potts model for that state, then rerun the spectral analysis. The run is incremental, so existing singles and pairs are reused unless overwrite is enabled.
