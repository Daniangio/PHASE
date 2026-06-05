# Spectral Set-Intersection And Piston Ligand Projection

This page implements the v5 core-halo allosteric piston model and the v5.1 masked ligand projection.

## Purpose

Hamiltonian spectral analysis produces two kinds of DADApy communities:

- **Structural communities** from a single-state Laplacian built from `F_single`.
- **Functional communities** from a pair differential Laplacian built from `|Delta F|`.

DADApy also separates each community into **core** and **halo** points. Core points are density-supported residues inside a topological basin. Halo points are background/noise residues near density saddles and are not allowed to define allosteric pistons.

The set-intersection asks:

```text
Which residues are core in a structural community and core in a functional community with the same composite key?
```

This filters the thermodynamic bulk before interpreting mechanical allostery.

## Inputs

- Cluster: the cluster workspace containing spectral analyses.
- Structural single-state analysis: a `hamiltonian_spectral_single` result with v5 `community_core_mask`.
- Functional pair analysis: a `hamiltonian_spectral_pair` result with v5 `community_core_mask`.
- Minimum piston group size: minimum number of core-core residues required for an allosteric piston. Default is `3`.
- Overwrite: recompute an existing matching intersection artifact.

If the selected spectral analyses lack `community_core_mask`, rerun Hamiltonian spectral analysis first so the v5 DADApy core-halo fields are produced.

## Core-Strict Piston Algorithm

For every residue `i`, PHASE reads:

```text
Cstruct(i), IsCore_struct(i)
Cfunc(i),   IsCore_func(i)
```

The piston grouping is an `O(N)` hashing operation over core-core residues only:

```text
for residue i:
    if IsCore_struct[i] and IsCore_func[i]:
        key = (Cstruct[i], Cfunc[i])
        groups[key].append(i)
```

A group is called an allosteric piston only when:

```text
group size >= minimum piston group size
```

Halo points are never included in piston groups.

## Residue Roles

### Allosteric Piston

Signature: `Core_struct AND Core_func`, in a composite group passing the size threshold.

Interpretation: highly cohesive structural units that rewire correlations collectively. These are candidate mechanical gears.

### Structural Scaffold

Signature: `Core_struct AND NOT Core_func`.

Interpretation: rigid architectural blocks with stable internal coupling, but thermodynamically deaf to the activation signal.

### Transient Switch

Signature: `NOT Core_struct AND Core_func`.

Interpretation: isolated residues lacking local cohesion whose dynamic correlations spike transiently to bridge the allosteric network.

### Thermodynamic Bulk

Signature: `NOT Core_struct AND NOT Core_func`.

Interpretation: background thermal bath. These residues are hidden or left visually quiet in piston-focused 3D views.

### Subthreshold Core Overlap

Signature: `Core_struct AND Core_func`, but the composite group is smaller than the selected minimum size.

Interpretation: a small core-core overlap that is not treated as a macroscopic piston under the current threshold.

## Ligand / Short-MD Projection

The v5.1 ligand projection replaces global `v^T M v` scoring with masked piston-specific scores.

For each selected ligand or short-MD sample, PHASE computes an empirical mutual information matrix `M_short` from its assigned residue-cluster labels. For each piston `k` with residue set `Omega_k`, PHASE chooses the functional Laplacian eigenvector with the strongest loading on that piston and masks it:

```text
v_tilde_i = v_i if i in Omega_k
v_tilde_i = 0   otherwise
```

The score is:

```text
P_k = v_tilde^T M_short v_tilde
```

The resulting bar chart shows the ligand's bias footprint across mechanical gears. Higher values mean the short trajectory engages that piston more strongly under this empirical MI projection.

## Web Interpretation

The 2D page shows:

- Residue role counts.
- A core-core composite heatmap of structural community versus functional community.
- A table of all allosteric piston groups and their residue lists.
- Optional ligand/short-MD projection grouped bars.

The 3D page shows:

- A monochrome base protein.
- Only allosteric piston residues highlighted with categorical colors.
- A side panel for isolating one piston at a time.

Colors are categorical labels. They do not encode piston strength or eigenvector magnitude.

## Output Files

Intersections are saved under:

```text
clusters/<cluster_id>/analyses/hamiltonian_spectral_intersection/<analysis_id>/
```

Ligand projections are saved under:

```text
clusters/<cluster_id>/analyses/piston_ligand_projection/<analysis_id>/
```

Important intersection NPZ arrays:

- `residue_keys`: residue labels aligned to all arrays.
- `structural_community_ids`: assigned structural community per residue.
- `functional_community_ids`: assigned functional community per residue.
- `structural_core_mask`: DADApy structural core flag per residue.
- `functional_core_mask`: DADApy functional core flag per residue.
- `piston_ids`: piston ID per residue, or `0` if not in a piston.
- `residue_class_codes`: `0=thermodynamic bulk`, `1=structural scaffold`, `2=transient switch`, `3=allosteric piston`, `4=subthreshold core overlap`.
- `class_counts`: role-code counts.
- `piston_group_ids`: IDs of detected pistons.
- `piston_structural_community_ids`: structural community component of each piston key.
- `piston_functional_community_ids`: functional community component of each piston key.
- `piston_sizes`: residue count per piston.
- `composite_structural_community_ids`, `composite_functional_community_ids`, `composite_group_sizes`: core-core composite groups, including those below threshold.
- `piston_members_json`: residue indices and labels for each piston group.

Important ligand projection NPZ arrays:

- `sample_ids`, `sample_names`, `sample_types`: projected samples.
- `frames_used`: number of frames used per sample.
- `piston_group_ids`, `piston_sizes`: piston axes.
- `piston_vector_indices`: functional Fiedler vector selected for each piston.
- `piston_vector_norms`: loading norm of the selected vector on each piston.
- `piston_scores`: shape `(n_samples, n_pistons)`, the masked projection scores.
