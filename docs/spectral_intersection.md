# Spectral Set-Intersection Analysis

This analysis identifies **allosteric pistons** by intersecting two categorical community assignments already computed by Hamiltonian spectral analysis.

It combines:

- A **single-state Laplacian** analysis, which assigns each residue to a structural community `Cstruct`.
- A **pair differential Laplacian** analysis, which assigns each residue to a functional community `Cfunc` from `|Delta F|`.

The analysis is intentionally set-theoretic. It does not recompute spectra and it does not compare every residue pair.

## Purpose

Single-state communities describe rigid structural modules inside one thermodynamic state. Pair communities describe functional rewiring pathways between two states.

Their intersection asks:

```text
Which residues are in the same structural module and also rewire together in the same functional pathway?
```

Those shared modules are reported as candidate **allosteric pistons**: cohesive mechanical units that move or rewire collectively during activation.

## Inputs

- Cluster: the cluster workspace containing spectral analyses.
- Structural single-state analysis: a `hamiltonian_spectral_single` result with v3 `community_ids`.
- Functional pair analysis: a `hamiltonian_spectral_pair` result with v3 `community_ids`.
- Minimum piston group size: minimum number of residues required for a composite group to be called an allosteric piston. Default is `3`.
- Overwrite: recompute an existing matching intersection artifact.

If the selected spectral analyses lack `community_ids`, rerun Hamiltonian spectral analysis first so the v3 Laplacian/DADApy fields are produced.

## Algorithm

For each residue `i`, PHASE reads two integer labels:

```text
Cstruct(i) = structural community from the single-state Laplacian
Cfunc(i)   = functional community from the pair differential Laplacian
```

Residues are grouped by the composite key:

```text
(Cstruct(i), Cfunc(i))
```

This is an `O(N)` hashing operation:

```text
for residue i:
    key = (Cstruct[i], Cfunc[i])
    groups[key].append(i)
```

A group is called an allosteric piston when:

```text
group size >= minimum piston group size
```

## Residue Roles

PHASE assigns a role code to each residue for visualization.

### Allosteric Piston

A composite `(Cstruct, Cfunc)` group with at least the minimum number of residues.

Interpretation: a cohesive structural unit that rewires its correlations collectively. These residues form a mechanical gear that shifts during activation.

### Structural Scaffold

A residue in a sufficiently large structural community but not in a sufficiently large composite functional intersection.

Interpretation: passive architectural support. These residues form a rigid block, but their internal coupling does not significantly change as a coherent piston.

### Transient Switch

A residue that is not part of a sufficiently large structural community but belongs to a sufficiently large functional community.

Interpretation: an isolated or weakly cohesive residue that can bridge independent structural domains when correlations spike during activation.

### Other

Residues that do not pass the current role heuristics.

## Web Interpretation

The 2D page shows:

- Residue role counts.
- A composite heatmap of structural community versus functional community, with cell intensity equal to residue count.
- A table of all allosteric piston groups and their residue lists.

The 3D page shows:

- A monochrome base protein.
- Allosteric piston residues highlighted with categorical colors.
- A side panel for isolating one piston at a time.

Colors are categorical labels. They do not encode piston strength or eigenvector magnitude.

## Output Files

Intersections are saved under:

```text
clusters/<cluster_id>/analyses/hamiltonian_spectral_intersection/<analysis_id>/
```

Each folder contains:

- `analysis_metadata.json`
- `analysis.npz`

Important NPZ arrays:

- `residue_keys`: residue labels aligned to all arrays.
- `structural_community_ids`: `Cstruct` per residue.
- `functional_community_ids`: `Cfunc` per residue.
- `piston_ids`: piston ID per residue, or `0` if not in a piston.
- `residue_class_codes`: `0=other`, `1=structural scaffold`, `2=transient switch`, `3=allosteric piston`.
- `class_counts`: role-code counts.
- `piston_group_ids`: IDs of detected pistons.
- `piston_structural_community_ids`: structural community component of each piston key.
- `piston_functional_community_ids`: functional community component of each piston key.
- `piston_sizes`: residue count per piston.
- `composite_structural_community_ids`, `composite_functional_community_ids`, `composite_group_sizes`: all composite groups, including those below the piston threshold.
- `piston_members_json`: residue indices and labels for each piston group.
