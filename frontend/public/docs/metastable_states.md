# Metastable state discovery (webserver pipeline)

This page describes the metastable-state pipeline used by the webserver and how
the hyperparameters map to the code in `phase/analysis/vamp_pipeline.py`.

## Pipeline overview
1) Load per-residue descriptor NPZ files produced during descriptor building.
2) Flatten all residue features into a single feature matrix per macro-state.
3) Standardize features and apply TICA to extract slow coordinates.
4) Cluster frames into microstates with k-means in TICA space.
5) Cluster microstate centers into metastable states and pick k by silhouette.
6) Save per-frame labels and representative structures for each metastable state.

## Hyperparameters and their roles
- TICA lag (frames): `tica_lag_frames` sets the lag for the time-lagged covariance.
- TICA dims: `tica_dim` selects how many slow components are retained.
- Microstates (k-means): `n_microstates` controls the number of micro clusters.
- Metastable min/max k: `k_meta_min` and `k_meta_max` define the k range tested
  for the second clustering stage (silhouette score).
- Random seed: `random_state` controls k-means initialization.

## Notes on the descriptors
- The pipeline consumes the same per-residue dihedral features used elsewhere in
  PHASE (phi/psi/chi1 angles transformed to sin/cos).
- Features are stacked in residue order to form a single frame vector.

## Outputs
- Per-frame metastable labels stored alongside each descriptor NPZ.
- Representative PDBs for each metastable state when topology/trajectory files exist.
- Saved TICA and k-means models for each macro-state.

## Related docs
- [VAMP/TICA details and references](doc:vamp_tica)
- [Markov State Models (MSM)](doc:msm)
- [PCCA+ coarse graining](doc:pcca)
