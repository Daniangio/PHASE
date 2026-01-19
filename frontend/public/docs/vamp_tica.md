# VAMP and TICA in AllosKin

This page explains how VAMP/TICA are used in the code base and how they connect
to the metastable pipeline.

## What VAMP/TICA compute
- TICA finds linear combinations of descriptors whose time-lagged correlations
  decay the slowest. These slow coordinates approximate long-time kinetics.
- VAMP (Variational Approach for Markov Processes) generalizes TICA by
  optimizing a variational score (often VAMP-2) for the Koopman operator.

## Where it appears in the code
- Webserver metastable pipeline uses `deeptime.decomposition.TICA` in
  `run_metastable_pipeline_for_macro`.
- Inputs are standardized with `StandardScaler` before projection.
- The TICA-projected data is used for k-means clustering.
- CLI VAMP path uses `run_vamp` in `vamp_pipeline.py` with
  `VAMP(lagtime=lag_frames, dim=dim, scaling="kinetic_map")`.

## Hyperparameters
- `tica_lag_frames` sets the lag used to build time-lagged covariances.
- `tica_dim` sets the number of slow components kept after projection.
- In the VAMP path, `dim` can be an integer or a kinetic-variance threshold.

## Why it matters for metastable discovery
The slow coordinates separate long-lived conformational basins. Clustering in
this reduced space is more robust than clustering in the raw descriptor space.

## References
- https://www.nature.com/articles/s41467-017-02388-1
- https://helper.ipam.ucla.edu/publications/mpsws2/mpsws2_13549.pdf#page=30.00
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11282584/

## Related docs
- [Metastable state overview](doc:metastable_states)
- [Markov State Models (MSM)](doc:msm)
- [PCCA+ coarse graining](doc:pcca)
