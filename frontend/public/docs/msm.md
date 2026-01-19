# Markov State Models (MSM)

An MSM is a kinetic model of a system where the dynamics are described as a
Markov chain between discrete states at a chosen lag time.

## Core idea
- Discretize trajectories into states (microstates).
- Count transitions at lag time tau and estimate a transition matrix T(tau).
- The eigenvalues and eigenvectors of T(tau) encode relaxation timescales and
  slow dynamical processes.

## How it is used in the code base
`vamp_pipeline.py` includes an MSM path built with
`deeptime.markov.msm.MaximumLikelihoodMSM`. This path is currently used by the
CLI utilities, while the webserver metastable pipeline relies on TICA + k-means
silhouette clustering.

## Key parameters
- Lag time (frames): the discretization interval for counting transitions.
- Reversibility: enforces detailed balance when estimating the transition
  matrix.

## References
- (Add MSM reference links here)

## Related docs
- [Metastable state overview](doc:metastable_states)
- [VAMP/TICA details](doc:vamp_tica)
- [PCCA+ coarse graining](doc:pcca)
