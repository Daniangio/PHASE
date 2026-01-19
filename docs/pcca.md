# PCCA+ coarse graining

PCCA+ (Perron Cluster Cluster Analysis) is a method for grouping MSM
microstates into metastable macrostates using the dominant eigenvectors of the
transition matrix.

## Core idea
- Start from an MSM estimated at lag time tau.
- Compute dominant eigenvectors of the transition matrix.
- Solve for a membership matrix that assigns microstates to metastable sets.

## How it is used in the code base
`vamp_pipeline.py` implements a PCCA+ path (`msm.pcca`) for the CLI workflow.
The webserver metastable pipeline currently uses a TICA + k-means approach for
macrostate assignment.

## Key parameters
- Number of macrostates k: chosen by a spectral-gap criterion in the MSM path.
- Membership thresholding: optional post-processing can mark low-membership
  microstates as outliers.

## References
- [Robust Perron cluster analysis in conformation dynamics](https://doi.org/10.1016/j.laa.2004.10.026)

## Related docs
- [Metastable state overview](doc:metastable_states)
- [VAMP/TICA details](doc:vamp_tica)
- [Markov State Models (MSM)](doc:msm)
