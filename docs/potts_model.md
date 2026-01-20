# Potts model fitting

This page explains the Potts model used in the pipeline and how it is fit.

## Model definition
We model the joint distribution over residue microstates x with a pairwise Potts energy:

E(x) = sum_r h_r[x_r] + sum_(r,s in edges) J_rs[x_r, x_s]

The probability used for sampling is p_beta(x) proportional to exp(-beta * E(x)).
The edge list comes from `contact_edge_index` in the NPZ file.

## Fitting methods
- PMI (fast heuristic): estimates h and J from single-site and pairwise co-occurrences.
- PLM (pseudolikelihood): optimizes negative pseudolikelihood with PyTorch.
  This is a symmetric fit that can start from the PMI initializer.

## Hyperparameters
- `plm-epochs`: training epochs for PLM.
- `plm-lr`, `plm-lr-min`, `plm-lr-schedule`: learning rate and optional cosine decay.
- `plm-l2`: L2 regularization on parameters.
- `plm-batch-size`: minibatch size for pseudolikelihood.
- `unassigned-policy`: how to handle -1 labels in the input.

## Notes
- PLM requires PyTorch; GPU is used automatically when available.
- PMI is useful as a quick baseline or as initialization for PLM.

## Related docs
- [Potts analysis overview](doc:potts_overview)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [SA/QUBO sampling](doc:potts_sa_qubo)
