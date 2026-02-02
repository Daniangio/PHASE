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
- `plm-device`: device for PLM training (`auto`, `cpu`, `cuda`, or any torch device string).
- `plm-init`: PLM initialization (`pmi`, `zero`, or `model`).
- `plm-init-model`: path to a Potts model NPZ used when `plm-init=model`.
- `plm-resume-model`: resume PLM from a saved model (keeps stored best loss if available).
- `plm-val-frac`: fraction of frames reserved for validation during PLM.
- `unassigned-policy`: how to handle -1 labels in the input.

## Fit-only mode
Use `--fit-only` to save a model without running sampling. The model is written to
`potts_model.npz` in the results directory (or `--model-out`).

Use `--model-npz` in a later run to reuse a pre-fit model and skip fitting. You can repeat
`--model-npz` (or pass a comma-separated list) to combine multiple models.

## Notes
- PLM requires PyTorch; GPU is used automatically when available.
- Use `--plm-device cuda` (or `cpu`) to override device selection.
- PMI is useful as a quick baseline or as initialization for PLM.
- PLM saves the best-loss checkpoint as it improves, so an interrupted run still leaves a usable model.

## Local fitting with the uv helper
To fit on a separate machine, run the setup once, then activate the venv and use the
interactive fitter:

```bash
./scripts/potts_setup.sh
source .venv-potts-fit/bin/activate
./scripts/potts_fit.sh
```

## Related docs
- [Potts analysis overview](doc:potts_overview)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [SA/QUBO sampling](doc:potts_sa_qubo)
- [PMI and PLM basics](doc:potts_pmi_plm)
