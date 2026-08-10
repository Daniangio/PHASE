# Potts model fitting

This page describes how PHASE fits standard Potts models and sparse delta-Potts models.

## What is being fit

For residue microstate labels `x`, PHASE uses the pairwise Potts energy

`E(x) = sum_r h_r[x_r] + sum_(r,s in edges) J_rs[x_r, x_s]`

where:
- `h_r` is the field for residue `r`
- `J_rs` is the coupling block for contact edge `(r,s)`
- the edge set stays sparse: only the selected contact graph is fit
- each residue can have its own alphabet size `K_r`

Sampling then uses `p_beta(x) ∝ exp(-beta * E(x))`.

## Fitting workflow

### Standard Potts fit

`fit = pmi | plm | pmi+plm`

- `pmi`: fast co-occurrence heuristic. Good as a baseline or initializer.
- `plm`: optimize negative pseudolikelihood directly with PyTorch.
- `pmi+plm`: build a PMI initializer first, then refine with PLM. This is the recommended default.

PLM works as follows:
1. Start from either PMI, zeros, or a supplied model.
2. For each minibatch and each residue `r`, compute conditional logits
   `h_r(k) + sum_s J_rs(k, x_s)`.
3. Sum cross-entropy over residues. This is the negative pseudolikelihood objective.
4. Add regularization.
5. Update raw trainable tensors with Adam.

Important implementation detail:
- when `zero_sum_gauge=True`, PHASE does **not** mutate the optimizer-owned parameters in place
- instead, each forward pass builds differentiable zero-sum-gauge views of `h` and `J`
- the saved model is exported from those projected views, so saved fields/couplings stay interpretable

### Delta Potts fit

The delta fit keeps a base model fixed and learns only `Δh` and `ΔJ` on top of it.

Effective logits use:
- `h_eff = h_base + Δh`
- `J_eff = J_base + ΔJ`

The delta penalties are applied to the **projected** delta tensors when gauge fixing is enabled. This matters scientifically: otherwise the size of `Δh` and `ΔJ` would depend on arbitrary gauge choice instead of reflecting minimal rewiring.

## Gauge handling

When zero-sum gauge is enabled:
- each field vector is centered so its mean is zero
- each coupling block is double-centered, so row sums and column sums are zero

PHASE now uses that gauge consistently in:
- the forward pass
- regularization
- exported checkpoints

This avoids the old failure mode where parameters were projected in place after `optimizer.step()`, which can desynchronize Adam's moment buffers from the actual parameter values.

## Standard fit parameters

- `plm-epochs`: number of PLM epochs.
- `plm-lr`: AdamW learning rate (default `1e-3`).
- `plm-lr-min`: minimum learning rate for cosine decay.
- `plm-lr-schedule`: `cosine` or `none`.
- `plm-l2`: L2 penalty weight on the effective PLM parameters.
- `plm-lambda`: group-Frobenius shrinkage on coupling blocks. This shrinks entire edge blocks; it is not an exact proximal group-lasso solver.
- `plm-batch-size`: micro-batch size.
- `plm-grad-accum-steps`: number of micro-batches to accumulate before each optimizer step. Effective batch size is `plm_batch_size * plm_grad_accum_steps`.
- `plm-progress-every`: epoch interval for progress callbacks.
- `plm-device`: `auto`, `cpu`, `cuda`, or any torch device string such as `cuda:0`.
- `plm-init`: `pmi`, `zero`, or `model`.
- `plm-init-model`: model used when `plm-init=model`.
- `plm-resume-model`: continue optimizing an existing model in place.
- `plm-val-frac`: optional validation fraction. When set, PHASE keeps the best validation checkpoint.

PHASE uses AdamW for both standard and delta Potts fitting with `betas=(0.9, 0.999)`, `eps=1e-8`, and decoupled `weight_decay=1e-2`. Explicit PLM/delta L2 and group penalties remain separate model regularizers.
- `unassigned-policy`: how `-1` labels are handled before fitting.

## Delta fit parameters

- `delta-epochs`: number of epochs.
- `delta-lr`: AdamW learning rate (default `1e-3`).
- `delta-lr-min`: minimum learning rate for cosine decay.
- `delta-lr-schedule`: `cosine` or `none`.
- `delta-batch-size`: micro-batch size.
- `delta-grad-accum-steps`: number of micro-batches to accumulate per optimizer update.
- `delta-seed`: random seed.
- `delta-device`: `auto`, `cpu`, `cuda`, or explicit torch device string.
- `delta-l2`: L2 penalty on the projected delta tensors.
- `delta-group-h`: group-L2 shrinkage on each residue field vector `Δh_r`.
- `delta-group-j`: group-Frobenius shrinkage on each edge block `ΔJ_rs`.
- `delta-no-combined`: save only the delta model, not the combined `base + delta` model.

## Checkpoint behavior

- Standard PLM writes the best checkpoint seen so far.
- If validation is enabled, "best" means lowest validation pseudolikelihood.
- Otherwise "best" means lowest training pseudolikelihood.
- Exported checkpoints are written in zero-sum gauge when gauge fixing is enabled.

## Notes on interpretation

- `plm-lambda`, `delta-group-h`, and `delta-group-j` are shrinkage penalties, not guaranteed exact sparsifiers.
- The saved `PottsModel` uses the project sign convention for energies. Internal logits-space tensors use the opposite sign during optimization.
- Sparse-edge semantics are preserved: PHASE does not silently densify all residue pairs during fitting.

## Fit-only mode

Use `--fit-only` to save a model without running sampling.

- Standard fit writes a Potts model NPZ.
- Delta fit writes delta NPZs and, unless disabled, combined `base + delta` NPZs.

## Local usage

```bash
./scripts/potts_setup.sh
source .venv-potts-fit/bin/activate
./scripts/potts_fit.sh
```

## Related docs

- [Potts analysis overview](doc:potts_overview)
- [PMI and PLM basics](doc:potts_pmi_plm)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [SA/QUBO sampling](doc:potts_sa_qubo)
