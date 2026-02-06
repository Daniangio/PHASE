# Delta Potts Evaluation: Analyses & Goals

This page exists to answer: **what do two delta-Potts fits actually imply in terms of likelihood / behavior**, not just raw parameter size.

All analyses are derived from comparing two models (A and B) on discrete trajectories `s_t`.

## Analysis 0: Gauge Fixing (Pre-step)

Goal: ensure that comparing `Δh` / `ΔJ` is meaningful.

What happens:

- Both models are converted to the same **zero-sum gauge** before any decomposition.
- This prevents “importance” caused purely by gauge artifacts.

## Analysis 1: Per-Frame Preference (ΔE)

Goal: quantify which model assigns lower energy (higher probability) to frames in an ensemble.

Computed:

- `ΔE(t) = E_A(s_t) − E_B(s_t)` for the selected MD sample.

Shown:

- histogram of `ΔE(t)`
- optional overlays: `ΔE(t)` on Potts-generated samples drawn from model A and/or model B

## Analysis 2: Localizing Contributions (Residues / Edges)

Goal: understand where the preference `ΔE` comes from.

Computed:

- per-residue `δ_i(t) = h^A_i(s_{t,i}) − h^B_i(s_{t,i})`
- per-edge `δ_ij(t) = J^A_{ij}(s_{t,i}, s_{t,j}) − J^B_{ij}(s_{t,i}, s_{t,j})`

Shown:

- mean `δ_i` bar plot
- mean `δ_ij` heatmap (on the model’s contact edges)

## Analysis 3: Transition-like (TS-band) Diagnostic (Validation Ladder 3)

Goal: test whether a third ensemble (Ensemble 3) concentrates near the “boundary” region between the two reference ensembles (Ensemble 1 and 2), using `ΔE` as a scalar coordinate.

Computed:

1. Reaction coordinate:
   - `z = (ΔE − median(train)) / MAD(train)` where `train = ensemble1 ∪ ensemble2`
2. TS-band around 0:
   - choose τ so that `P_train(|z| ≤ τ) = band_fraction` (default 10%)
3. Enrichment:
   - compare `P(|z| ≤ τ)` in ensemble 3 vs train
4. Term-level “partial commitment” (fields-only for now):
   - rank residues by training separation `D_i = E_1[δ_i] − E_2[δ_i]`
   - compute `q_i(X) = P(δ_i < 0 | X)` across ensembles and TS-band

Shown:

- overlay histogram of `z` for ensemble 1 / 2 / 3 (with ±τ)
- enrichment plot (`p_train` vs `p_3`) and log-enrichment
- heatmap of `q_i(X)` for top residues
- bar plot of `D_i` for top residues

Notes / limitations:

- This is an **operational** diagnostic, not a claim of a true physical transition state.
- Edge-level `D_ij` and module/network analyses can be added next (Validation Ladder 3, Analysis 2).

