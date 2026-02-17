# Quantum Annealing Benchmark Protocol for Potts Sampling

## Scope

This document defines a benchmark to compare Quantum Annealing (QA) against classical samplers for the PHASE Potts workflow.

Assumed fitted models:

- `M_base`: Potts model fitted on pooled `Active + Inactive` data.
- `M_A_delta`: delta model fitted on Active (relative to `M_base`).
- `M_I_delta`: delta model fitted on Inactive (relative to `M_base`).
- Optional combined endpoint models:
  - `M_A = M_base + M_A_delta`
  - `M_I = M_base + M_I_delta`

## Primary Question

Is QA better than classical methods at generating useful samples from these Potts models under matched compute budget?

## Candidate Samplers to Compare

Minimum set:

- Gibbs sampling (single-chain and multi-chain).
- Simulated Annealing (SA).
- Parallel Tempering / Replica Exchange (PT/REX) if available.
- Hybrid SA + local search (if implemented).
- Quantum Annealing (hardware or simulator).

For QA, report hardware details (topology, embedding overhead, retries, chain strength, anneal schedule).

## What "Better" Means

Do not use one metric only. Use a multi-objective scorecard:

- **Optimization quality**: reaches lower energies.
- **Distribution fidelity**: reproduces target ensemble statistics where expected.
- **Exploration quality**: avoids collapse to near-duplicates.
- **Biophysical usefulness**: enriches functionally informative states/features.
- **Efficiency**: quality per unit wall-clock and per unit compute cost.

## Benchmark Design

## 1) Controlled sampling tasks

Run all samplers on each target model:

- Task T1: sample from `M_base`.
- Task T2: sample from `M_A`.
- Task T3: sample from `M_I`.
- Task T4 (optional): lambda sweep `M(lambda) = (1-lambda) M_I + lambda M_A`, lambda grid in `[0, 1]`.

Use matched budgets:

- Same number of returned samples.
- Same number of independent chains.
- Same initialization policy.
- Same max objective evaluations (or as close as possible).
- Also report wall-clock and energy evaluations separately.

Run multiple seeds (at least 5, ideally 10).

## 2) Train/validation split discipline

Keep a clean split for evaluation:

- Fit models on training MD frames only.
- Evaluate sampling quality against held-out MD frames.
- If comparing Active vs Inactive discrimination, evaluate on held-out frames from both.

This prevents over-claiming due to memorizing training marginals.

## 3) Metrics to report

### A. Energy metrics (per model)

- Best energy reached.
- Mean energy and standard deviation.
- Quantiles (1%, 5%, 50%).
- Fraction below fixed energy thresholds.
- Energy CDF overlap with reference MD-evaluated labels.

### B. Distribution metrics (node and edge)

- Per-residue JS divergence to reference MD (`p_sample` vs `p_md`).
- Per-edge JS divergence on selected edges (or all edges where feasible).
- Aggregate stats: median JS, top-k worst residues/edges.

### C. Diversity / degeneracy metrics

- Unique configuration rate.
- Mean pairwise Hamming distance.
- Effective sample size proxies from autocorrelation.
- Cluster occupancy entropy over sampled cluster labels.

### D. Delta-model diagnostics

For pairs `M_A` vs `M_I`:

- Commitment distributions per residue on each sample set.
- Discriminative residue ranking stability across seeds.
- Separation score between Active-MD and Inactive-MD under the same centering mode.

### E. Efficiency metrics

- Time-to-energy-threshold.
- Quality-at-fixed-time.
- Quality-at-fixed-evaluations.
- Optional cost-normalized metric (cloud/hardware cost).

## 4) Statistical analysis

- Report mean, std, and 95% bootstrap CI across seeds.
- Use paired comparisons per seed/budget.
- Avoid only "best seed" plots.

## Suggested acceptance criteria

QA should beat at least one strong classical baseline on:

- lower median energy at fixed budget, and
- equal or better JS/diversity (no severe mode collapse).

If QA only improves minimum energy but worsens diversity strongly, mark as mixed outcome.

## Experimental-data conditioning: does it make sense?

Yes, but use it as **soft constraints**, not hard replacement of the Potts objective.

Recommended form:

- Sample from modified objective  
  `E_total(x) = E_potts(x) + sum_k w_k * E_exp_k(x)`

where `E_exp_k` encodes agreement with an experiment (distance, contact, protection factor, etc.) and `w_k` is calibrated.

Good experimental signals to integrate:

- NMR restraints (NOE, PRE, RDC-derived tendencies).
- DEER/FRET distance distributions.
- HDX protection trends (coarse-grained structural proxies).
- Cross-linking MS contact restraints.
- Mutational scan constraints (state preference tendencies).

## Important risk you raised (MD vs experiment mismatch)

This is real. If experiments and MD live in different ensemble regimes:

- Potts alone may extrapolate poorly.
- Hard experimental constraints can force unrealistic states if inconsistent.

Mitigations:

- Treat experimental terms as soft penalties with uncertainty-aware weights.
- Calibrate weights on a validation subset of observables.
- Keep an unconstrained Potts baseline for comparison.
- Track tradeoff curves: Potts energy vs experimental agreement.

## Strong use-cases for Potts sampling in this project

- Fast generation of discrete-state ensembles consistent with learned couplings.
- Rare-state enrichment when MD is expensive.
- Screening hypotheses before expensive all-atom refinement.
- Comparing biased-MD-derived models after reweighting.

Practical strategy:

- Fit from reweighted biased MD.
- Sample from learned unbiased Potts objective.
- Validate against held-out unbiased observables and independent experimental readouts.

## Minimal benchmark package to save per run

- Sampler config (all hyperparameters, seed, budget, hardware info).
- Output sample NPZ (labels/states only, per current lightweight format).
- Analysis NPZ for metrics above.
- Metadata JSON with references to model id, cluster id, and analysis version.

## Recommended first benchmark matrix

- Models: `M_base`, `M_A`, `M_I`.
- Samplers: Gibbs, SA, PT/REX, QA.
- Seeds: 10.
- Budget tiers: short, medium, long.
- Outputs: energy tables, JS tables, diversity stats, commitment comparisons.

This matrix is usually enough to decide if QA is promising before deeper hardware-specific tuning.

