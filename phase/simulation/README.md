# Reduced-Dimensional Potts + QUBO Pipeline (with Replica Exchange + β_eff calibration)

This project builds a **reduced conformational model** from MD by:
1) clustering each residue’s torsions (φ, ψ, χ1) into discrete **microstates**  
2) fitting a **Potts model** over the microstate labels  
3) sampling the model via **classical baselines** (Gibbs / replica exchange) and a **QA-proxy** route (Potts → QUBO → SA)  
4) comparing samples to MD statistics (marginals + pairwise edge joints)

The pipeline is designed to stay *scientifically diagnosable*: if results differ, you can tell whether the bottleneck is **representation**, **model fit**, or **sampler/encoding**.

---

## What’s in the updated bundle

The updated code adds two key capabilities:

### 1) Replica exchange (parallel tempering) for Gibbs
Single-chain Gibbs can get stuck in a metastable basin. Replica exchange runs several chains at different **β** (inverse temperature) and swaps configurations between adjacent β’s to improve mixing. You then read samples from your target β replica.

### 2) Effective temperature calibration for SA (β_eff)
SA/QUBO samples often behave like draws from a Boltzmann distribution at an **unknown effective temperature**. We estimate **β_eff** by scanning a β grid and finding which Gibbs reference samples best match the SA samples in reduced space.

---

## Requirements

- Python 3.10+ recommended
- `numpy`
- `neal` + `dimod` SA using D-Wave’s reference SA
- Optional (nice-to-have): `tqdm` for progress bars
- Optional: `torch` if you want **PLM (pseudolikelihood)** Potts fitting

---

## Input `.npz` schema

`main.py` expects (names must match):

- `merged__labels`: shape `(T, N)` integer labels per frame & residue  
- `merged__cluster_counts`: shape `(N,)` number of clusters per residue  
- `contact_edge_index`: shape `(2, E)` residue indices for edges (contact graph)  
- `residue_keys`: shape `(N,)` (optional, used for plot labels)

---

## How temperature (β) enters the pipeline

### Potts model sampling
The Potts model defines:
\[
p_\beta(x)\propto \exp(-\beta E(x)).
\]
In Gibbs sampling, **β multiplies the energy** inside the conditional probabilities. Higher β = colder (more low-energy), lower β = hotter.

Replica exchange runs multiple Gibbs chains at different β and swaps states.

### QUBO + SA sampling
There is a **temperature-like knobs**:
1) **β** passed into `potts_to_qubo_onehot(model, beta=...)`  
   - This scales the Potts energy terms when building the QUBO

Because SA depends on its schedule, it typically does **not** correspond to the intended Boltzmann temperature. That’s why we estimate **β_eff** empirically.

### Can we map β to MD temperature (300 K)?
Not cleanly, in general. β here is **dimensionless** and depends on how your Potts energy is scaled by fitting + representation. A safe interpretation is:
- treat **β = 1** as “the model’s native scale” (often the one that best matches MD statistics),
- treat other β’s as **tempering**: \(p_\beta(x)\propto p_{\text{MD}}(x)^\beta\) *if* your fitted energy approximates \(-\log p_{\text{MD}}\).

To truly map to Kelvin you’d need calibration (e.g., MD at multiple temperatures or another absolute energy reference).

---

## Running the pipeline

> NOTE: the updated `main.py` in the bundle uses package-style imports (`phase...`).  
> If your repo uses flat scripts, either:
> - place files under the matching package structure, or
> - adjust imports to your local layout.

### Basic command template

```bash
python main.py \
  --npz path/to/data.npz \
  --unassigned-policy drop_frames \
  --fit plm \
  --beta 1.0 \
  --gibbs-method single \
````

---

## CLI parameters (what they do)

### Data

* `--npz` : input npz file
* `--unassigned-policy` : handling of -1 labels

  * `drop_frames` (default): drop frames containing any -1 (cleanest)
  * `treat_as_state`: map -1 to an extra “transition” state per residue
  * `error`: fail if -1 exists

### Potts fit

* `--fit` : `pmi` | `plm` | `pmi+plm`

  * `pmi`: fast heuristic from co-occurrences (debug / init)
  * `plm`: pseudolikelihood fit (requires torch)
  * `pmi+plm`: fast heuristic from co-occurrences as initial guess + pseudolikelihood fit (recommended baseline; requires torch)
* `--beta`: target inverse temperature for sampling
* `--fit-only`: fit and save `potts_model.npz` then exit
* `--model-npz`: reuse a pre-fit Potts model and skip fitting
* `--plm-device`: device for PLM training (`auto`, `cpu`, `cuda`, or torch device string)

### Local fitting

To fit on a separate machine, run the setup once, activate the venv, then run the
interactive fitter:

```bash
./scripts/potts_setup.sh
source .venv-potts-fit/bin/activate
./scripts/potts_fit.sh
```

### Gibbs sampling

* `--gibbs-method`: `single` or `rex`
* Single-chain:

  * `--gibbs-samples`, `--gibbs-burnin`, `--gibbs-thin`
* Replica exchange:

  * `--rex-betas` (explicit ladder) OR auto-ladder settings:

    * `--rex-n-replicas`, `--rex-beta-min`, `--rex-beta-max`, `--rex-spacing`
  * `--rex-rounds` (total across chains), `--rex-burnin-rounds`, `--rex-sweeps-per-round`, `--rex-thin-rounds`
  * `--rex-chains`: run multiple independent REX chains in parallel (total rounds split across chains; samples concatenated)

### SA/QUBO sampling (QA proxy)

* `--sa-reads`, `--sa-sweeps`
* `--penalty-safety`: scales one-hot constraint penalties (higher = fewer invalid samples)
* `--repair`: `none` or `argmax`

  * `none`: keep samples as-is; report invalid rates (best for honesty)
  * `argmax`: force a valid label per residue (hides constraint issues; use cautiously)

### β_eff estimation (optional)

* `--estimate-beta-eff`: enable calibration
* `--beta-eff-grid`: comma-separated betas to scan (otherwise uses ladder)
* `--beta-eff-w-marg`, `--beta-eff-w-pair`: weights for distance function
* requires `--gibbs-method rex`; reuses the baseline REX samples when the scan grid is contained in the baseline ladder, otherwise runs one extra REX over the requested grid
* output plot is always written to `beta_scan.html` inside `--results-dir` when enabled

### Plotting / misc

* `--results-dir` (required): directory to store run artifacts (summary npz, metadata, plots)
* `--plot-only`: skip sampling and render plots from an existing summary (defaults to `run_summary.npz` in `--results-dir`)
* `--annotate-plots`: more plot annotation (if supported)
* `--seed`: RNG seed
* `--progress`: progress bars

---

## Suggested setups (recommended runs)

### 0) Smoke test (fast, checks plumbing)

Use PMI, short samplers, no plots.

```bash
python main.py --npz data.npz \
  --fit pmi --beta 1.0 \
  --gibbs-method single --gibbs-samples 200 --gibbs-burnin 200 --gibbs-thin 2 \
  --sa-reads 200 --sa-sweeps 500 --sa-tstart 10 --sa-tend 0.1 \
  --penalty-safety 3.0 --repair none
```

What you want to see:

* low invalid sample rate for QUBO
* Gibbs vs MD not completely crazy (it may be imperfect under PMI)

---

### 1) “Baseline science” run (PLM + Gibbs at β=1)

This tells you if the **model fit** is good.

```bash
python main.py --npz data.npz \
  --fit pmi+plm --beta 1.0 \
  --gibbs-method single --gibbs-samples 5000 --gibbs-burnin 2000 --gibbs-thin 2 \
  --sa-reads 1000 --sa-sweeps 2000 \
  --penalty-safety 3.0 --repair none \
  --results-dir results/run_beta1 --progress
```

Interpretation:

* If Gibbs already deviates from MD → representation or model limitations (pairwise Potts, clustering, etc.)
* If Gibbs matches MD but SA does not → QUBO penalties / SA schedule / sampler effect

---

### 2) Hard landscapes: Replica exchange Gibbs reference

Use this when single-chain Gibbs looks unstable or inconsistent.

```bash
python main.py --npz data.npz \
  --fit pmi+plm --beta 1.0 \
  --gibbs-method rex \
  --rex-beta-min 0.2 --rex-beta-max 1 --rex-n-replicas 10 --rex-spacing geom \
  --rex-rounds 4000 --rex-burnin-rounds 1000 --rex-sweeps-per-round 2 --rex-thin-rounds 1 \
  --sa-reads 1000 --sa-sweeps 2000 \
  --penalty-safety 3.0 --repair none \
  --results-dir results/run_rex --progress
```

Watch:

* reported `swap_accept_rate` should not be ~0 (if it is, the ladder spacing is too wide)

---

### 3) Estimate β_eff for SA/QUBO samples

This is the clean way to compare SA to a properly sampled Potts distribution.

```bash
python main.py --npz data.npz \
  --fit pmi+plm --beta 1.0 \
  --gibbs-method rex \
  --rex-beta-min 0.2 --rex-beta-max 1.2 --rex-n-replicas 12 --rex-spacing geom \
  --rex-rounds 4000 --rex-burnin-rounds 1000 \
  --sa-reads 2000 --sa-sweeps 2000 \
  --estimate-beta-eff \
  --beta-eff-w-marg 1.0 --beta-eff-w-pair 1.0 \
  --results-dir results/run_beta_eff \
  --progress
```

What you get:

* a printed table of `D(beta)` values
* `beta_eff` = beta minimizing the distance between SA samples and Gibbs reference samples
* REX reuse: if the scan grid is already in the baseline ladder, the scan reuses those samples; otherwise an extra REX across the grid is run
* HTML plots: `marginals.html` and `beta_scan.html` inside `--results-dir`

---

### 4) Debugging high invalid rates in QUBO

If you see many invalid samples:

* increase `--penalty-safety` (e.g., 5–10)
* temporarily use `--repair argmax` to see “best-case” stats (but still report invalid fraction)

```bash
python main.py --npz data.npz \
  --fit plm --beta 1.0 \
  --gibbs-method rex \
  --sa-reads 1000 --sa-sweeps 2000 \
  --penalty-safety 8.0 --repair none
```

---

## What we compare (the core evaluation)

We compare distributions in reduced space using:

* per-residue marginal distributions (p(x_r))
* pairwise joint distributions (p(x_r,x_s)) on contact edges

and summarize mismatch using Jensen–Shannon divergence (JS):

* JS(MD, Gibbs) tells you: **is the fitted Potts model + sampler reproducing MD statistics?**
* JS(MD, SA) tells you: **does QUBO+SA match MD directly?**
* JS(SA, Gibbs@β_eff) tells you: **does SA behave like Boltzmann sampling at some effective temperature?**

---

## Next extensions (optional but valuable)

* Energy histogram comparisons and ESS/autocorrelation diagnostics (energy traces are already returned by replica exchange)
* Hybrid “QA proposal + Metropolis correction” sampler (turns QA/SA into a principled proposal mechanism)

---

## Practical caveats

* PMI is a great initializer/debugger, but not guaranteed to reproduce MD statistics when sampled.
* PLM is the recommended baseline if you want interpretability + a consistent global energy model.
* β values are model-scale parameters; mapping them to Kelvin is not generally meaningful without calibration.

---
