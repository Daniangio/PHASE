### 1) Potts “generalization” and why it’s not automatically meaningful

Devil’s advocate: a Potts model **will almost always reproduce what it was fit on** in the *same representation*, especially if you evaluate it with metrics that are “training-statistics-adjacent” (single-site marginals, some pairwise joints). That’s not yet evidence it’s a good CG model of the protein; it can just mean you successfully fit a maximum-entropy model *of your discretization*. Your own proposal is already on the right track in insisting on separating **representation quality vs model fit vs sampler behavior**. 

So you need validation questions that actually discriminate:

* **(R) Representation validity:** do the per-residue microstates you define form a stable “alphabet” across conditions?
* **(M) Model validity:** does pairwise max-ent explain unseen statistics *without refitting / reclustering*?
* **(S) Sampling validity:** can your chosen sampler reproduce the Potts distribution (or at least match Gibbs at some βeff), without hand-wavy calibration?

#### 1a) Your “merged clustering” assumption (“intermediates are combinations of existing colors”)

This is plausible but absolutely not guaranteed. In A2A, the paper you cite identifies a **pseudo-active state (pAS)** with a **state-specific connector conformation** and microswitch rearrangements; it’s explicitly described as “rather different from both inactive and active” at the structural level, while also mixing features across regions. That’s exactly the kind of scenario where **some residues may need new torsion basins** (new “colors”), even if many do not. 

**Make this falsifiable with a simple diagnostic:**

* Cluster on the merged set (active+inactive) → define microstates.
* Take the third trajectory (your mixed active→pAS long-lived run) and compute:

  * **out-of-support rate**: fraction of torsion points that end up “unassigned/halo” or far from any cluster center if you use a distance threshold (your proposal already treats −1 labels as first-class and lists options like “dedicated transition microstates”). 
  * **novelty test by reclustering held-out**: recluster *only* the held-out condition per residue and measure whether new basins appear that are not recoverable as splits/merges of the merged clustering.
* Report this per residue, and focus on the microswitch/connector residues highlighted in the A2A work (TM3/TM6 region, ionic locks). 

If novelty is non-negligible, the honest conclusion is: **the alphabet is condition-dependent**, so “CG via a fixed per-residue discrete state space” needs either (i) explicit transition/halo states, (ii) soft assignments, or (iii) a different representation.

#### 1b) Critique of your two proposed generalization tests

**Idea 1: train/test split within a trajectory + compare energy histograms / JS**

* Problem: **energy histogram matching is weak evidence.** Energies can match while the model gets correlations wrong (“right marginals, wrong joints”), and energy is not an observable from MD anyway—only your fitted model defines it.
* Better: since you fit by pseudolikelihood, evaluate **held-out negative log pseudolikelihood** (cross-entropy) per residue and per condition. That’s a real predictive metric for Potts in discrete spaces. Then also evaluate:

  * per-residue marginals JS (ok, but not enough),
  * pairwise joints on your edge set,
  * *targeted higher-order* checks around known motifs (e.g., triple of residues spanning TM3–TM6 lock + TM5 residue) to probe “pairwise sufficiency”.

**Make it actually a “generalization” test by splitting by condition, not by time:**

* Train on (active+inactive), test on (mixed/pAS trajectory) **without reclustering**.
* Then rotate: train on (inactive + mixed), test on (active+G-protein) etc.
  That’s a real OOD shift and it hits your main claim.

**Idea 2: remove one (or more) discrete “sequence patterns” and see if sampling recovers them**

* As stated, it’s not clean, because “pattern 1,2,3” across residues mixes three effects:

  1. the pattern might be physically impossible (high energy),
  2. it might be possible but just rare (needs correct couplings),
  3. it might be absent because your clustering/labeling never produced it.
* Also, for a pairwise model, “recovering a held-out joint configuration” is not guaranteed even if the model is correct—if the missing pattern sits in a narrow, entropically small pocket, you won’t see it unless you control for sampling effort.

**A much more solid version of this idea is conditional prediction (inpainting):**

* Randomly choose a subset of residues as “observed” (or pick a microswitch set), clamp them to their true states from held-out frames, and ask the Potts model to predict the remaining residues:

  * compute conditional accuracy / conditional NLL / calibration
  * compare across conditions (active vs inactive vs pAS)
    This directly supports your later “marginal conditioning from experiment” story.

**How to do “sequence-space clustering” if you still want it:**
Treat each frame as a categorical vector and use a **Hamming / weighted-Hamming** distance (weights = residue entropy or functional importance), then do k-medoids / hierarchical / HDBSCAN. But I would treat this as *macrostate discovery*, not as a primary validation of Potts.

---

### 2) Does QA find global minima on a general Potts/QUBO?

Bluntly: **there is no general guarantee**, and “global minimum” is not even the right target if your scientific goal is *sampling* rather than *optimization* (your proposal already warns QA is not assumed to be a speedup and must be diagnosed vs Gibbs). 

If you still want to test “global minimum” capability, do it in a way that’s unambiguous:

**A. Use problems where the true optimum is known**

* Small instances: exact enumeration (feasible for small N or small total state count).
* Medium instances: ILP / branch-and-bound solvers on the QUBO.
* Planted-solution benchmarks: generate QUBOs with a known ground state and controllable gap; then measure success probability.

**B. Separate three confounders**

1. **Encoding correctness** (one-hot penalties, chain breaks if hardware embedding is used). Track violation rate and penalty energy separately (you already plan this). 
2. **Schedule/freeze-out** (QA returns samples from some effective distribution, often not the ground state unless tuned).
3. **Connectivity/embedding overhead** (dense Potts edges + one-hot constraints can explode the binary graph).

**C. Compare to strong classical baselines under matched budget**

* Simulated annealing, parallel tempering, population annealing / SMC.
  If QA doesn’t beat these on success probability vs wall-clock (or vs energy gap), your claim should not be “QA finds global minima,” but rather “QA is a viable *proposal mechanism* / heuristic optimizer in this pipeline.”

---

### 3) If (1) and (2) hold, can you claim “sample metastable states under marginal constraints”?

Yes, but only if you phrase it carefully, and only after you validate **conditional correctness**.

**Key conceptual point:** “experiment measures a microswitch” is **an observation model**, not a hard clamp.

* Hard clamp = set x_r = k.
* Observation model = add a **soft field**: (E'(x)=E(x)-\lambda \log p(\text{obs}\mid x)) or simply bias toward states consistent with the measurement.

**What you can legitimately claim if you do the right experiments:**

1. **Conditional inference in the reduced space:** given constraints on a subset of residues, your method can generate samples of the remaining residues consistent with the learned couplings.
2. **Efficient MAP search under constraints:** QA (or SA) can be used to propose low-energy configurations satisfying constraints, *then* corrected/refined by MCMC (your proposal already lists “QA/SA proposals + Metropolis correction” as the principled way to keep correctness). 

**What you must *not* overclaim without extra work:**

* That these constrained samples correspond to true physical metastable states in 3D, unless you show:

  * backmapped structures are stable after relaxation/short restrained MD,
  * and the constrained ensemble reproduces independent observables (contacts, distances, etc.).

**Critical validation experiment for the “microswitch conditioning” story (do this early):**

* Pick a microswitch set highlighted in A2A (e.g., ionic lock states that differentiate active/inactive/pAS). 
* On MD, identify frames where that switch is active vs inactive.
* Train Potts on all frames (or on two conditions), then:

  * condition on the microswitch being active,
  * sample the rest,
  * compare the resulting distributions of other residues to the MD subset with that microswitch active (and same for inactive).
    If this fails, the entire “conditioning from experiment” claim collapses, regardless of QA.

---

### A tight, high-impact validation ladder (if you want a single narrative)

1. **Alphabet stability across conditions** (merged clustering + novelty/out-of-support; handle halo/transition states explicitly). 
2. **Leave-one-condition-out Potts validation** (held-out pseudolikelihood + marginals + pairwise + targeted higher-order).
3. **Conditional prediction / inpainting** (microswitch clamp → recover the rest).
4. **Sampler benchmarking** (QA/SA vs Gibbs/tempering; show whether QA is optimizer, sampler, or proposal+correction). 

If you execute (1)–(3) cleanly, you’ll have a strong reason to use Potts even if QA ends up being “just” a heuristic proposal engine rather than a magic global-min finder.
