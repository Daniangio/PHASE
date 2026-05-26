# Transient-State Analysis

This analysis finds low-occupancy cluster states that are selectively enriched in one trajectory compared with the other selected trajectories.

It is complementary to Delta JS. Delta JS highlights residues whose full marginal cluster distributions differ. Transient-state analysis instead asks whether a residue or edge briefly visits a rare state that is enriched in a specific trajectory, even if the total occupancy is too small to dominate JS.

## When To Use It

Use this analysis when you suspect short-lived switches, intermediate-like events, ligand-specific rare states, or transition-pathway signatures.

Good examples are residues that have modest Delta JS but show rare cluster visits in one trajectory, such as a rotameric switch that appears in recurrent short bursts.

Do not use this as a standalone proof of mechanism. Treat it as a ranking and hypothesis-generation tool, then inspect the underlying structures/frames.

## Inputs

- `Samples/trajectories`: at least two samples are required. The background for each sample is built from all other selected samples.
- `MD label mode`: `assigned` uses the nearest assigned cluster labels. `halo` uses halo labels for MD samples when available.
- `p_min`: minimum occupancy required in the focal trajectory. Default `0.005` means at least 0.5% of frames.
- `p_max`: maximum occupancy allowed in the focal trajectory. Default `0.05` restricts hits to states below 5% occupancy.
- `enrichment_min`: minimum log2 enrichment over the leave-one-out background. Default `1.0` means at least twofold enrichment.
- `top_k_nodes`: maximum number of residue-cluster hits stored.
- `Compute edge states`: also evaluates joint cluster states on residue pairs.
- `edge_mode`: `cluster` uses the cluster/contact edge set; `all_vs_all` scans every residue pair and can be expensive.
- `delta_pmi_min`: optional cutoff for pairwise-specific enrichment. Positive values keep edge states enriched beyond what is expected from marginal residue occupancies.
- `top_k_edges`: maximum number of edge-cluster hits stored.

## Node Criterion

For residue `i`, cluster `k`, and selected trajectory `m`, the analysis computes:

```text
p_i^m(k) = fraction of frames where residue i is in cluster k
```

The background is leave-one-out:

```text
p_i^not_m(k) = occupancy of the same cluster after pooling all other selected trajectories
```

A residue-cluster state is reported when:

```text
p_min <= p_i^m(k) <= p_max
log2((p_i^m(k) + epsilon) / (p_i^not_m(k) + epsilon)) > enrichment_min
```

The score is:

```text
score = log2_enrichment * sqrt(count)
```

This rewards enrichment while penalizing unsupported one-frame events.

## Edge Criterion

For edge `(i,j)`, the analysis evaluates joint cluster states `(k,l)`:

```text
p_ij^m(k,l) = fraction of frames where i is in k and j is in l
```

The same occupancy and enrichment criteria are applied against the leave-one-out background.

For edge hits, the analysis also computes a PMI-like correction:

```text
PMI_m = log((p_ij^m(k,l) + epsilon) / (p_i^m(k) * p_j^m(l) + epsilon))
Delta PMI = PMI_m - PMI_background
```

Positive `Delta PMI` means the joint state is enriched beyond what would be expected from the two residues independently. These are usually the more interesting edge hits.

## Result Columns

### Node Table

- `Residue`: residue label from the cluster topology.
- `Sample`: trajectory/sample where the transient state is enriched.
- `Cluster`: residue cluster ID.
- `Occ.`: occupancy in the focal sample.
- `Bg.`: leave-one-out background occupancy.
- `log2 enrich`: log2 fold enrichment over background.
- `Episodes`: number of contiguous visits to the cluster.
- `Mean dwell`: mean duration of visits in frames.
- `Max dwell`: longest visit in frames.
- `Score`: enrichment weighted by support, `log2_enrichment * sqrt(count)`.

Column Definitions & Structural Meaning
Residue: The specific amino acid residue label from your structural topology. It identifies where the transient event is occurring in the protein.Sample: The specific simulation trajectory where this particular state is heavily populated. This is your focal trajectory. In your data, these are distinct states like MD active, MD pas (passive), or MD inactive.Cluster: The structural/conformational cluster identifier (e.g., c0, c1, c2) that the residue is visiting. This represents a specific local state, such as a rotameric position or a backbone dihedral state.Occ. (Occupancy): The percentage of total simulation frames in the focal trajectory where this specific residue was found in this specific cluster. For example, res_174 spends exactly 4.50% of its time in cluster c0 during the MD active simulation.Bg. (Background): The pooled occupancy of this exact same cluster across all other trajectories combined (the leave-one-out background).Crucial Observation: For your top rows (res_174, res_21, res_74, res_51), the background is 0.00%. This means this conformational state never appears in the other simulation conditions, making it completely unique to the focal sample.log2 enrich (Log2 Enrichment): A logarithmic measure of how much more frequent this state is in the focal sample versus the background.When background (Bg.) is absolute 0.00%, this value spikes to a mathematical ceiling (around 23.0 to 25.0 in your data) driven entirely by the regularization constant $\epsilon$.For res_182, the enrichment is 6.07. This means it is $2^{6.07} \approx 67$ times more frequent in MD active ($8.71\%$) than in the background ($0.13\%$).Episodes: The number of discrete, contiguous blocks of frames where the residue entered and remained in this cluster.res_174 enters cluster c0 211 separate times during the simulation, meaning it is a highly recurrent, fluttering switch.res_43 has exactly 1 episode, meaning it transitioned into cluster c1 exactly once, stayed there for a while, and left (or stayed until the simulation ended).Mean dwell: The average duration (measured in simulation frames) that the residue remains inside the cluster during a single episode. For res_174, each of its 211 visits lasted an average of 24.0 frames.Max dwell: The longest single continuous visit (in frames) recorded during the simulation. For res_21, while its average visit was 94.0 frames, its longest single block lasted for an impressive 1,994 frames.Score: The final metric used to rank the relevance of these hits, calculated as $\text{Score} = \text{log2\_enrichment} \times \sqrt{\text{total\_count}}$. It balances high enrichment with physical statistical support (total frames).

### Edge Table

- `Edge`: residue pair.
- `Sample`: trajectory/sample where the joint state is enriched.
- `Clusters`: joint cluster state `c_i/c_j`.
- `Occ.`: joint occupancy in the focal sample.
- `Bg.`: leave-one-out joint occupancy.
- `log2 enrich`: log2 fold enrichment over background.
- `Delta PMI`: pairwise-specific enrichment beyond marginal residue effects.
- `Episodes`, `Mean dwell`, `Max dwell`: temporal recurrence statistics.
- `Score`: enrichment/support score, boosted by positive `Delta PMI`.

## Interpretation Tips

- High `log2 enrich` and reasonable `count` indicate a trajectory-specific rare state.
- Many short `Episodes` with small `Mean dwell` can indicate a fast recurrent switch.
- One episode with very short dwell can be clustering noise or a single outlier frame.
- `Occ.` close to `p_max` may be less “transient” and more like a minor stable substate.
- Edge hits with high enrichment but low or negative `Delta PMI` may be driven by one residue marginally changing.
- Edge hits with positive `Delta PMI` are better candidates for altered pairwise context.
- Prefer `cluster` edge mode for routine use. Use `all_vs_all` only for smaller systems or targeted checks.

## Practical Defaults

A reasonable first pass is:

```text
p_min = 0.005
p_max = 0.05
enrichment_min = 1.0
edge_mode = cluster
delta_pmi_min = blank or 0.0
```

For cleaner, more conservative tables, raise `p_min`, raise `enrichment_min`, or set `delta_pmi_min > 0` for edges.

## Flexibility Filtering

Flexible loops often generate many local clusters and can dominate transient-state tables with non-specific flickering. PHASE reports `K`, the number of clusters available for each residue. A large `K` often indicates high local flexibility or noisy conformational diversity.

The web page includes a `Max residue clusters` filter. For example, setting it to `6` hides residues with more than six clusters. This is not a proof that the remaining residues are functional switches, but it is a useful first-pass filter to reduce loop-dominated hits.

Interpretation rule of thumb:

- Low or moderate `K` plus repeated enriched visits: more switch-like.
- High `K`, many low-occupancy clusters, and very short dwells: more likely flexible-loop behavior.
- High `K` residues can still be important, but they require structure/frame inspection.

## 3D Viewer

The transient 3D page colors residues by their strongest transient-state score after the active sample and `K` filters. The structure can be loaded either from the representative state PDB or from a specific frame of a stored state trajectory.

Frame loading requires that the state has a stored trajectory. If only a PDB was uploaded, frame `0` is equivalent to the static structure.

Use the 3D view to check whether hits cluster in a meaningful region, occur near known motifs, or are dominated by solvent-exposed loops.

## Trajectory Frame Panel

The 3D viewer has a dedicated frame panel. Choose a reference state PDB for static coloring, or enable trajectory-frame loading to extract one stored frame from that state's trajectory. If the state has no trajectory, upload one from the System page: open the States panel details, choose the trajectory file for the state, and click Upload & Build.

The current implementation loads one frame at a time as a PDB instance. Use it to inspect whether a transient hit corresponds to a plausible structural switch or to broad loop flexibility.

## Raw Mol* Trajectory Test Page

The `Mol* traj test` page loads a topology file plus a raw trajectory file directly into Mol*. It is useful when you want native Mol* frame playback instead of the single-frame PDB extraction used by the transient 3D page.

Supported coordinate inputs depend on Mol* browser support, but XTC, DCD, TRR, and NCTRAJ are exposed in the loader. The topology and trajectory must have matching atom order and atom count.

Current behavior:

- Stored PHASE state: loads the stored state structure and raw stored trajectory from the webserver.
- Local files: loads a local PDB/mmCIF/GRO topology and local XTC/DCD/TRR trajectory through browser object URLs.
- Frame selection: Mol* receives the full trajectory and its native frame controls are used for scrolling. Server-side frame-range subsetting is not implemented in this test page.

If a stored state was created from `phase_console` with an absolute trajectory path outside `PHASE_DATA_ROOT`, Docker may not be able to stream that file. In that case the raw trajectory test page will ask you to either re-upload the trajectory from the System page, which stores it under the shared data root, or bind-mount the original host trajectory directory into the backend container.
