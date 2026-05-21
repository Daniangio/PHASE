# Transient-State Analysis

This analysis finds low-occupancy cluster states that are selectively enriched in one trajectory compared with the other selected trajectories.

It is complementary to Delta JS. Delta JS highlights residues whose full marginal cluster distributions differ. Transient-state analysis instead asks whether a residue or edge briefly visits a rare state that is enriched in a specific trajectory, even if the total occupancy is too small to dominate JS.

## When To Use It

Use this analysis when you suspect short-lived switches, intermediate-like events, ligand-specific rare states, or transition-pathway signatures.

Good examples are residues that have modest Delta JS but show rare cluster visits in one trajectory, such as a rotameric switch that appears in recurrent short bursts.

Do not use this as a standalone proof of mechanism. Treat it as a ranking and hypothesis-generation tool, then inspect the underlying structures/frames.

## Inputs

- `Samples/trajectories`: Requires at least two simulation trajectories. The analysis uses a "leave-one-out" scheme: when evaluating a specific "focal" trajectory, all other selected trajectories are pooled together to form the comparative background.
- `MD label mode`: `assigned` uses the nearest assigned cluster labels. `halo` uses halo labels for MD samples when available.
- `p_min` (Default: 0.005): The minimum occupancy threshold. The state must appear in at least 0.5% of the focal trajectory's frames to filter out single-frame artifacts or simulation noise.
- `p_max` (Default: 0.05): The maximum occupancy threshold. Restricts results to states that occur in fewer than 5% of the focal frames. This ensures the state is truly a rare, transient intermediate rather than a major stable substate.
- `enrichment_min` (Default: 1.0): The minimum log-base-2 fold change required over the background.What is Enrichment? It measures how much more frequently a rare state occurs in your trajectory of interest compared to all other trajectories combined. Because it uses a $\log_2$ scale, an enrichment of 1.0 means the state is 2x more frequent ($2^1$) in the focal trajectory. An enrichment of 3.0 means it is 8x more frequent ($2^3$). This isolates behaviors unique to specific simulation conditions.
- `top_k_nodes`: The maximum number of single-residue hits to save in the final output table.
- `Compute edge states`: Toggle to expand analysis from single residues to pairs of residues moving together.
- `edge_mode`: `cluster`: Scans only residue pairs that are in physical contact (highly recommended for performance). `all_vs_all`: Scans every possible pair of residues in the system (computationally expensive).
- `delta_pmi_min` (Default: blank): The strictness filter for coordinated, cooperative motion.Why do we need this? A pair of residues can show high enrichment for two completely different physical reasons:
    - True Cooperative Motion: The two residues are physically coupled and visit a joint state because they are moving together.
    - Trivial Driven Motion: Residue $i$ changes its behavior drastically on its own, while residue $j$ does nothing unusual. The pair looks "enriched" only because residue $i$ is dominating the math.
To separate these cases, the tool calculates the Pointwise Mutual Information ($\Delta \text{PMI} = \text{PMI}_{\text{focal}} - \text{PMI}_{\text{background}}$).$\Delta \text{PMI} \le 0$: Trivial enrichment driven by a single residue.$\Delta \text{PMI} > 0$: True cooperative or altered pairwise context. Setting a positive value (e.g., 0.2) purges trivial single-residue drivers from your edge table.
- `top_k_edges`: The maximum number of pairwise edge hits to save in the final output table.

## Recommended Balancing Regimes

Regime	enrichment_min	delta_pmi_min	Best Used For...	What It Catches
Exploratory / High Sensitivity	1.0 (2-fold)	0.0 or blank	Small systems or preliminary scans where you do not want to miss subtle trends.	Everything, including trivial single-residue drivers. Requires manual structure filtering.
Standard Balanced (Recommended)	1.5 to 2.0 (~3 to 4-fold)	0.1 to 0.3	General production runs to find clean, distinct pocket or pathway rearrangements.	Modestly rare states that exhibit clear, mathematically verifiable cooperative motion.
Strict Structural Coupling	2.0 to 3.0 (4 to 8-fold)	> 0.5	Identifying major localized allosteric switches or highly coordinated lock-and-key state changes.	Exclusively joint events where the two residues must move together to create the state.

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
