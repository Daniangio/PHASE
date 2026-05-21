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
