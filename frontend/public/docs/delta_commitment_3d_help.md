# Delta Commitment (3D)

This page visualizes **per-residue commitment** for a fixed pair of Potts models (A,B) on top of a 3D structure.

## What Is Being Colored

For each residue `i`, commitment is a value `q_i` derived from the Potts field difference:

`q_i = Pr(δ_i < 0)` where `δ_i(t) = h^A_i(s_{t,i}) - h^B_i(s_{t,i})`.

Interpretation:
- `q_i ~ 0`: that residue tends to favor model **B** (δ_i mostly positive).
- `q_i ~ 1`: that residue tends to favor model **A** (δ_i mostly negative).
- `q_i ~ 0.5`: ambiguous / mixed signs (partial commitment).

## Commitment Modes

The UI lets you switch between multiple visualization modes:

- **Base: `Pr(Δh < 0)`**
- **Centered: `Pr(Δh ≤ median(ref))`**
  - Uses a per-residue threshold computed from a selected *reference ensemble* so neutral residues appear closer to white.
- **Mean: `sigmoid(-E[Δh]/scale)`**
  - Smooth view based on the mean field difference per residue.

### Reading Centered Mode

Centered mode is best interpreted as a **reference-normalized** view:

- Pick a reference ensemble (often an MD ensemble).
- In centered mode, the reference is calibrated to look ~white per residue.
- When you switch to another ensemble while keeping the same reference:
  - residues turning **red** indicate that ensemble is more **A-like** than the reference at those residues
  - residues turning **blue** indicate it is more **B-like** than the reference at those residues

## Colors

The residue overlay uses a diverging palette:
- **blue**: `q → 0`
- **white**: `q ≈ 0.5`
- **red**: `q → 1`

## Coupling Links (Optional)

If enabled, the page draws cylinder links for the **top Potts edges** (ranked by `|ΔJ|`).

Important:
- Edge commitment `q_ij` is stored only for the **top-K edges** chosen when the delta-commitment analysis is computed.
- If you request more links than were stored, only the available ones can be shown.
- Default `top_k_edges` is set high (currently `2000`) so you can visualize hundreds of links without recomputing.

## Edge-Weighted Residue Coloring (Optional)

If enabled, residue colors are blended with the average edge commitment of incident top edges (weighted by `|ΔJ|`).
This can make coupling-supported patches easier to see on the structure.

## Practical Notes

- The selection tries to match residues by PDB residue numbering (`auth_seq_id`) when residue labels contain integers (e.g. `res_279`).
  If not available, it falls back to sequential residue indices (`label_seq_id`).
- If you don’t see colored residues, verify you loaded a structure compatible with the cluster residue indexing.
