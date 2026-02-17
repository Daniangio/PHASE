# Gibbs Relaxation 3D: interpretation

This page maps residue-level Gibbs relaxation metrics onto a 3D structure.

## What is colored
- Residues are colored by **flip percentile**:
  - blue: low percentile (typically slower to flip)
  - white: middle percentile
  - red: high percentile (typically earlier/faster to flip)

## Delta (A-B) view
- Delta uses: `flip_percentile_fast(A) - flip_percentile_fast(B)` for each residue.
- Interpretation:
  - red (positive delta): residue flips earlier/faster in **A** than in **B**
  - blue (negative delta): residue flips earlier/faster in **B** than in **A**
  - white (near zero): similar behavior in A and B
- This highlights **asymmetric swappers**: residues that are strongly shifted in one analysis but not the other.
- Sign convention is important: swapping A/B flips the delta sign and inverts red/blue.

## Animation mode (wave)
- With **Animate wave** enabled, residues stay gray until their first-flip statistic reaches the current step.
- As the step slider increases (or Play runs), residues appear in percentile color at their flip time.
- This gives a temporal propagation view of where the system changes first.

## Controls
- **Flip statistic**: choose which first-flip summary drives animation timing (`mean`, `median`, `q25`, `q75`).
- **Residue mapping**:
  - `PDB numbering (auth_seq_id)`: use residue numbers from the PDB file.
  - `Sequential (label_seq_id)`: use 1..N indexing, useful if auth numbering does not align.
- **Load structure (PDB)**: switch among available state structures; coloring is reapplied to the selected structure.
- **Delta color range (|Δ|)**:
  - sets saturation/clipping for delta colors
  - values with `|Δ| >= range` are shown as fully saturated red/blue
  - smaller range: stronger visual contrast
  - larger range: smoother, less saturated map

## Practical reading pattern
1. Start with `mean` and animation enabled.
2. Play once to see the broad wave of change.
3. Switch to `q25` to emphasize earliest responders.
4. Switch to `q75` to see more conservative/late flip behavior.
5. If nothing colors, try switching residue mapping mode.

## Suggested delta workflow
1. Select two analyses (A left, B right) and enable delta.
2. Start with `Delta color range` around `0.20–0.30`.
3. Inspect saturated red/blue patches first (strong asymmetry).
4. Increase range if too much is saturated; decrease if map is too white.
5. Verify biological interpretation by swapping A/B once to confirm sign behavior.
