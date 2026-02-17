# Gibbs Relaxation Analysis

This analysis runs a controlled relaxation experiment:

1. Choose a starting sample (typically an `md_eval` sample).
2. Choose a target Potts model (the Hamiltonian used for Gibbs updates).
3. Randomly select `n_start_frames` frames from the starting sample.
4. From each selected frame, run one Gibbs trajectory of length `gibbs_sweeps`.
5. Aggregate per-residue first-flip statistics across all trajectories.

Defaults:

- `n_start_frames = 100`
- `gibbs_sweeps = 1000`

## Main outputs

- `first_flip_steps[run, residue]`: first sweep where the residue changed state from its start value.
- `mean_first_flip_steps[residue]`: average first-flip sweep across runs.
- `flip_percentile_fast[residue]`: percentile rank where `1.0` = fastest flipper, `0.0` = slowest.
- `flip_prob_time[sweep, residue]`: probability that residue has flipped at that sweep.
- `energy_mean[sweep]`, `energy_std[sweep]`: energy relaxation curve under the selected model.

## Color interpretation

Recommended mapping:

- red = high `flip_percentile_fast` (early/faster responders),
- blue = low `flip_percentile_fast` (late/slower responders).

This percentile-based coloring is rank-robust and makes experiments comparable across systems and run lengths.

## Practical notes

- Use `start_label_mode=assigned` unless you explicitly need halo labels.
- If `keep_invalid=false`, invalid rows are dropped before selecting starts.
- If many starting frames are invalid/out-of-range for the selected model, fewer starts may be used than requested.

