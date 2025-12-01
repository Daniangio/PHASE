#!/usr/bin/env python3
"""
VAMP-2 + MSM + PCCA+ metastable state analysis on phi/psi sin-cos features.

Pipeline
--------
1. MDAnalysis: read topology + trajectory
2. Compute Ramachandran (phi, psi) for all residues in a selection
3. Build sin/cos embedding (features: sinφ, cosφ, sinψ, cosψ per residue)
4. VAMP on features at lag in ps (converted to frames)
5. k-means microstate clustering in VAMP space
6. Maximum-likelihood MSM on microstates
7. PCCA+ metastable coarse graining:
   - choose number of macrostates k in [k_min, k_max] via spectral gap
   - get membership matrix chi (microstate → metastates)
   - hard-assign microstates, but mark:
       - microstates with max chi < min_membership as outliers
       - macrostates with population < min_macro_pop_frac as outliers
   - frames belonging to outlier microstates get label -1
8. Save:
   - per-frame macro labels (with -1 = outliers)
   - frame index lists per macrostate and for outliers
   - representative PDB for each non-outlier macrostate
   - simple diagnostic plots (VAMP singular values, VAMP scatter, macro TS, MSM timescales)

Dependencies
------------
    pip install MDAnalysis deeptime scikit-learn matplotlib

"""

import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran

import matplotlib.pyplot as plt

from deeptime.decomposition import VAMP
from deeptime.markov.msm import MaximumLikelihoodMSM
from sklearn.cluster import KMeans


# =========================
# Feature construction
# =========================

def compute_phi_psi_sincos(universe, selection="protein"):
    """
    Compute φ/ψ dihedrals for all residues in `selection`,
    return sin/cos embedding.

    Returns
    -------
    X : np.ndarray, shape (n_frames, n_features)
        flattened [sinφ, cosφ, sinψ, cosψ] for each residue.
    n_residues : int
    """
    rama = Ramachandran(universe, select=selection)
    rama.run()  # fills rama.angles [T, R, 2] in degrees

    angles = rama.angles  # (T, R, 2)
    n_frames, n_res, _ = angles.shape

    phi = np.deg2rad(angles[..., 0])
    psi = np.deg2rad(angles[..., 1])

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    feats = np.stack([sin_phi, cos_phi, sin_psi, cos_psi], axis=-1)  # (T, R, 4)
    X = feats.reshape(n_frames, -1)

    return X, n_res


# =========================
# Lag time handling
# =========================

def infer_frame_dt_ps(universe, user_dt_ps=None):
    """
    Infer frame time step in ps from MDAnalysis trajectory.
    Fallback to user-specified dt if trajectory.dt is missing.
    """
    dt_ps = None
    if hasattr(universe.trajectory, "dt") and universe.trajectory.dt is not None:
        dt_ps = float(universe.trajectory.dt)

    if dt_ps is None:
        if user_dt_ps is None:
            raise ValueError(
                "Could not infer dt from trajectory. "
                "Please provide --frame-dt-ps explicitly."
            )
        dt_ps = float(user_dt_ps)

    if dt_ps <= 0:
        raise ValueError(f"Non-positive dt_ps detected: {dt_ps}")

    return dt_ps


def lag_ps_to_frames(desired_lag_ps, frame_dt_ps, n_frames):
    """
    Convert desired lag in ps to lag in frames, respecting trajectory stride.

    We enforce:
        effective_tau_ps = max(desired_lag_ps, frame_dt_ps)
        lag_frames = round(effective_tau_ps / frame_dt_ps)

    and clip so that lag_frames < n_frames.
    """
    if desired_lag_ps <= 0:
        raise ValueError("desired_lag_ps must be > 0")

    effective_tau_ps = max(desired_lag_ps, frame_dt_ps)
    lag_frames = int(round(effective_tau_ps / frame_dt_ps))

    if lag_frames < 1:
        lag_frames = 1

    if lag_frames >= n_frames:
        raise ValueError(
            f"Requested lag {effective_tau_ps} ps -> {lag_frames} frames, "
            f"but trajectory has only {n_frames} frames. "
            "Choose a smaller lag or use a longer trajectory."
        )

    return lag_frames, effective_tau_ps


# =========================
# VAMP
# =========================

def run_vamp(X, lag_frames, dim, scaling="kinetic_map"):
    """
    Run VAMP on a single trajectory of features.

    X : (T, d)
    lag_frames : int
    dim : int or float (deeptime semantics: dimension or kinetic variance threshold)
    scaling : None or "kinetic_map"

    Returns
    -------
    Y : (T, d_vamp) VAMP-projected trajectory
    model : deeptime.decomposition.CovarianceKoopmanModel
    """
    estimator = VAMP(lagtime=lag_frames, dim=dim, scaling=scaling)
    model = estimator.fit(X).fetch_model()
    Y = model.transform(X)
    return Y, model


# =========================
# MSM + PCCA
# =========================

def cluster_microstates(Y, n_microstates, random_state=0):
    """
    K-means microstate clustering in VAMP space.

    Returns
    -------
    micro_labels : (T,) int
    centers : (n_microstates, d_vamp)
    """
    km = KMeans(
        n_clusters=n_microstates,
        n_init="auto",
        random_state=random_state
    )
    micro_labels = km.fit_predict(Y)
    centers = km.cluster_centers_
    return micro_labels, centers


def build_msm(micro_labels, lag_frames, reversible=True):
    """
    Build maximum-likelihood MSM on microstate trajectory.
    """
    estimator = MaximumLikelihoodMSM(reversible=reversible)
    msm = estimator.fit(micro_labels, lagtime=lag_frames).fetch_model()
    return msm


def choose_n_macrostates_by_spectral_gap(msm, k_min, k_max):
    """
    Choose number of macrostates using spectral gap in MSM eigenvalues.

    Eigenvalues are sorted descending; λ_1 = 1.
    We look at gaps Δ_k = λ_k - λ_(k+1) for k in [k_min, k_max],
    and pick k where Δ_k is maximal.

    Returns
    -------
    best_k : int
    eigenvalues : ndarray
    gaps : dict{k: gap}
    """
    evals = msm.eigenvalues()
    # sort descending just to be explicit
    evals = np.sort(evals)[::-1]

    k_max_eff = min(k_max, len(evals) - 1)  # need λ_{k+1}
    if k_max_eff < k_min:
        raise ValueError("Not enough eigenvalues to choose macrostates in given range.")

    gaps = {}
    best_k = None
    best_gap = None

    for k in range(k_min, k_max_eff + 1):
        gap = evals[k - 1] - evals[k]  # λ_k - λ_{k+1}, 1-based indexing in text
        gaps[k] = gap
        if best_gap is None or gap > best_gap:
            best_gap = gap
            best_k = k

    return best_k, evals, gaps


def pcca_metastable_sets(msm, n_macrostates):
    """
    Run PCCA+ to obtain metastable memberships.

    Returns
    -------
    pcca_model : deeptime.markov.PCCAModel
    memberships : (n_microstates, n_macrostates) array
    """
    pcca_model = msm.pcca(n_macrostates)
    memberships = pcca_model.memberships
    return pcca_model, memberships


def derive_macro_labels_with_outliers(
        micro_labels,
        memberships,
        min_membership=0.5,
        min_macro_pop_frac=0.01
):
    """
    Derive per-frame macrostate labels with an outlier state (-1).

    - Start with hard assignment: micro -> argmax_j memberships[i, j]
    - Mark microstates with max_membership < min_membership as outliers
    - Compute macro populations; macrostates with population fraction
      < min_macro_pop_frac are treated as 'small outliers'.
    - Frames mapped to outlier microstates or to small macrostates
      get macro label -1.

    Returns
    -------
    macro_labels : (T,) int, in {0,1,...,K-1} or -1 for outliers
    macro_pop_frac : (K,) float (original before outlier removal)
    macro_is_small : (K,) bool
    micro_best_macro : (n_micro,) int (hard assignment before outlier masks)
    micro_is_outlier : (n_micro,) bool
    """
    n_micro, n_macro = memberships.shape

    max_membership_vals = memberships.max(axis=1)
    micro_best_macro = memberships.argmax(axis=1)

    micro_is_outlier = max_membership_vals < min_membership

    # initial per-frame assignment from microstates
    micro_to_macro = micro_best_macro.copy()
    n_frames = len(micro_labels)

    macro_labels = micro_to_macro[micro_labels]

    # mark frames whose microstate is "membership-outlier"
    outlier_micro_states = np.where(micro_is_outlier)[0]
    outlier_micro_mask = np.zeros(n_micro, dtype=bool)
    outlier_micro_mask[outlier_micro_states] = True
    frame_is_membership_outlier = outlier_micro_mask[micro_labels]

    # macro populations BEFORE removing small sets
    macro_pop = np.zeros(n_macro, dtype=int)
    for m in range(n_macro):
        macro_pop[m] = np.sum(macro_labels == m)
    macro_pop_frac = macro_pop / float(n_frames)

    macro_is_small = macro_pop_frac < min_macro_pop_frac

    # build final label array
    final_labels = macro_labels.copy()
    # membership-based outliers:
    final_labels[frame_is_membership_outlier] = -1
    # size-based small macrostates:
    for m in range(n_macro):
        if macro_is_small[m]:
            final_labels[macro_labels == m] = -1

    return final_labels, macro_pop_frac, macro_is_small, micro_best_macro, micro_is_outlier


# =========================
# Representative PDBs
# =========================

def choose_representative_frames(Y, macro_labels):
    """
    For each non-outlier macrostate, choose a representative frame
    as the frame closest to the macrostate mean in VAMP space.

    Returns
    -------
    rep_frames : dict {macro_state: frame_index}
    """
    rep_frames = {}
    unique_macros = sorted(m for m in np.unique(macro_labels) if m >= 0)

    for m in unique_macros:
        idx = np.where(macro_labels == m)[0]
        if len(idx) == 0:
            continue
        Y_m = Y[idx]
        center = Y_m.mean(axis=0, keepdims=True)
        d2 = np.sum((Y_m - center)**2, axis=1)
        rep_frame = idx[np.argmin(d2)]
        rep_frames[m] = rep_frame

    return rep_frames


def write_representative_pdbs(universe, rep_frames, out_prefix):
    """
    Write a PDB snapshot for each macrostate representative frame.

    Files: <out_prefix>_macro_M_rep.pdb
    """
    for m, frame in rep_frames.items():
        universe.trajectory[frame]
        out_name = f"{out_prefix}_macro_{m}_rep.pdb"
        with mda.Writer(out_name, universe.atoms.n_atoms) as W:
            W.write(universe.atoms)
        print(f"[INFO] Wrote representative PDB for macro {m}: frame {frame} -> {out_name}")


# =========================
# Simple plots
# =========================

def plot_singular_values(svals, out_prefix):
    plt.figure()
    plt.bar(range(1, len(svals) + 1), svals)
    plt.xlabel("VAMP mode index")
    plt.ylabel("Singular value")
    plt.title("VAMP singular values")
    plt.tight_layout()
    plt.savefig(out_prefix + "_vamp_singular_values.png")
    plt.close()


def plot_vamp_scatter(Y, micro_labels, out_prefix):
    if Y.shape[1] < 2:
        print("[WARN] VAMP dimension <2 → skipping scatter plot.")
        return
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=5, c=micro_labels, cmap="tab20", alpha=0.8)
    plt.xlabel("VAMP-1")
    plt.ylabel("VAMP-2")
    plt.title("VAMP reduced space — microstate clustering")
    plt.tight_layout()
    plt.savefig(out_prefix + "_vamp_scatter_micro.png", dpi=250)
    plt.close()


def plot_macro_timeseries(macro_labels, out_prefix):
    plt.figure()
    plt.plot(macro_labels, lw=0.8)
    plt.xlabel("Frame")
    plt.ylabel("Macrostate (-1 = outlier)")
    plt.title("Macrostate time series")
    plt.tight_layout()
    plt.savefig(out_prefix + "_macro_timeseries.png")
    plt.close()


def plot_msm_timescales(msm, frame_dt_ps, out_prefix):
    """
    Plot MSM implied timescales in physical time (ps), excluding stationary process.
    """
    timescales_lag = msm.timescales()  # in units of lag steps
    # timescales[0] is for the slowest non-stationary mode; timescales include lag
    # Multiply by frame dt * lagtime to get physical units.
    lag_frames = msm.lagtime
    timescales_ps = timescales_lag * frame_dt_ps * lag_frames

    plt.figure()
    plt.semilogy(range(1, len(timescales_ps) + 1), timescales_ps, marker='o')
    plt.xlabel("MSM timescale index")
    plt.ylabel("Timescale [ps]")
    plt.title("MSM implied timescales")
    plt.tight_layout()
    plt.savefig(out_prefix + "_msm_timescales.png")
    plt.close()


# =========================
# Main CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="VAMP-2 + MSM + PCCA+ metastable analysis on phi/psi sin-cos features."
    )
    parser.add_argument("topology", help="Topology file (PDB/PSF/GRO/etc.)")
    parser.add_argument("trajectory", help="Trajectory file (DCD/XTC/TRR/etc.)")

    parser.add_argument(
        "--selection",
        default="protein",
        help="MDAnalysis atom selection for Ramachandran (default: protein)."
    )
    parser.add_argument(
        "--lag-ps",
        type=float,
        required=True,
        help="Desired lag time in picoseconds for VAMP/MSM."
    )
    parser.add_argument(
        "--frame-dt-ps",
        type=float,
        default=None,
        help="Override frame time step in ps if not stored in trajectory."
    )
    parser.add_argument(
        "--vamp-dim",
        default=5,
        help=(
            "VAMP output dimension. "
            "int (>=1) or float in (0,1] for kinetic variance threshold. "
            "Default: 5."
        ),
    )
    parser.add_argument(
        "--n-microstates",
        type=int,
        default=100,
        help="Number of k-means microstates in VAMP space (default: 100)."
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum number of macrostates for spectral-gap selection (default: 2)."
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=4,
        help="Maximum number of macrostates for spectral-gap selection (default: 4)."
    )
    parser.add_argument(
        "--min-membership",
        type=float,
        default=0.5,
        help="Minimum PCCA membership for a microstate to be considered assigned "
             "(default: 0.5). Microstates below are treated as outliers."
    )
    parser.add_argument(
        "--min-macro-pop-frac",
        type=float,
        default=0.01,
        help="Minimum macrostate population fraction (0-1). "
             "Macrostates below are treated as small outliers (default: 0.01)."
    )
    parser.add_argument(
        "--out-prefix",
        default="vamp_msm_pcca",
        help="Prefix for output files."
    )

    args = parser.parse_args()

    # Load trajectory
    u = mda.Universe(args.topology, args.trajectory)

    # Features
    X, n_res = compute_phi_psi_sincos(u, selection=args.selection)
    n_frames, n_feat = X.shape
    print(f"[INFO] φ/ψ features: {n_frames} frames, {n_res} residues, {n_feat} features.")

    # Lag handling
    dt_ps = infer_frame_dt_ps(u, user_dt_ps=args.frame_dt_ps)
    lag_frames, effective_tau_ps = lag_ps_to_frames(args.lag_ps, dt_ps, n_frames)
    print(f"[INFO] Frame dt = {dt_ps:.3f} ps.")
    print(f"[INFO] Desired lag = {args.lag_ps:.3f} ps, "
          f"effective lag (max(desired, dt)) = {effective_tau_ps:.3f} ps.")
    print(f"[INFO] Using lagtime = {lag_frames} frames for VAMP + MSM.")

    # Parse VAMP dim arg
    try:
        vamp_dim = int(args.vamp_dim)
    except ValueError:
        vamp_dim = float(args.vamp_dim)

    # VAMP
    Y, vamp_model = run_vamp(X, lag_frames=lag_frames, dim=vamp_dim, scaling="kinetic_map")
    print(f"[INFO] VAMP output shape: {Y.shape}")
    print(f"[INFO] VAMP singular values (first 5): {vamp_model.singular_values[:5]}")

    # Microstates
    n_micro = args.n_microstates
    if n_micro > n_frames:
        raise ValueError(
            f"Number of microstates ({n_micro}) cannot exceed number of frames ({n_frames})."
        )

    micro_labels, centers = cluster_microstates(Y, n_microstates=n_micro)
    print(f"[INFO] Microstates: {n_micro}, unique occupied: {len(np.unique(micro_labels))}.")

    # MSM
    msm = build_msm(micro_labels, lag_frames=lag_frames, reversible=True)
    print(f"[INFO] MSM built with {msm.n_states} microstates (connected set).")

    # Choose n_macrostates via spectral gap
    best_k, evals, gaps = choose_n_macrostates_by_spectral_gap(
        msm,
        k_min=args.k_min,
        k_max=args.k_max
    )
    print(f"[INFO] MSM eigenvalues (first 6): {evals[:6]}")
    print(f"[INFO] Spectral gaps in range k={args.k_min}..{args.k_max}: {gaps}")
    print(f"[INFO] Selected number of macrostates (PCCA): k = {best_k}")

    # PCCA metastable sets
    pcca_model, memberships = pcca_metastable_sets(msm, best_k)
    print(f"[INFO] PCCA memberships shape: {memberships.shape}")

    # Macro labels with outliers
    macro_labels, macro_pop_frac, macro_is_small, micro_best_macro, micro_is_outlier = \
        derive_macro_labels_with_outliers(
            micro_labels,
            memberships,
            min_membership=args.min_membership,
            min_macro_pop_frac=args.min_macro_pop_frac
        )

    n_outliers = np.sum(macro_labels == -1)
    print(f"[INFO] Macrostate population fractions (before small-state removal): "
          f"{macro_pop_frac}")
    print(f"[INFO] Macrostates considered 'small' (pop < {args.min_macro_pop_frac}): "
          f"{macro_is_small}")
    print(f"[INFO] Frames labeled as outliers (macro = -1): {n_outliers} / {n_frames} "
          f"({n_outliers / n_frames:.3f})")

    # Save numeric outputs
    prefix = args.out_prefix
    np.save(prefix + "_features.npy", X)
    np.save(prefix + "_vamp_Y.npy", Y)
    np.save(prefix + "_vamp_singular_values.npy", vamp_model.singular_values)
    np.save(prefix + "_micro_dtraj.npy", micro_labels)
    np.save(prefix + "_micro_centers.npy", centers)
    np.save(prefix + "_msm_eigenvalues.npy", evals)
    np.save(prefix + "_pcca_memberships.npy", memberships)
    np.save(prefix + "_macro_dtraj.npy", macro_labels)

    # Save frame index lists per metastable state and outliers
    unique_macros = sorted(m for m in np.unique(macro_labels) if m >= 0)
    macro_frames = {}
    for m in unique_macros:
        macro_frames[f"macro_{m}"] = np.where(macro_labels == m)[0]
    macro_frames["outliers"] = np.where(macro_labels == -1)[0]
    np.savez(prefix + "_macro_frames.npz", **macro_frames)

    print("[INFO] Saved arrays:")
    print(f"  {prefix}_features.npy")
    print(f"  {prefix}_vamp_Y.npy")
    print(f"  {prefix}_vamp_singular_values.npy")
    print(f"  {prefix}_micro_dtraj.npy")
    print(f"  {prefix}_micro_centers.npy")
    print(f"  {prefix}_msm_eigenvalues.npy")
    print(f"  {prefix}_pcca_memberships.npy")
    print(f"  {prefix}_macro_dtraj.npy")
    print(f"  {prefix}_macro_frames.npz")

    # Representative PDBs per non-outlier macrostate
    rep_frames = choose_representative_frames(Y, macro_labels)
    write_representative_pdbs(u, rep_frames, prefix)

    # Plots
    plot_singular_values(vamp_model.singular_values, prefix)
    plot_vamp_scatter(Y, micro_labels, prefix)
    plot_macro_timeseries(macro_labels, prefix)
    plot_msm_timescales(msm, frame_dt_ps=dt_ps, out_prefix=prefix)

    print("[INFO] Plots saved:")
    print(f"  {prefix}_vamp_singular_values.png")
    print(f"  {prefix}_vamp_scatter_micro.png")
    print(f"  {prefix}_macro_timeseries.png")
    print(f"  {prefix}_msm_timescales.png")


if __name__ == "__main__":
    main()
