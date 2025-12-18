from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alloskin.io.data import load_npz
from alloskin.simulation.potts_model import fit_potts_pmi, fit_potts_pseudolikelihood_torch
from alloskin.simulation.qubo import potts_to_qubo_onehot, decode_onehot
from alloskin.simulation.sampling import (
    gibbs_sample_potts,
    make_beta_ladder,
    replica_exchange_gibbs_potts,
    sa_sample_qubo_neal,
)
from alloskin.simulation.metrics import (
    marginals,
    pairwise_joints_on_edges,
    per_residue_js,
    combined_distance,
)


def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", required=True)
    ap.add_argument("--unassigned-policy", default="drop_frames", choices=["drop_frames", "treat_as_state", "error"])

    ap.add_argument("--fit", default="pmi", choices=["pmi", "plm", "pmi+plm"])
    ap.add_argument("--beta", type=float, default=1.0)

    # Gibbs / REX-Gibbs
    ap.add_argument("--gibbs-method", default="single", choices=["single", "rex"])
    ap.add_argument("--gibbs-samples", type=int, default=500, help="How many Potts samples to collect from Gibbs (the returned sample count).")
    ap.add_argument("--gibbs-burnin", type=int, default=50, help="How many initial Gibbs sweeps to discard (let the chain forget initialization).")
    ap.add_argument("--gibbs-thin", type=int, default=2, help="Keep one sample every thin sweeps after burn-in (helps reduce correlation).")

    # Replica exchange controls (only used if --gibbs-method rex OR for beta_eff scan)
    ap.add_argument("--rex-betas", type=str, default="", help="Comma-separated betas (ascending), e.g. 0.2,0.3,0.5,0.8,1.0")
    ap.add_argument("--rex-n-replicas", type=int, default=8, help="Number of betas (replicas) when auto-constructing the ladder.")
    ap.add_argument("--rex-beta-min", type=float, default=0.2, help="Minimum beta in the ladder (hottest replica).")
    ap.add_argument("--rex-beta-max", type=float, default=1.0, help="Maximum beta in the ladder (coldest replica).")
    ap.add_argument("--rex-spacing", type=str, default="geom", choices=["geom", "lin"], help="How betas are spaced: geom (geometric): usually better for tempering; lin (linear): sometimes fine for narrow ranges.")
    
    ap.add_argument("--rex-rounds", type=int, default=2000, help="Number of replica-exchange rounds. Each round does: 1) local Gibbs sweeps in each replica; 2) swap attempts between adjacent replicas.")
    ap.add_argument("--rex-burnin-rounds", type=int, default=50, help="Number of initial rounds discarded before saving samples.")
    ap.add_argument("--rex-sweeps-per-round", type=int, default=2, help="How many Gibbs sweeps each replica does per round before swap attempts.")
    ap.add_argument("--rex-thin-rounds", type=int, default=1, help="Save samples every this many rounds after burn-in.")

    # SA/QUBO
    ap.add_argument("--sa-reads", type=int, default=2000, help="Number of independent SA runs (“reads”). Each read outputs one bitstring sample.")
    ap.add_argument("--sa-sweeps", type=int, default=2000, help="Number of sweeps per read. More sweeps = more annealing time.")
    ap.add_argument("--penalty-safety", type=float, default=3.0, help="Controls how strong the one-hot constraint penalties are in the QUBO. Higher = fewer invalid assignments, but can make the QUBO landscape harder.")
    ap.add_argument("--repair", type=str, default="none", choices=["none", "argmax"], help="What to do when a QUBO bitstring violates one-hot constraints: none: decode invalid slices as “invalid” (still assigns label 0, but validity is tracked; best for honesty). argmax: forcibly repair each residue by picking the largest bit (hides violations but produces a valid label vector).")

    # beta_eff estimation
    ap.add_argument("--estimate-beta-eff", action="store_true", help="If set, the script estimates 𝛽_eff such that Gibbs samples at 𝛽_eff are closest to SA samples.")
    ap.add_argument("--beta-eff-grid", type=str, default="", help="Comma-separated betas to scan. Default: use rex-betas/ladder.")
    ap.add_argument("--beta-eff-w-marg", type=float, default=1.0, help="Weight of marginal-distribution mismatch (per-residue JS divergence) in the distance function.")
    ap.add_argument("--beta-eff-w-pair", type=float, default=1.0, help="Weight of pairwise-on-edges mismatch in the distance function.")
    ap.add_argument("--beta-eff-plot", type=str, default="", help="If set, writes an HTML plot of the distance curve D(β) to this path.")

    # plotting
    ap.add_argument("--plot-path", type=str, default="", help="If provided, writes an HTML “marginals comparison” dashboard (MD vs Gibbs vs SA).")
    ap.add_argument("--annotate-plots", action="store_true", help="If set, adds extra annotations to plots (depends on your plotting helper).")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true")

    args = ap.parse_args()

    ds = load_npz(args.npz, unassigned_policy=args.unassigned_policy)
    labels = ds.labels
    K = ds.cluster_counts
    edges = ds.edges

    print(f"[data] T={labels.shape[0]}  N={labels.shape[1]}  edges={len(edges)}")

    # Fit model(s)
    model = None
    if args.fit in ("pmi", "pmi+plm"):
        model_pmi = fit_potts_pmi(labels, K, edges)
        model = model_pmi
    if args.fit in ("plm", "pmi+plm"):
        model_plm = fit_potts_pseudolikelihood_torch(
            labels,
            K,
            edges,
            l2=1e-3,
            lr=1e-2,
            epochs=200,
            batch_size=512,
            seed=args.seed,
            verbose=True,
        )
        model = model_plm

    assert model is not None

    # --- Sampling baseline: Gibbs or REX-Gibbs at beta=args.beta ---
    if args.gibbs_method == "single":
        X_gibbs = gibbs_sample_potts(
            model,
            beta=args.beta,
            n_samples=args.gibbs_samples,
            burn_in=args.gibbs_burnin,
            thinning=args.gibbs_thin,
            seed=args.seed,
            progress=args.progress,
        )
        rex_info = None
    else:
        if args.rex_betas.strip():
            betas = _parse_float_list(args.rex_betas)
        else:
            betas = make_beta_ladder(
                beta_min=args.rex_beta_min,
                beta_max=args.rex_beta_max,
                n_replicas=args.rex_n_replicas,
                spacing=args.rex_spacing,
            )

        # ensure target beta is in ladder (append + sort if needed)
        if all(abs(b - args.beta) > 1e-12 for b in betas):
            betas = sorted(set(betas + [float(args.beta)]))

        rex_info = replica_exchange_gibbs_potts(
            model,
            betas=betas,
            sweeps_per_round=args.rex_sweeps_per_round,
            n_rounds=args.rex_rounds,
            burn_in_rounds=args.rex_burnin_rounds,
            thinning_rounds=args.rex_thin_rounds,
            seed=args.seed,
            progress=args.progress,
        )
        samples_by_beta = rex_info["samples_by_beta"]  # type: ignore
        X_gibbs = samples_by_beta[float(args.beta)]

        acc = rex_info["swap_accept_rate"]  # type: ignore
        print(f"[rex] betas={betas}")
        print(f"[rex] swap_accept_rate (adjacent): mean={float(np.mean(acc)):.3f}, min={float(np.min(acc)):.3f}, max={float(np.max(acc)):.3f}")

    # --- SA/QUBO sampling ---
    qubo = potts_to_qubo_onehot(model, beta=args.beta, penalty_safety=args.penalty_safety)

    Z_sa = sa_sample_qubo_neal(
        qubo,
        n_reads=args.sa_reads,
        sweeps=args.sa_sweeps,
        seed=args.seed,
        progress=args.progress,
    )

    repair = None if args.repair == "none" else args.repair
    X_sa = np.zeros((Z_sa.shape[0], len(qubo.var_slices)), dtype=int)
    valid_counts = np.zeros(Z_sa.shape[0], dtype=int)

    for i in range(Z_sa.shape[0]):
        x, valid = decode_onehot(Z_sa[i], qubo, repair=repair)
        X_sa[i] = x
        valid_counts[i] = int(valid.sum())

    viol = np.array([np.any(qubo.constraint_violations(z) != 0) for z in Z_sa], dtype=bool)
    print(f"[qubo] invalid_samples={viol.mean()*100:.2f}%  avg_valid_residues={valid_counts.mean():.1f}/{len(qubo.var_slices)}  repair={args.repair}")

    # --- Compare to MD ---
    p_md = marginals(labels, K)
    p_g = marginals(X_gibbs, K)
    p_sa = marginals(X_sa, K)

    js_g = per_residue_js(p_md, p_g)
    js_sa = per_residue_js(p_md, p_sa)

    print(f"[marginals] JS(MD, Gibbs@beta={args.beta}): mean={js_g.mean():.4f}  median={np.median(js_g):.4f}  max={js_g.max():.4f}")
    print(f"[marginals] JS(MD, SA-QUBO):          mean={js_sa.mean():.4f}  median={np.median(js_sa):.4f}  max={js_sa.max():.4f}")

    # Optional: pairwise summary
    if len(edges) > 0:
        P_md = pairwise_joints_on_edges(labels, K, edges)
        P_g = pairwise_joints_on_edges(X_gibbs, K, edges)
        P_sa = pairwise_joints_on_edges(X_sa, K, edges)

        # mean over edges via combined_distance pair term
        # (reuse combined_distance with only pair term by passing w_marg=0)
        js_pair_g = combined_distance(labels, X_gibbs, K=K, edges=edges, w_marg=0.0, w_pair=1.0)
        js_pair_sa = combined_distance(labels, X_sa, K=K, edges=edges, w_marg=0.0, w_pair=1.0)
        print(f"[pairs]   JS(MD, Gibbs) over edges: {js_pair_g:.4f}")
        print(f"[pairs]   JS(MD, SA-QUBO) over edges: {js_pair_sa:.4f}")

    # --- Estimate beta_eff for SA (optional) ---
    if args.estimate_beta_eff:
        if args.beta_eff_grid.strip():
            grid = _parse_float_list(args.beta_eff_grid)
            grid = sorted(set(grid))
        else:
            # default: use the same ladder as rex if available, else construct one around args.beta
            if args.rex_betas.strip():
                grid = sorted(set(_parse_float_list(args.rex_betas)))
            else:
                grid = make_beta_ladder(
                    beta_min=min(args.rex_beta_min, args.beta / 5.0),
                    beta_max=max(args.rex_beta_max, args.beta),
                    n_replicas=max(args.rex_n_replicas, 8),
                    spacing=args.rex_spacing,
                )
                if all(abs(b - args.beta) > 1e-12 for b in grid):
                    grid = sorted(set(grid + [float(args.beta)]))

        print(f"[beta_eff] scanning betas={grid}")

        # Get reference Gibbs samples for each beta in grid:
        # Use replica exchange across 'grid' for efficiency.
        rex_scan = replica_exchange_gibbs_potts(
            model,
            betas=grid,
            sweeps_per_round=args.rex_sweeps_per_round,
            n_rounds=args.rex_rounds,
            burn_in_rounds=args.rex_burnin_rounds,
            thinning_rounds=args.rex_thin_rounds,
            seed=args.seed + 123,
            progress=args.progress,
        )
        ref = rex_scan["samples_by_beta"]  # type: ignore

        distances = []
        for b in grid:
            X_ref = ref[float(b)]
            d = combined_distance(
                X_sa,
                X_ref,
                K=K,
                edges=edges,
                w_marg=args.beta_eff_w_marg,
                w_pair=args.beta_eff_w_pair,
            )
            distances.append(d)

        b_eff = grid[int(np.argmin(distances))]
        print(f"[beta_eff] beta_eff={b_eff:.6g}  (min distance={min(distances):.6g})")
        # print a compact table
        for b, d in zip(grid, distances):
            mark = "*" if abs(b - b_eff) < 1e-12 else " "
            print(f"[beta_eff] {mark} beta={b:10.6g}  D={d:.6g}")

        if args.beta_eff_plot.strip():
            from alloskin.simulation.plotting import plot_beta_scan_curve
            outp = plot_beta_scan_curve(betas=grid, distances=distances, out_path=args.beta_eff_plot)
            print(f"[beta_eff] saved D(beta) plot to {outp}")

    # --- Plot marginals dashboard (optional) ---
    if args.plot_path:
        from alloskin.simulation.plotting import plot_marginal_summary

        out_path = plot_marginal_summary(
            p_md=p_md,
            p_gibbs=p_g,
            p_sa=p_sa,
            js_gibbs=js_g,
            js_sa=js_sa,
            residue_labels=getattr(ds, "residue_keys", np.arange(len(p_md))),
            out_path=args.plot_path,
            annotate=args.annotate_plots,
        )
        print(f"[plot] saved marginal comparison to {out_path}")

    print("[done]")


if __name__ == "__main__":
    main()
