from __future__ import annotations

import argparse
import numpy as np

from alloskin.io.data import load_npz
from alloskin.simulation.potts_model import fit_potts_pmi, fit_potts_pseudolikelihood_torch
from alloskin.simulation.qubo import potts_to_qubo_onehot, decode_onehot
from alloskin.simulation.sampling import gibbs_sample_potts, sa_sample_qubo_neal
from alloskin.simulation.metrics import marginals, pairwise_joints_on_edges, per_residue_js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--unassigned-policy", default="drop_frames", choices=["drop_frames", "treat_as_state", "error"])

    ap.add_argument("--fit", default="pmi", choices=["pmi", "plm", "pmi+plm"])
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--gibbs-samples", type=int, default=500)
    ap.add_argument("--gibbs-burnin", type=int, default=500)
    ap.add_argument("--gibbs-thin", type=int, default=2)

    ap.add_argument("--sa-reads", type=int, default=500)
    ap.add_argument("--sa-sweeps", type=int, default=10)
    ap.add_argument("--sa-tstart", type=float, default=10.0)
    ap.add_argument("--sa-tend", type=float, default=0.1)

    ap.add_argument("--penalty-safety", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--progress", dest="progress", action="store_true", help="Show progress bars during sampling.")
    ap.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars during sampling.")
    ap.add_argument("--plot", dest="plot_path", help="Save interactive marginal comparison HTML to this path (e.g., outputs/marginals.html).")
    ap.add_argument("--no-annotate", dest="annotate_plots", action="store_false", help="Disable per-cell text annotations on marginal heatmaps.")
    ap.set_defaults(annotate_plots=True, progress=True)

    args = ap.parse_args()

    ds = load_npz(args.npz, unassigned_policy=args.unassigned_policy)
    labels = ds.labels
    K = ds.cluster_counts
    edges = ds.edges

    print(f"[data] T={labels.shape[0]}  N={labels.shape[1]}  edges={len(edges)}")

    if args.fit == "pmi":
        model = fit_potts_pmi(labels, K, edges)
    else:
        model = fit_potts_pseudolikelihood_torch(
            labels, K, edges,
            l2=1e-3, lr=1e-3, epochs=100,
            batch_size=512, seed=args.seed,
            init_from_pmi=args.fit == "pmi+plm",
            verbose=True
        )

    # Baseline: Gibbs on Potts
    X_gibbs = gibbs_sample_potts(
        model,
        beta=args.beta,
        n_samples=args.gibbs_samples,
        burn_in=args.gibbs_burnin,
        thinning=args.gibbs_thin,
        seed=args.seed,
        progress=args.progress,
    )

    # QUBO + SA (QA-proxy)
    qubo = potts_to_qubo_onehot(model, beta=args.beta, penalty_safety=args.penalty_safety)
    Z_sa = sa_sample_qubo_neal(
        qubo,
        n_reads=args.sa_reads,
        sweeps=args.sa_sweeps,
        t_start=args.sa_tstart,
        t_end=args.sa_tend,
        seed=args.seed,
        progress=args.progress,
    )
    X_sa = np.zeros((Z_sa.shape[0], len(qubo.var_slices)), dtype=int)
    valid_counts = np.zeros(Z_sa.shape[0], dtype=int)
    for i in range(Z_sa.shape[0]):
        x, valid = decode_onehot(Z_sa[i], qubo, repair=None)
        X_sa[i] = x
        valid_counts[i] = int(valid.sum())

    # Report constraint violations
    viol = np.array([np.any(qubo.constraint_violations(z) != 0) for z in Z_sa], dtype=bool)
    print(f"[qubo] invalid_samples={viol.mean()*100:.2f}%  avg_valid_residues={valid_counts.mean():.1f}/{len(qubo.var_slices)}")
    if viol.mean() > 0.01:
        print("       (If this is high: increase penalty_safety or use repair='argmax' and report it explicitly.)")

    # Compare marginals
    p_md = marginals(labels, K)
    p_g = marginals(X_gibbs, K)
    p_sa = marginals(X_sa, K)

    js_g = per_residue_js(p_md, p_g)
    js_sa = per_residue_js(p_md, p_sa)

    print(f"[marginals] JS(MD, Gibbs): mean={js_g.mean():.4f}  median={np.median(js_g):.4f}  max={js_g.max():.4f}")
    print(f"[marginals] JS(MD, SA-QUBO): mean={js_sa.mean():.4f}  median={np.median(js_sa):.4f}  max={js_sa.max():.4f}")

    # Optional: compare pairwise joints on edges (coarser, but important)
    P_md = pairwise_joints_on_edges(labels, K, edges)
    P_g = pairwise_joints_on_edges(X_gibbs, K, edges)
    P_sa = pairwise_joints_on_edges(X_sa, K, edges)

    def avg_js_pair(Pa, Pb):
        vals = []
        for e in edges:
            vals.append(float(per_residue_js([Pa[e].ravel()], [Pb[e].ravel()])[0]))
        return float(np.mean(vals))

    print(f"[pairs] avg JS(MD, Gibbs) over edges: {avg_js_pair(P_md, P_g):.4f}")
    print(f"[pairs] avg JS(MD, SA-QUBO) over edges: {avg_js_pair(P_md, P_sa):.4f}")

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
