#!/usr/bin/env python3
"""
Toy SA debug script.

Builds a small random Potts model, fabricates MD-like labels, and compares
SA energy distributions with and without warm-starts or restarts.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Sequence

import numpy as np

from phase.potts.potts_model import PottsModel
from phase.potts.qubo import potts_to_qubo_onehot, decode_onehot, encode_onehot
from phase.potts.sampling import sa_sample_qubo_neal


def build_toy_model(
    n_res: int,
    k: int,
    edge_prob: float,
    h_scale: float,
    j_scale: float,
    seed: int,
) -> PottsModel:
    rng = np.random.default_rng(seed)
    h = [rng.normal(scale=h_scale, size=k) for _ in range(n_res)]
    edges = []
    J = {}
    for r in range(n_res):
        for s in range(r + 1, n_res):
            if rng.random() < edge_prob:
                edges.append((r, s))
                J[(r, s)] = rng.normal(scale=j_scale, size=(k, k))
    return PottsModel(h=h, J=J, edges=edges)


def greedy_state(model: PottsModel, iters: int = 5) -> np.ndarray:
    x = np.array([int(np.argmin(hr)) for hr in model.h], dtype=int)
    n_res = len(model.h)
    for _ in range(iters):
        for r in range(n_res):
            best = x[r]
            best_e = None
            for k in range(len(model.h[r])):
                x[r] = k
                e = model.energy(x)
                if best_e is None or e < best_e:
                    best = k
                    best_e = e
            x[r] = best
    return x


def make_md_labels(
    base: np.ndarray,
    n_frames: int,
    flip_prob: float,
    rng: np.random.Generator,
    k_list: Sequence[int],
) -> np.ndarray:
    out = np.tile(base, (n_frames, 1))
    for i in range(n_frames):
        for r in range(out.shape[1]):
            if rng.random() < flip_prob:
                out[i, r] = rng.integers(0, int(k_list[r]))
    return out


def describe(label: str, energies: np.ndarray) -> None:
    if energies.size == 0:
        print(f"{label}: no samples")
        return
    print(
        f"{label}: n={energies.size} "
        f"mean={energies.mean():.3f} "
        f"median={np.median(energies):.3f} "
        f"min={energies.min():.3f} "
        f"max={energies.max():.3f}"
    )


def hamming_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    if a.shape != b.shape or a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    d = np.sum(a != b, axis=1).astype(float)
    return float(np.mean(d)), float(np.median(d)), float(np.max(d))


def energy_correlation(e0: np.ndarray, e1: np.ndarray) -> float:
    if e0.size == 0 or e1.size == 0 or e0.shape != e1.shape:
        return float("nan")
    if np.std(e0) < 1e-12 or np.std(e1) < 1e-12:
        return float("nan")
    return float(np.corrcoef(e0, e1)[0, 1])


def run_sa(
    *,
    model: PottsModel,
    qubo,
    n_reads: int,
    sweeps: int,
    seed: int,
    beta_range: tuple[float, float] | None,
    init_labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    init_states = encode_onehot(init_labels, qubo) if init_labels is not None else None
    Z = sa_sample_qubo_neal(
        qubo,
        n_reads=n_reads,
        sweeps=sweeps,
        seed=seed,
        beta_range=beta_range,
        initial_states=init_states,
    )
    X = np.zeros((Z.shape[0], len(qubo.var_slices)), dtype=int)
    for i in range(Z.shape[0]):
        x, _ = decode_onehot(Z[i], qubo, repair="argmax")
        X[i] = x
    E = model.energy_batch(X)
    return E, X


CSV_FIELDS = [
    "timestamp",
    "n_res",
    "k",
    "edge_prob",
    "h_scale",
    "j_scale",
    "beta",
    "beta_hot",
    "beta_cold",
    "schedule_mode",
    "init_mode",
    "restart_mode",
    "run_label",
    "reads",
    "sweeps",
    "seed",
    "md_frames",
    "md_flip_prob",
    "restart_topk",
    "init_mean",
    "init_std",
    "init_median",
    "init_min",
    "init_max",
    "final_mean",
    "final_std",
    "final_median",
    "final_min",
    "final_max",
    "ham_mean",
    "ham_median",
    "ham_max",
    "energy_corr",
]


def _stats(energies: np.ndarray) -> tuple[float, float, float, float, float]:
    if energies.size == 0:
        return (float("nan"),) * 5
    return (
        float(np.mean(energies)),
        float(np.std(energies)),
        float(np.median(energies)),
        float(np.min(energies)),
        float(np.max(energies)),
    )


def resolve_csv_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    try:
        with open(path, "r", newline="") as f:
            first = f.readline().strip()
    except Exception:
        return path
    header = [h.strip() for h in first.split(",")] if first else []
    if header == CSV_FIELDS:
        return path
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".csv"
    new_path = f"{base}_v2{ext}"
    if not os.path.exists(new_path):
        print(f"[csv] Existing header differs; writing to {new_path}")
    return new_path


def append_csv(path: str, row: dict) -> str:
    path = resolve_csv_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def analyze_csv(path: str) -> None:
    if not os.path.exists(path):
        print(f"No CSV found at {path}")
        return
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        print("CSV is empty.")
        return

    def to_float(val: str) -> float:
        try:
            return float(val)
        except Exception:
            return float("nan")

    if "final_mean" in rows[0]:
        grouped = {}
        for row in rows:
            sweeps = to_float(row.get("sweeps", "nan"))
            hot = to_float(row.get("beta_hot", "nan"))
            cold = to_float(row.get("beta_cold", "nan"))
            schedule = row.get("schedule_mode", "custom")
            init_mode = row.get("init_mode", "md")
            restart_mode = row.get("restart_mode", "baseline")
            key = (schedule, hot, cold)
            line_key = f"{init_mode}-{restart_mode}"
            grouped.setdefault(key, {})
            grouped[key].setdefault(line_key, {})
            grouped[key][line_key].setdefault(
                sweeps,
                {
                    "final_means": [],
                    "final_stds": [],
                    "final_mins": [],
                    "final_maxs": [],
                    "init_means": [],
                    "init_stds": [],
                    "init_mins": [],
                    "init_maxs": [],
                    "ham_means": [],
                    "corrs": [],
                },
            )
            slot = grouped[key][line_key][sweeps]
            slot["final_means"].append(to_float(row.get("final_mean", "nan")))
            slot["final_stds"].append(to_float(row.get("final_std", "nan")))
            slot["final_mins"].append(to_float(row.get("final_min", "nan")))
            slot["final_maxs"].append(to_float(row.get("final_max", "nan")))
            slot["init_means"].append(to_float(row.get("init_mean", "nan")))
            slot["init_stds"].append(to_float(row.get("init_std", "nan")))
            slot["init_mins"].append(to_float(row.get("init_min", "nan")))
            slot["init_maxs"].append(to_float(row.get("init_max", "nan")))
            slot["ham_means"].append(to_float(row.get("ham_mean", "nan")))
            slot["corrs"].append(to_float(row.get("energy_corr", "nan")))

        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; summary by sweeps:")
            for key, lines in grouped.items():
                schedule, hot, cold = key
                print(f"{schedule} hot={hot} cold={cold}")
                for line_key, sweeps_map in lines.items():
                    for sweeps, vals in sorted(sweeps_map.items()):
                        fm = np.nanmean(vals["final_means"]) if vals["final_means"] else float("nan")
                        fs = np.nanmean(vals["final_stds"]) if vals["final_stds"] else float("nan")
                        fmin = np.nanmean(vals["final_mins"]) if vals["final_mins"] else float("nan")
                        fmax = np.nanmean(vals["final_maxs"]) if vals["final_maxs"] else float("nan")
                        hm = np.nanmean(vals["ham_means"]) if vals["ham_means"] else float("nan")
                        cr = np.nanmean(vals["corrs"]) if vals["corrs"] else float("nan")
                        print(
                            f"  {line_key} sweeps={sweeps}: mean={fm:.3f} std={fs:.3f} min={fmin:.3f} "
                            f"max={fmax:.3f} ham={hm:.3f} corr={cr:.3f}"
                        )
            return

        def _is_auto(hot_val: float, cold_val: float) -> bool:
            return (hot_val == 0.0 or np.isnan(hot_val)) and (cold_val == 0.0 or np.isnan(cold_val))

        for key, lines in grouped.items():
            schedule, hot, cold = key
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
            ax_energy = axes[0, 0]
            ax_init = axes[0, 1]
            ax_ham = axes[1, 0]
            ax_corr = axes[1, 1]
            for line_key, sweeps_map in lines.items():
                sweeps_sorted = sorted(sweeps_map.keys())
                f_means = [np.nanmean(sweeps_map[s]["final_means"]) for s in sweeps_sorted]
                f_stds = [np.nanmean(sweeps_map[s]["final_stds"]) for s in sweeps_sorted]
                f_mins = [np.nanmean(sweeps_map[s]["final_mins"]) for s in sweeps_sorted]
                f_maxs = [np.nanmean(sweeps_map[s]["final_maxs"]) for s in sweeps_sorted]
                i_means = [np.nanmean(sweeps_map[s]["init_means"]) for s in sweeps_sorted]
                i_stds = [np.nanmean(sweeps_map[s]["init_stds"]) for s in sweeps_sorted]
                i_mins = [np.nanmean(sweeps_map[s]["init_mins"]) for s in sweeps_sorted]
                i_maxs = [np.nanmean(sweeps_map[s]["init_maxs"]) for s in sweeps_sorted]
                ham = [np.nanmean(sweeps_map[s]["ham_means"]) for s in sweeps_sorted]
                corr = [np.nanmean(sweeps_map[s]["corrs"]) for s in sweeps_sorted]

                line = ax_energy.plot(sweeps_sorted, f_means, marker="o", linestyle="-", label=f"{line_key} final")
                color = line[0].get_color()
                ax_energy.errorbar(
                    sweeps_sorted,
                    f_means,
                    yerr=f_stds,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    capthick=1.2,
                    zorder=3,
                )
                if len(sweeps_sorted) > 1:
                    cap_w = min(np.diff(sweeps_sorted)) * 0.1
                else:
                    cap_w = max(0.2, sweeps_sorted[0] * 0.05 if sweeps_sorted[0] else 0.2)
                for x, y_min, y_max in zip(sweeps_sorted, f_mins, f_maxs):
                    ax_energy.vlines(x, y_min, y_max, alpha=0.5, linewidth=1.2, colors=color, zorder=2)
                    ax_energy.hlines([y_min, y_max], x - cap_w, x + cap_w, alpha=0.5, linewidth=1.2, colors=color, zorder=2)

                line_i = ax_init.plot(sweeps_sorted, i_means, marker="o", linestyle="-", label=f"{line_key} init")
                color_i = line_i[0].get_color()
                ax_init.errorbar(
                    sweeps_sorted,
                    i_means,
                    yerr=i_stds,
                    fmt="none",
                    ecolor=color_i,
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                )
                for x, y_min, y_max in zip(sweeps_sorted, i_mins, i_maxs):
                    ax_init.vlines(x, y_min, y_max, alpha=0.4, linewidth=1.0, colors=color_i)

                ax_ham.plot(sweeps_sorted, ham, marker="o", linestyle="-", label=line_key)
                ax_corr.plot(sweeps_sorted, corr, marker="o", linestyle="-", label=line_key)

            ax_energy.set_xlabel("Sweeps")
            ax_energy.set_ylabel("Final mean energy")
            ax_init.set_xlabel("Sweeps")
            ax_init.set_ylabel("Init mean energy")
            ax_ham.set_xlabel("Sweeps")
            ax_ham.set_ylabel("Hamming mean")
            ax_corr.set_xlabel("Sweeps")
            ax_corr.set_ylabel("Energy corr")

            if schedule == "auto" or _is_auto(hot, cold):
                title = "auto schedule"
            else:
                title = f"hot={hot} cold={cold}"
            fig.suptitle(title)
            ax_energy.legend(fontsize=8)
            ax_init.legend(fontsize=8)
            ax_ham.legend(fontsize=8)
            ax_corr.legend(fontsize=8)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(
                os.path.dirname(path) or ".",
                f"debug_sa_plot_{stamp}_{schedule}_hot{hot}_cold{cold}.png",
            )
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Saved plot: {out}")
        return

    grouped = {}
    for row in rows:
        sweeps = to_float(row.get("sweeps", "nan"))
        hot = to_float(row.get("beta_hot", "nan"))
        cold = to_float(row.get("beta_cold", "nan"))
        key = (hot, cold)
        grouped.setdefault(key, {})
        grouped[key].setdefault(sweeps, {"warm": [], "rand": [], "restart": []})
        grouped[key][sweeps]["warm"].append(to_float(row.get("warm_mean", "nan")))
        grouped[key][sweeps]["rand"].append(to_float(row.get("rand_mean", "nan")))
        grouped[key][sweeps]["restart"].append(to_float(row.get("restart_mean", "nan")))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; summary by sweeps:")
        for (hot, cold), sweeps_map in grouped.items():
            print(f"beta_hot={hot} beta_cold={cold}")
            for sweeps, vals in sorted(sweeps_map.items()):
                wm = np.nanmean(vals["warm"]) if vals["warm"] else float("nan")
                rm = np.nanmean(vals["rand"]) if vals["rand"] else float("nan")
                sm = np.nanmean(vals["restart"]) if vals["restart"] else float("nan")
                print(f"  sweeps={sweeps}: warm={wm:.3f} rand={rm:.3f} restart={sm:.3f}")
        return

    for (hot, cold), sweeps_map in grouped.items():
        sweeps_sorted = sorted(sweeps_map.keys())
        warm_means = [np.nanmean(sweeps_map[s]["warm"]) for s in sweeps_sorted]
        rand_means = [np.nanmean(sweeps_map[s]["rand"]) for s in sweeps_sorted]
        restart_means = [np.nanmean(sweeps_map[s]["restart"]) for s in sweeps_sorted]

        plt.figure(figsize=(6, 4))
        plt.plot(sweeps_sorted, warm_means, marker="o", label="warm-start")
        plt.plot(sweeps_sorted, rand_means, marker="o", label="random")
        if any(np.isfinite(v) for v in restart_means):
            plt.plot(sweeps_sorted, restart_means, marker="o", label="restart top-k")
        plt.xlabel("Sweeps")
        plt.ylabel("Mean energy")
        plt.title(f"SA mean energy vs sweeps (beta_hot={hot}, beta_cold={cold})")
        plt.legend()
        plt.tight_layout()
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.dirname(path) or ".", f"debug_sa_plot_{stamp}_hot{hot}_cold{cold}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved plot: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-res", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--edge-prob", type=float, default=0.35)
    ap.add_argument("--h-scale", type=float, default=1.0)
    ap.add_argument("--j-scale", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-hot", type=float, default=0.0)
    ap.add_argument("--beta-cold", type=float, default=0.0)
    ap.add_argument("--beta-hot-list", type=str, default="")
    ap.add_argument("--beta-cold-list", type=str, default="")
    ap.add_argument("--reads", type=int, default=500)
    ap.add_argument("--sweeps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md-frames", type=int, default=500)
    ap.add_argument("--md-flip-prob", type=float, default=0.08)
    ap.add_argument("--restart-topk", type=int, default=100)
    ap.add_argument("--csv", type=str, default="scripts/debug_sa_results.csv")
    ap.add_argument("--analyze", action="store_true", help="Analyze CSV and plot results.")
    ap.add_argument("--also-auto", action="store_true", help="Also run an auto beta schedule alongside custom.")
    args = ap.parse_args()

    if args.analyze:
        analyze_csv(args.csv)
        return

    model = build_toy_model(
        n_res=args.n_res,
        k=args.k,
        edge_prob=args.edge_prob,
        h_scale=args.h_scale,
        j_scale=args.j_scale,
        seed=args.seed,
    )
    qubo = potts_to_qubo_onehot(model, beta=float(args.beta))
    rng = np.random.default_rng(args.seed + 1)

    low_x = greedy_state(model)
    md_labels = make_md_labels(
        low_x,
        n_frames=args.md_frames,
        flip_prob=args.md_flip_prob,
        rng=rng,
        k_list=model.K_list(),
    )
    def random_labels(n: int) -> np.ndarray:
        out = np.zeros((n, len(model.h)), dtype=int)
        for r, k in enumerate(model.K_list()):
            out[:, r] = rng.integers(0, int(k), size=n)
        return out

    def _parse_list(raw: str) -> list[float]:
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [float(p) for p in parts]

    hot_list = _parse_list(args.beta_hot_list)
    cold_list = _parse_list(args.beta_cold_list)
    if hot_list or cold_list:
        if len(hot_list) != len(cold_list):
            raise ValueError("--beta-hot-list and --beta-cold-list must have the same length.")

    schedule_modes = [("auto", None)]
    if args.beta_hot > 0 and args.beta_cold > 0:
        schedule_modes.append(("custom", (float(args.beta_hot), float(args.beta_cold))))
    for hot, cold in zip(hot_list, cold_list):
        schedule_modes.append(("custom", (float(hot), float(cold))))

    used_path = None
    for schedule_mode, beta_range in schedule_modes:
        print(f"== schedule={schedule_mode} beta_range={beta_range} ==")
        schedule_hot = args.beta_hot
        schedule_cold = args.beta_cold
        if schedule_mode == "auto" or beta_range is None:
            schedule_hot = 0.0
            schedule_cold = 0.0
        elif beta_range is not None:
            schedule_hot = float(beta_range[0])
            schedule_cold = float(beta_range[1])

        for init_mode in ("warm", "random"):
            if init_mode == "warm":
                init_idx = rng.integers(0, md_labels.shape[0], size=args.reads)
                base_init = md_labels[init_idx]
            else:
                base_init = random_labels(args.reads)
            init_energy = model.energy_batch(base_init)
            describe(f"init ({init_mode})", init_energy)

            e_base, x_base = run_sa(
                model=model,
                qubo=qubo,
                n_reads=args.reads,
                sweeps=args.sweeps,
                seed=args.seed,
                beta_range=beta_range,
                init_labels=base_init,
            )
            describe(f"SA {init_mode} baseline", e_base)
            mean_hd, med_hd, max_hd = hamming_stats(base_init, x_base)
            corr = energy_correlation(init_energy, e_base)
            print(
                f"  baseline Hamming to init: mean={mean_hd:.2f} median={med_hd:.2f} max={max_hd:.2f} "
                f"(N={base_init.shape[1]})"
            )
            print(f"  baseline energy corr(init, final) = {corr:.3f}")

            init_stats = _stats(init_energy)
            final_stats = _stats(e_base)
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "n_res": args.n_res,
                "k": args.k,
                "edge_prob": args.edge_prob,
                "h_scale": args.h_scale,
                "j_scale": args.j_scale,
                "beta": args.beta,
                "beta_hot": schedule_hot,
                "beta_cold": schedule_cold,
                "schedule_mode": schedule_mode,
                "init_mode": init_mode,
                "restart_mode": "baseline",
                "run_label": f"{init_mode}-baseline",
                "reads": args.reads,
                "sweeps": args.sweeps,
                "seed": args.seed,
                "md_frames": args.md_frames,
                "md_flip_prob": args.md_flip_prob,
                "restart_topk": args.restart_topk,
                "init_mean": init_stats[0],
                "init_std": init_stats[1],
                "init_median": init_stats[2],
                "init_min": init_stats[3],
                "init_max": init_stats[4],
                "final_mean": final_stats[0],
                "final_std": final_stats[1],
                "final_median": final_stats[2],
                "final_min": final_stats[3],
                "final_max": final_stats[4],
                "ham_mean": mean_hd,
                "ham_median": med_hd,
                "ham_max": max_hd,
                "energy_corr": corr,
            }
            used_path = append_csv(args.csv, row)

            for restart_mode in ("independent", "topk"):
                if restart_mode == "independent":
                    if init_mode == "warm":
                        idx2 = rng.integers(0, md_labels.shape[0], size=args.reads)
                        init2 = md_labels[idx2]
                    else:
                        init2 = random_labels(args.reads)
                else:
                    order = np.argsort(e_base)
                    topk = max(1, min(int(args.restart_topk), len(order)))
                    pool = x_base[order[:topk]]
                    idx2 = rng.integers(0, pool.shape[0], size=args.reads)
                    init2 = pool[idx2]

                e2, x2 = run_sa(
                    model=model,
                    qubo=qubo,
                    n_reads=args.reads,
                    sweeps=args.sweeps,
                    seed=args.seed + 13,
                    beta_range=beta_range,
                    init_labels=init2,
                )
                describe(f"SA {init_mode} + {restart_mode}", e2)
                init2_energy = model.energy_batch(init2)
                mean_hd2, med_hd2, max_hd2 = hamming_stats(init2, x2)
                corr2 = energy_correlation(init2_energy, e2)
                print(
                    f"  {restart_mode} Hamming to init: mean={mean_hd2:.2f} median={med_hd2:.2f} max={max_hd2:.2f} "
                    f"(N={init2.shape[1]})"
                )
                print(f"  {restart_mode} energy corr(init, final) = {corr2:.3f}")

                init_stats2 = _stats(init2_energy)
                final_stats2 = _stats(e2)
                row2 = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "n_res": args.n_res,
                    "k": args.k,
                    "edge_prob": args.edge_prob,
                    "h_scale": args.h_scale,
                    "j_scale": args.j_scale,
                    "beta": args.beta,
                    "beta_hot": schedule_hot,
                    "beta_cold": schedule_cold,
                    "schedule_mode": schedule_mode,
                    "init_mode": init_mode,
                    "restart_mode": restart_mode,
                    "run_label": f"{init_mode}-{restart_mode}",
                    "reads": args.reads,
                    "sweeps": args.sweeps,
                    "seed": args.seed,
                    "md_frames": args.md_frames,
                    "md_flip_prob": args.md_flip_prob,
                    "restart_topk": args.restart_topk,
                    "init_mean": init_stats2[0],
                    "init_std": init_stats2[1],
                    "init_median": init_stats2[2],
                    "init_min": init_stats2[3],
                    "init_max": init_stats2[4],
                    "final_mean": final_stats2[0],
                    "final_std": final_stats2[1],
                    "final_median": final_stats2[2],
                    "final_min": final_stats2[3],
                    "final_max": final_stats2[4],
                    "ham_mean": mean_hd2,
                    "ham_median": med_hd2,
                    "ham_max": max_hd2,
                    "energy_corr": corr2,
                }
                used_path = append_csv(args.csv, row2)

        if used_path:
            print(f"Appended results to {used_path}")


if __name__ == "__main__":
    main()
