from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.sampling_run import run_sampling
from phase.scripts.potts_utils import persist_sample
from phase.services.project_store import ProjectStore


def _differs_from_default(parser, key: str, value: object) -> bool:
    try:
        default = parser.get_default(key)
    except Exception:
        return True
    if value == default:
        return False
    if isinstance(value, list) and isinstance(default, list):
        return value != default
    return True


def _filter_sampling_params(args: object, parser) -> dict:
    if not hasattr(args, "__dict__"):
        return {}
    raw = vars(args)
    sampling_method = raw.get("sampling_method") or "gibbs"
    gibbs_method = raw.get("gibbs_method") or "single"

    allow = {"sampling_method", "beta", "seed"}
    if sampling_method == "gibbs":
        allow |= {"gibbs_method"}
        if gibbs_method == "single":
            allow |= {"gibbs_samples", "gibbs_burnin", "gibbs_thin", "gibbs_chains"}
        else:
            allow |= {
                "rex_betas",
                "rex_n_replicas",
                "rex_beta_min",
                "rex_beta_max",
                "rex_spacing",
                "rex_rounds",
                "rex_burnin_rounds",
                "rex_sweeps_per_round",
                "rex_thin_rounds",
                "rex_chains",
            }
    else:
        allow |= {
            "md_sample_id",
            "sa_reads",
            "sa_chains",
            "sa_sweeps",
            "sa_beta_hot",
            "sa_beta_cold",
            "sa_schedule_type",
            "sa_custom_beta_schedule",
            "sa_num_sweeps_per_beta",
            "sa_randomize_order",
            "sa_acceptance_criteria",
            "sa_init",
            "sa_init_md_frame",
            "sa_restart",
            "sa_md_state_ids",
            "penalty_safety",
            "repair",
        }

    out = {"sampling_method": sampling_method}
    if sampling_method == "gibbs":
        out["beta"] = raw.get("beta")
        out["gibbs_method"] = gibbs_method
    else:
        # Keep key SA parameters even when defaults are used (mirrors the Gibbs behavior).
        out["beta"] = raw.get("beta")
        out["sa_restart"] = raw.get("sa_restart") or "independent"
        out["sa_sweeps"] = raw.get("sa_sweeps")
    for key in allow:
        if key not in raw:
            continue
        val = raw.get(key)
        if val in (None, "", [], {}):
            continue
        if key in out:
            continue
        if not _differs_from_default(parser, key, val):
            continue
        out[key] = val
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample a fitted Potts model (Gibbs or SA) and persist a minimal sample.npz.")
    parser.add_argument("--npz", required=True, help="Input cluster NPZ file (for SA init + metadata linkage).")
    parser.add_argument("--results-dir", required=True, help="Output directory (will contain sample.npz).")
    parser.add_argument(
        "--model-npz",
        action="append",
        default=[],
        help="Potts model NPZ path. Repeat or pass comma-separated paths to combine multiple models.",
    )
    parser.add_argument("--sampling-method", default="gibbs", choices=["gibbs", "sa"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action="store_true")

    # Gibbs / REX-Gibbs
    parser.add_argument("--gibbs-method", default="single", choices=["single", "rex"])
    parser.add_argument("--gibbs-samples", type=int, default=500)
    parser.add_argument("--gibbs-burnin", type=int, default=50)
    parser.add_argument("--gibbs-thin", type=int, default=2)
    parser.add_argument("--gibbs-chains", type=int, default=1)
    parser.add_argument("--rex-betas", type=str, default="")
    parser.add_argument("--rex-n-replicas", type=int, default=8)
    parser.add_argument("--rex-beta-min", type=float, default=0.2)
    parser.add_argument("--rex-beta-max", type=float, default=1.0)
    parser.add_argument("--rex-spacing", type=str, default="geom", choices=["geom", "lin"])
    parser.add_argument("--rex-rounds", type=int, default=2000)
    parser.add_argument("--rex-burnin-rounds", type=int, default=50)
    parser.add_argument("--rex-sweeps-per-round", type=int, default=2)
    parser.add_argument("--rex-thin-rounds", type=int, default=1)
    parser.add_argument("--rex-chains", type=int, default=1)

    # SA/QUBO
    parser.add_argument("--sa-reads", type=int, default=2000)
    parser.add_argument("--sa-chains", type=int, default=1, help="Independent SA chains (processes). Total reads are split across chains.")
    parser.add_argument("--sa-sweeps", type=int, default=2000)
    parser.add_argument("--sa-beta-hot", type=float, default=None, help="Default: 0.01, or 0 with --sa-beta-cold 0 for neal auto range.")
    parser.add_argument("--sa-beta-cold", type=float, default=None, help="Default: 2.0, or 0 with --sa-beta-hot 0 for neal auto range.")
    parser.add_argument("--sa-schedule-type", type=str, default="geometric", choices=["geometric", "linear", "custom"])
    parser.add_argument("--sa-custom-beta-schedule", type=str, default="", help="Comma-separated custom beta schedule.")
    parser.add_argument("--sa-num-sweeps-per-beta", type=int, default=2)
    parser.add_argument(
        "--sa-randomize-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize variable update order within each sweep (recommended default).",
    )
    parser.add_argument("--sa-acceptance-criteria", type=str, default="Metropolis", choices=["Metropolis", "Gibbs"])
    parser.add_argument("--sa-init", type=str, default="md", choices=["md", "md-frame", "random-h", "random-uniform"])
    parser.add_argument("--sa-init-md-frame", type=int, default=-1)
    parser.add_argument(
        "--sa-restart",
        type=str,
        default="independent",
        choices=["previous", "md", "independent"],
        help="How to initialize each SA read after the first within a chain.",
    )
    parser.add_argument(
        "--sa-md-state-ids",
        type=str,
        default="",
        help="Comma-separated state IDs to restrict MD frames used for SA init (when using md/md-frame restart/init).",
    )
    parser.add_argument("--penalty-safety", type=float, default=4.0)
    parser.add_argument("--repair", type=str, default="none", choices=["none", "argmax"])

    parser.add_argument("--project-id", default="")
    parser.add_argument("--system-id", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--root", default="", help="Offline data root containing projects/.")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--sample-name", default="")
    parser.add_argument("--md-sample-id", default="", help="md_eval sample id used as the SA warm-start pool.")
    args = parser.parse_args(argv)
    last_progress = {"message": None, "pct": -1}

    def progress_cb(message: str, pct: int) -> None:
        if not bool(args.progress):
            return
        pct = int(max(0, min(100, pct)))
        if last_progress["message"] == message and last_progress["pct"] == pct:
            return
        last_progress["message"] = message
        last_progress["pct"] = pct
        print(f"[sampling] {message} ({pct}%)")

    sa_md_sample_npz = ""
    if str(args.sampling_method) == "sa":
        md_sample_id = str(args.md_sample_id or "").strip()
        if not md_sample_id:
            raise SystemExit("--md-sample-id is required for SA sampling.")
        project_id = str(args.project_id or "").strip()
        system_id = str(args.system_id or "").strip()
        cluster_id = str(args.cluster_id or "").strip()
        if not (project_id and system_id and cluster_id):
            raise SystemExit("--project-id, --system-id, and --cluster-id are required to resolve --md-sample-id.")
        root = Path(args.root) if str(args.root or "").strip() else Path(os.environ.get("PHASE_DATA_ROOT", ""))
        if not str(root).strip():
            raise SystemExit("--root or PHASE_DATA_ROOT is required to resolve --md-sample-id.")
        store = ProjectStore(base_dir=root / "projects")
        system_meta = store.get_system(project_id, system_id)
        entry = next((c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id), None)
        if not entry:
            raise SystemExit(f"Cluster not found: {cluster_id}")
        sample_entry = next(
            (
                s for s in (entry.get("samples") or [])
                if isinstance(s, dict) and str(s.get("sample_id") or "").strip() == md_sample_id
            ),
            None,
        )
        if not sample_entry:
            raise SystemExit(f"MD sample not found on cluster: {md_sample_id}")
        if str(sample_entry.get("type") or "").strip() != "md_eval":
            raise SystemExit(f"Selected sample is not md_eval: {md_sample_id}")
        rel = ((sample_entry.get("paths") or {}).get("summary_npz") or sample_entry.get("path"))
        if not rel:
            raise SystemExit(f"MD sample NPZ path missing: {md_sample_id}")
        sample_path = Path(str(rel))
        if not sample_path.is_absolute():
            sample_path = store.resolve_path(project_id, system_id, str(rel))
        if not sample_path.exists():
            raise SystemExit(f"MD sample NPZ missing on disk: {sample_path}")
        sa_md_sample_npz = str(sample_path)

    custom_sa_schedule = str(args.sa_custom_beta_schedule or "").strip()
    if args.sa_beta_hot is None and args.sa_beta_cold is None:
        sa_beta_hot = 0.0 if custom_sa_schedule else 0.01
        sa_beta_cold = 0.0 if custom_sa_schedule else 2.0
    elif args.sa_beta_hot is None or args.sa_beta_cold is None:
        raise SystemExit("Provide both --sa-beta-hot and --sa-beta-cold, or neither.")
    else:
        sa_beta_hot = float(args.sa_beta_hot)
        sa_beta_cold = float(args.sa_beta_cold)
    args.sa_beta_hot = sa_beta_hot
    args.sa_beta_cold = sa_beta_cold

    try:
        results = run_sampling(
            cluster_npz=str(args.npz),
            sa_md_sample_npz=sa_md_sample_npz,
            results_dir=str(args.results_dir),
            model_npz=getattr(args, "model_npz", []) or [],
            sampling_method=str(args.sampling_method),
            beta=float(args.beta),
            seed=int(args.seed),
            progress=bool(args.progress),
            gibbs_method=str(args.gibbs_method),
            gibbs_samples=int(args.gibbs_samples),
            gibbs_burnin=int(args.gibbs_burnin),
            gibbs_thin=int(args.gibbs_thin),
            gibbs_chains=int(args.gibbs_chains),
            rex_betas=str(args.rex_betas),
            rex_n_replicas=int(args.rex_n_replicas),
            rex_beta_min=float(args.rex_beta_min),
            rex_beta_max=float(args.rex_beta_max),
            rex_spacing=str(args.rex_spacing),
            rex_rounds=int(args.rex_rounds),
            rex_burnin_rounds=int(args.rex_burnin_rounds),
            rex_sweeps_per_round=int(args.rex_sweeps_per_round),
            rex_thin_rounds=int(args.rex_thin_rounds),
            rex_chains=int(args.rex_chains),
            sa_reads=int(args.sa_reads),
            sa_chains=int(args.sa_chains),
            sa_sweeps=int(args.sa_sweeps),
            sa_beta_hot=sa_beta_hot,
            sa_beta_cold=sa_beta_cold,
            sa_schedule_type=str(args.sa_schedule_type),
            sa_custom_beta_schedule=str(args.sa_custom_beta_schedule),
            sa_num_sweeps_per_beta=int(args.sa_num_sweeps_per_beta),
            sa_randomize_order=bool(args.sa_randomize_order),
            sa_acceptance_criteria=str(args.sa_acceptance_criteria),
            sa_init=str(args.sa_init),
            sa_init_md_frame=int(args.sa_init_md_frame),
            sa_restart=str(args.sa_restart),
            sa_md_sample_id=str(args.md_sample_id),
            sa_md_state_ids=str(args.sa_md_state_ids),
            penalty_safety=float(args.penalty_safety),
            repair=str(args.repair),
            progress_callback=progress_cb if bool(args.progress) else None,
        )
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    project_id = (args.project_id or "").strip()
    system_id = (args.system_id or "").strip()
    cluster_id = (args.cluster_id or "").strip()
    if project_id and system_id and cluster_id and results:
        summary = Path(results.sample_path)
        model_paths = [Path(str(raw)) for raw in getattr(args, "model_npz", []) or []]
        persist_sample(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            summary_path=summary,
            metadata_path=None,
            sample_name=args.sample_name or None,
            sample_type="potts_sampling",
            method=str(args.sampling_method) if getattr(args, "sampling_method", None) else None,
            params=_filter_sampling_params(args, parser),
            model_paths=model_paths,
            sample_id=args.sample_id or None,
            sa_diagnostics=results.sa_diagnostics,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
