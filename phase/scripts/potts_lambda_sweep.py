from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from phase.potts.analysis_run import compute_lambda_sweep_analysis
from phase.potts.potts_model import interpolate_potts_models, load_potts_model, zero_sum_gauge_model
from phase.potts.sample_io import save_sample_npz
from phase.potts.sampling import gibbs_sample_potts, make_beta_ladder, replica_exchange_gibbs_potts
from phase.services.project_store import ProjectStore


def _parse_float_list(raw: str) -> list[float]:
    parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
    return [float(p) for p in parts]


def _filter_sampling_params(raw: dict) -> dict:
    """
    Keep a minimal, stable config in metadata (skip defaults).
    Mirrors backend/tasks.py lambda-sweep defaults.
    """
    defaults = {
        "gibbs_samples": 500,
        "gibbs_burnin": 50,
        "gibbs_thin": 2,
        "rex_beta_min": 0.2,
        "rex_beta_max": 1.0,
        "rex_spacing": "geom",
        "rex_n_replicas": 8,
        "rex_rounds": 2000,
        "rex_burnin_rounds": 50,
        "rex_sweeps_per_round": 2,
        "rex_thin_rounds": 1,
    }

    out: dict = {"sampling_method": "gibbs"}

    def _maybe(key: str, value: object) -> None:
        if value in (None, "", [], {}):
            return
        if key in defaults and value == defaults[key]:
            return
        out[key] = value

    gm = str(raw.get("gibbs_method") or "rex").lower()
    if gm not in {"single", "rex"}:
        gm = "rex"
    _maybe("gibbs_method", gm)

    beta = float(raw.get("beta") or 1.0)
    _maybe("beta", beta)

    if gm == "single":
        _maybe("gibbs_samples", int(raw.get("gibbs_samples") or defaults["gibbs_samples"]))
        _maybe("gibbs_burnin", int(raw.get("gibbs_burnin") or defaults["gibbs_burnin"]))
        _maybe("gibbs_thin", int(raw.get("gibbs_thin") or defaults["gibbs_thin"]))
    else:
        rex_betas = raw.get("rex_betas")
        if isinstance(rex_betas, list):
            rex_betas = ",".join(str(v) for v in rex_betas)
        if isinstance(rex_betas, str) and rex_betas.strip():
            _maybe("rex_betas", str(rex_betas).strip())
        else:
            _maybe("rex_beta_min", float(raw.get("rex_beta_min") or defaults["rex_beta_min"]))
            _maybe("rex_beta_max", float(raw.get("rex_beta_max") or defaults["rex_beta_max"]))
            _maybe("rex_n_replicas", int(raw.get("rex_n_replicas") or defaults["rex_n_replicas"]))
            _maybe("rex_spacing", str(raw.get("rex_spacing") or defaults["rex_spacing"]))

        _maybe("rex_rounds", int(raw.get("rex_rounds") or defaults["rex_rounds"]))
        _maybe("rex_burnin_rounds", int(raw.get("rex_burnin_rounds") or defaults["rex_burnin_rounds"]))
        _maybe("rex_sweeps_per_round", int(raw.get("rex_sweeps_per_round") or defaults["rex_sweeps_per_round"]))
        _maybe("rex_thin_rounds", int(raw.get("rex_thin_rounds") or defaults["rex_thin_rounds"]))

    seed = raw.get("seed")
    if seed is not None:
        _maybe("seed", int(seed))
    return out


def _run_gibbs_chain_worker(payload: dict[str, object]) -> dict[str, object]:
    labels = gibbs_sample_potts(
        payload["model"],  # type: ignore[arg-type]
        beta=float(payload["beta"]),
        n_samples=int(payload["n_samples"]),
        burn_in=int(payload["burn_in"]),
        thinning=int(payload["thinning"]),
        seed=int(payload["seed"]),
        progress=bool(payload.get("progress", False)),
        progress_mode=str(payload.get("progress_mode", "samples")),
        progress_desc=str(payload.get("progress_desc", "Gibbs samples")),
        progress_position=payload.get("progress_position"),  # type: ignore[arg-type]
    )
    return {"labels": labels}


def _run_rex_chain_worker(payload: dict[str, object]) -> dict[str, object]:
    run = replica_exchange_gibbs_potts(
        payload["model"],  # type: ignore[arg-type]
        betas=payload["betas"],  # type: ignore[arg-type]
        sweeps_per_round=int(payload["sweeps_per_round"]),
        n_rounds=int(payload["n_rounds"]),
        burn_in_rounds=int(payload["burn_in_rounds"]),
        thinning_rounds=int(payload["thinning_rounds"]),
        seed=int(payload["seed"]),
        progress=bool(payload.get("progress", False)),
        progress_mode=str(payload.get("progress_mode", "samples")),
        progress_desc=str(payload.get("progress_desc", "REX samples")),
        progress_position=payload.get("progress_position"),  # type: ignore[arg-type]
    )
    return {"run": run}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lambda interpolation sweep between two Potts models (Gibbs sampling at each lambda), "
            "persisting each lambda sample as a correlated sample series and writing a dedicated analysis."
        )
    )
    parser.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--cluster-id", required=True)

    parser.add_argument("--model-a-id", required=True, help="Endpoint model A (lambda=1).")
    parser.add_argument("--model-b-id", required=True, help="Endpoint model B (lambda=0).")

    parser.add_argument("--md-sample-id-1", required=True)
    parser.add_argument("--md-sample-id-2", required=True)
    parser.add_argument("--md-sample-id-3", required=True)
    parser.add_argument("--md-label-mode", default="assigned", choices=["assigned", "halo"])

    parser.add_argument("--lambda-count", type=int, default=21)
    parser.add_argument("--series-id", default="", help="Optional series UUID (otherwise generated).")
    parser.add_argument("--series-label", default="", help="Display label for this sweep.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Node/edge mixing weight for match curve.")
    parser.add_argument("--keep-invalid", action="store_true", help="Keep frames with invalid labels (-1) in analysis.")

    # Gibbs / REX-Gibbs
    parser.add_argument("--gibbs-method", default="rex", choices=["single", "rex"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action="store_true", help="Show sampler progress bars (best-effort with multi-processing).")
    parser.add_argument("--gibbs-samples", type=int, default=500)
    parser.add_argument("--gibbs-burnin", type=int, default=50)
    parser.add_argument("--gibbs-thin", type=int, default=2)
    parser.add_argument("--gibbs-chains", type=int, default=1, help="Independent Gibbs chains (processes). Total samples are split across chains.")

    parser.add_argument("--rex-betas", type=str, default="")
    parser.add_argument("--rex-n-replicas", type=int, default=8)
    parser.add_argument("--rex-beta-min", type=float, default=0.2)
    parser.add_argument("--rex-beta-max", type=float, default=1.0)
    parser.add_argument("--rex-spacing", type=str, default="geom", choices=["geom", "lin"])
    parser.add_argument("--rex-rounds", type=int, default=2000)
    parser.add_argument("--rex-burnin-rounds", type=int, default=50)
    parser.add_argument("--rex-sweeps-per-round", type=int, default=2)
    parser.add_argument("--rex-thin-rounds", type=int, default=1)
    parser.add_argument("--rex-chains", type=int, default=1, help="Parallel replica-exchange chains (processes). Total rounds are split across chains.")

    args = parser.parse_args(argv)

    if args.model_a_id == args.model_b_id:
        raise SystemExit("Select two different endpoint models (model-a-id != model-b-id).")
    if args.lambda_count < 2:
        raise SystemExit("--lambda-count must be >= 2.")
    if not (0.0 <= float(args.alpha) <= 1.0):
        raise SystemExit("--alpha must be in [0,1].")

    md_ids = [str(args.md_sample_id_1), str(args.md_sample_id_2), str(args.md_sample_id_3)]
    if len(set(md_ids)) != 3:
        raise SystemExit("MD reference samples must be 3 distinct sample IDs.")

    # Store
    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    data_root = Path(root)
    os.environ["PHASE_DATA_ROOT"] = str(data_root)
    store = ProjectStore(base_dir=data_root / "projects")

    project_id = str(args.project_id)
    system_id = str(args.system_id)
    cluster_id = str(args.cluster_id)
    system_meta = store.get_system(project_id, system_id)
    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not isinstance(entry, dict):
        raise SystemExit(f"Cluster '{cluster_id}' not found in system metadata.")

    # Endpoint models (paths from cluster metadata)
    models_meta = entry.get("potts_models") or []
    model_a_meta = next((m for m in models_meta if m.get("model_id") == args.model_a_id), None)
    model_b_meta = next((m for m in models_meta if m.get("model_id") == args.model_b_id), None)
    if not isinstance(model_a_meta, dict) or not isinstance(model_b_meta, dict):
        raise SystemExit("Could not locate both endpoint models in cluster potts_models metadata.")
    for mid, meta in [(args.model_a_id, model_a_meta), (args.model_b_id, model_b_meta)]:
        params = meta.get("params") or {}
        if isinstance(params, dict):
            dk = str(params.get("delta_kind") or "").strip().lower()
            if dk.startswith("delta"):
                raise SystemExit(
                    f"Endpoint model {mid} appears delta-only (params.delta_kind={dk!r}). "
                    "Please select sampleable endpoint models (standard or combined)."
                )

    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    samples_root = cluster_dirs["samples_dir"]

    model_a_path = store.resolve_path(project_id, system_id, str(model_a_meta.get("path") or ""))
    model_b_path = store.resolve_path(project_id, system_id, str(model_b_meta.get("path") or ""))
    if not model_a_path.exists() or not model_b_path.exists():
        raise SystemExit("Endpoint model NPZ not found on disk.")

    endpoint_a = zero_sum_gauge_model(load_potts_model(str(model_a_path)))
    endpoint_b = zero_sum_gauge_model(load_potts_model(str(model_b_path)))

    # Sweep
    series_id = (str(args.series_id or "").strip() or str(uuid.uuid4()))
    ts = datetime.utcnow().strftime("%Y%m%d %H:%M")
    series_label = (str(args.series_label or "").strip() or f"Lambda sweep {ts}")

    lambdas = np.linspace(0.0, 1.0, int(args.lambda_count))
    sample_ids: list[str] = []
    sample_names: list[str] = []

    sampling_params = {
        "gibbs_method": str(args.gibbs_method),
        "beta": float(args.beta),
        "seed": int(args.seed),
        "gibbs_samples": int(args.gibbs_samples),
        "gibbs_burnin": int(args.gibbs_burnin),
        "gibbs_thin": int(args.gibbs_thin),
        "gibbs_chains": int(args.gibbs_chains),
        "rex_betas": str(args.rex_betas),
        "rex_n_replicas": int(args.rex_n_replicas),
        "rex_beta_min": float(args.rex_beta_min),
        "rex_beta_max": float(args.rex_beta_max),
        "rex_spacing": str(args.rex_spacing),
        "rex_rounds": int(args.rex_rounds),
        "rex_burnin_rounds": int(args.rex_burnin_rounds),
        "rex_sweeps_per_round": int(args.rex_sweeps_per_round),
        "rex_thin_rounds": int(args.rex_thin_rounds),
        "rex_chains": int(args.rex_chains),
    }

    for idx, lam in enumerate(lambdas):
        print(f"[lambda_sweep] sampling {idx + 1}/{len(lambdas)}  λ={float(lam):.3f}")
        model_lam = interpolate_potts_models(endpoint_b, endpoint_a, float(lam))
        sample_id = str(uuid.uuid4())
        sample_dir = samples_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        gm = str(args.gibbs_method).lower()
        beta = float(args.beta)
        base_seed = int(args.seed) + int(idx) * 10000  # deterministic but distinct per lambda
        if gm == "single":
            total_samples = max(0, int(args.gibbs_samples))
            n_chains = max(1, int(args.gibbs_chains))
            if n_chains > max(1, total_samples):
                n_chains = max(1, total_samples)
            base = total_samples // n_chains if n_chains else total_samples
            extra = total_samples % n_chains if n_chains else 0
            chain_samples = [base + (1 if i < extra else 0) for i in range(n_chains)]

            parts: list[np.ndarray] = []
            if n_chains <= 1:
                labels = gibbs_sample_potts(
                    model_lam,
                    beta=beta,
                    n_samples=int(total_samples),
                    burn_in=int(args.gibbs_burnin),
                    thinning=int(args.gibbs_thin),
                    seed=int(base_seed),
                    progress=bool(args.progress),
                    progress_mode="samples",
                    progress_desc=f"λ={float(lam):.3f} Gibbs",
                )
            else:
                with ProcessPoolExecutor(max_workers=n_chains) as executor:
                    futures = {}
                    for cidx, n_samp in enumerate(chain_samples):
                        if int(n_samp) <= 0:
                            continue
                        futures[executor.submit(
                            _run_gibbs_chain_worker,
                            {
                                "model": model_lam,
                                "beta": beta,
                                "n_samples": int(n_samp),
                                "burn_in": int(args.gibbs_burnin),
                                "thinning": int(args.gibbs_thin),
                                "seed": int(base_seed) + cidx,
                                "progress": False,  # progress bars across processes tend to garble output
                            },
                        )] = cidx
                    completed = 0
                    for fut in as_completed(futures):
                        out = fut.result()
                        arr = out.get("labels")
                        if isinstance(arr, np.ndarray) and arr.size:
                            parts.append(arr)
                        completed += 1
                        if bool(args.progress):
                            print(f"[lambda_sweep]   λ={float(lam):.3f} Gibbs chains {completed}/{n_chains}")
                labels = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(model_lam.h)), dtype=int)
        else:
            rex_betas_raw = str(args.rex_betas or "").strip()
            if rex_betas_raw:
                betas = _parse_float_list(rex_betas_raw)
            else:
                betas = make_beta_ladder(
                    beta_min=float(args.rex_beta_min),
                    beta_max=float(args.rex_beta_max),
                    n_replicas=int(args.rex_n_replicas),
                    spacing=str(args.rex_spacing),
                )
            if all(abs(float(b) - float(beta)) > 1e-12 for b in betas):
                betas = sorted(set(list(betas) + [float(beta)]))
            total_rounds = max(1, int(args.rex_rounds))
            n_chains = max(1, int(args.rex_chains))
            if n_chains > total_rounds:
                n_chains = total_rounds
            if n_chains > 1:
                base_rounds = total_rounds // n_chains
                extra = total_rounds % n_chains
                chain_rounds = [base_rounds + (1 if i < extra else 0) for i in range(n_chains)]
            else:
                chain_rounds = [total_rounds]

            chain_runs: list[dict[str, object] | None] = [None] * n_chains
            burnin_clipped = False
            if n_chains <= 1:
                burn_in = min(int(args.rex_burnin_rounds), max(0, total_rounds - 1))
                burnin_clipped = burn_in != int(args.rex_burnin_rounds)
                chain_runs[0] = replica_exchange_gibbs_potts(
                    model_lam,
                    betas=betas,
                    sweeps_per_round=int(args.rex_sweeps_per_round),
                    n_rounds=int(total_rounds),
                    burn_in_rounds=int(burn_in),
                    thinning_rounds=int(args.rex_thin_rounds),
                    seed=int(base_seed),
                    progress=bool(args.progress),
                    progress_mode="samples",
                    progress_desc=f"λ={float(lam):.3f} REX",
                )
            else:
                with ProcessPoolExecutor(max_workers=n_chains) as executor:
                    futures = {}
                    for cidx in range(n_chains):
                        rounds = int(chain_rounds[cidx])
                        burn_in = min(int(args.rex_burnin_rounds), max(0, rounds - 1))
                        futures[executor.submit(
                            _run_rex_chain_worker,
                            {
                                "model": model_lam,
                                "betas": betas,
                                "sweeps_per_round": int(args.rex_sweeps_per_round),
                                "n_rounds": rounds,
                                "burn_in_rounds": int(burn_in),
                                "thinning_rounds": int(args.rex_thin_rounds),
                                "seed": int(base_seed) + cidx,
                                "progress": False,  # progress bars across processes tend to garble output
                                "progress_mode": "samples",
                            },
                        )] = (cidx, burn_in != int(args.rex_burnin_rounds))
                    completed = 0
                    for fut in as_completed(futures):
                        cidx, clipped = futures[fut]
                        out = fut.result()
                        run = out.get("run")
                        chain_runs[cidx] = run if isinstance(run, dict) else None
                        burnin_clipped = burnin_clipped or bool(clipped)
                        completed += 1
                        if bool(args.progress):
                            print(f"[lambda_sweep]   λ={float(lam):.3f} REX chains {completed}/{n_chains}")

            if burnin_clipped and bool(args.progress):
                print(f"[lambda_sweep]   λ={float(lam):.3f} note: burn-in rounds truncated for short chains.")

            parts = []
            for run in chain_runs:
                if not isinstance(run, dict):
                    continue
                samples_by_beta = run.get("samples_by_beta")
                if not isinstance(samples_by_beta, dict):
                    continue
                arr = samples_by_beta.get(float(beta))
                if isinstance(arr, np.ndarray) and arr.size:
                    parts.append(arr)
            labels = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(model_lam.h)), dtype=int)

        summary_path = save_sample_npz(sample_dir / "sample.npz", labels=labels)
        try:
            rel_summary = str(summary_path.relative_to(system_dir))
        except Exception:
            rel_summary = str(summary_path)

        display_name = f"{series_label} \u03bb={float(lam):.3f}"
        sample_entry: dict = {
            "sample_id": sample_id,
            "name": display_name,
            "type": "potts_lambda_sweep",
            "method": "gibbs",
            "source": "lambda_sweep",
            "model_id": None,
            "model_ids": [str(args.model_b_id), str(args.model_a_id)],
            "model_names": [str(model_b_meta.get("name") or args.model_b_id), str(model_a_meta.get("name") or args.model_a_id)],
            "created_at": datetime.utcnow().isoformat(),
            "path": rel_summary,
            "paths": {"summary_npz": rel_summary},
            "params": _filter_sampling_params(sampling_params),
            "series_kind": "lambda_sweep",
            "series_id": series_id,
            "series_label": series_label,
            "lambda": float(lam),
            "lambda_index": int(idx),
            "lambda_count": int(args.lambda_count),
            "endpoint_model_a_id": str(args.model_a_id),
            "endpoint_model_b_id": str(args.model_b_id),
        }

        # Persist into system metadata immediately (so partial progress survives interruption)
        system_meta = store.get_system(project_id, system_id)
        clusters = system_meta.metastable_clusters or []
        cluster_entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
        if not isinstance(cluster_entry, dict):
            raise SystemExit(f"Cluster '{cluster_id}' not found while persisting samples.")
        samples_list = cluster_entry.get("samples")
        if not isinstance(samples_list, list):
            samples_list = []
        samples_list.append(sample_entry)
        cluster_entry["samples"] = samples_list
        system_meta.metastable_clusters = clusters
        store.save_system(system_meta)

        sample_ids.append(sample_id)
        sample_names.append(display_name)

    # Analysis artifact
    analysis_payload = compute_lambda_sweep_analysis(
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        model_a_ref=str(args.model_a_id),
        model_b_ref=str(args.model_b_id),
        lambda_sample_ids=sample_ids,
        lambdas=lambdas,
        ref_md_sample_ids=md_ids,
        md_label_mode=str(args.md_label_mode),
        drop_invalid=not bool(args.keep_invalid),
        alpha=float(args.alpha),
    )

    analyses_dir = cluster_dirs["cluster_dir"] / "analyses" / "lambda_sweep"
    analyses_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = str(uuid.uuid4())
    out_dir = analyses_dir / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "analysis.npz"
    meta_path = out_dir / "analysis_metadata.json"

    np.savez_compressed(
        npz_path,
        lambdas=np.asarray(analysis_payload["lambdas"], dtype=float),
        edges=np.asarray(analysis_payload["edges"], dtype=int),
        node_js_mean=np.asarray(analysis_payload["node_js_mean"], dtype=float),
        edge_js_mean=np.asarray(analysis_payload["edge_js_mean"], dtype=float),
        combined_distance=np.asarray(analysis_payload["combined_distance"], dtype=float),
        deltaE_mean=np.asarray(analysis_payload["deltaE_mean"], dtype=float),
        deltaE_q25=np.asarray(analysis_payload["deltaE_q25"], dtype=float),
        deltaE_q75=np.asarray(analysis_payload["deltaE_q75"], dtype=float),
        sample_ids=np.asarray(analysis_payload["sample_ids"], dtype=str),
        sample_names=np.asarray(analysis_payload["sample_names"], dtype=str),
        ref_md_sample_ids=np.asarray(analysis_payload["ref_md_sample_ids"], dtype=str),
        ref_md_sample_names=np.asarray(analysis_payload["ref_md_sample_names"], dtype=str),
        alpha=np.asarray([analysis_payload["alpha"]], dtype=float),
        match_ref_index=np.asarray([analysis_payload["match_ref_index"]], dtype=int),
        lambda_star_index=np.asarray([analysis_payload["lambda_star_index"]], dtype=int),
        lambda_star=np.asarray([analysis_payload["lambda_star"]], dtype=float),
        match_min=np.asarray([analysis_payload["match_min"]], dtype=float),
    )

    meta = {
        "analysis_id": analysis_id,
        "analysis_type": "lambda_sweep",
        "created_at": datetime.utcnow().isoformat(),
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "series_kind": "lambda_sweep",
        "series_id": series_id,
        "series_label": series_label,
        "model_a_id": str(args.model_a_id),
        "model_a_name": model_a_meta.get("name"),
        "model_b_id": str(args.model_b_id),
        "model_b_name": model_b_meta.get("name"),
        "md_sample_ids": md_ids,
        "md_sample_names": analysis_payload.get("ref_md_sample_names") or [],
        "md_label_mode": str(args.md_label_mode),
        "drop_invalid": bool(not args.keep_invalid),
        "alpha": float(args.alpha),
        "lambda_count": int(args.lambda_count),
        "paths": {
            "analysis_npz": str(npz_path.relative_to(system_dir)),
        },
        "summary": {
            "lambda_star": analysis_payload.get("lambda_star"),
            "match_min": analysis_payload.get("match_min"),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[lambda_sweep] series_id={series_id}")
    print(f"[lambda_sweep] wrote {len(sample_ids)} samples under clusters/{cluster_id}/samples/")
    print(f"[lambda_sweep] wrote analysis under clusters/{cluster_id}/analyses/lambda_sweep/{analysis_id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
