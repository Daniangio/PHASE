from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from phase.services.project_store import ProjectStore


def sanitize_model_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    safe = safe.strip("._-")
    return safe or "potts_model"


def persist_model(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_path: Path,
    model_name: str | None,
    params: dict,
    source: str = "offline",
    model_id: str | None = None,
) -> tuple[str | None, str | None]:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    try:
        system_meta = store.get_system(project_id, system_id)
    except FileNotFoundError:
        print(f"[potts] warning: system {project_id}/{system_id} not found; leaving model in place.")
        return None, None

    entry = next(
        (c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id),
        None,
    )
    dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = dirs["system_dir"]
    model_dir = dirs["potts_models_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    display_name = model_name
    if not display_name:
        if isinstance(entry, dict):
            display_name = entry.get("potts_model_name")
        if not display_name and isinstance(entry, dict):
            cluster_name = entry.get("name")
            if isinstance(cluster_name, str) and cluster_name.strip():
                display_name = f"{cluster_name} Potts Model"
        if not display_name:
            display_name = f"{cluster_id} Potts Model"

    model_id = model_id or str(uuid.uuid4())
    base_name = sanitize_model_filename(display_name)
    filename = f"{base_name}.npz"
    model_bucket = model_dir / model_id
    model_bucket.mkdir(parents=True, exist_ok=True)
    dest_path = model_bucket / filename
    if dest_path.exists():
        suffix = cluster_id[:8]
        dest_path = model_bucket / f"{base_name}-{suffix}.npz"
        counter = 2
        while dest_path.exists():
            dest_path = model_bucket / f"{base_name}-{suffix}-{counter}.npz"
            counter += 1

    if model_path.resolve() != dest_path.resolve():
        shutil.copy2(model_path, dest_path)
    try:
        rel_path = str(dest_path.relative_to(system_dir))
    except Exception:
        rel_path = str(dest_path)

    if isinstance(entry, dict):
        models = entry.get("potts_models")
        if not isinstance(models, list):
            models = []
        models.append(
            {
                "model_id": model_id,
                "name": display_name,
                "path": rel_path,
                "created_at": datetime.utcnow().isoformat(),
                "source": source,
                "params": params,
            }
        )
        entry["potts_models"] = models
        store.save_system(system_meta)
    else:
        print("[potts] warning: cluster entry not found; model copied without metadata update.")

    return rel_path, model_id


def persist_sample(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    summary_path: Path,
    metadata_path: Path | None,
    sample_name: str | None,
    sample_type: str,
    method: str | None,
    params: dict,
    model_paths: list[Path] | None = None,
    sample_id: str | None = None,
) -> str | None:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    try:
        system_meta = store.get_system(project_id, system_id)
    except FileNotFoundError:
        print(f"[potts] warning: system {project_id}/{system_id} not found; leaving sample in place.")
        return None

    entry = next(
        (c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id),
        None,
    )
    if not isinstance(entry, dict):
        print("[potts] warning: cluster entry not found; sample metadata not updated.")
        return None

    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    samples_dir = cluster_dirs["samples_dir"]
    system_dir = cluster_dirs["system_dir"]

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    resolved_sample_id = sample_id
    if not resolved_sample_id:
        if summary_path.parent.parent.resolve() == samples_dir.resolve():
            resolved_sample_id = summary_path.parent.name
        else:
            resolved_sample_id = str(uuid.uuid4())

    sample_dir = samples_dir / resolved_sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    def _copy_if_needed(path: Path | None) -> Path | None:
        if path is None:
            return None
        if not path.exists():
            return None
        dest = path
        if dest.parent.resolve() != sample_dir.resolve():
            dest = sample_dir / path.name
            if dest.resolve() != path.resolve():
                shutil.copy2(path, dest)
        return dest

    summary_dest = _copy_if_needed(summary_path)
    meta_dest = _copy_if_needed(metadata_path) if metadata_path else None

    paths: dict[str, str] = {}
    if summary_dest:
        try:
            paths["summary_npz"] = str(summary_dest.relative_to(system_dir))
        except Exception:
            paths["summary_npz"] = str(summary_dest)
    if meta_dest:
        try:
            paths["metadata_json"] = str(meta_dest.relative_to(system_dir))
        except Exception:
            paths["metadata_json"] = str(meta_dest)

    primary_path = paths.get("summary_npz") or paths.get("metadata_json")

    model_ids: list[str] = []
    model_names: list[str] = []
    model_id = None
    if model_paths:
        for path in model_paths:
            abs_path = path
            if not abs_path.is_absolute():
                abs_path = store.resolve_path(project_id, system_id, str(path))
            for model in entry.get("potts_models") or []:
                rel = model.get("path")
                if not rel:
                    continue
                stored = store.resolve_path(project_id, system_id, rel)
                if stored.resolve() == abs_path.resolve():
                    mid = model.get("model_id")
                    if mid:
                        model_ids.append(str(mid))
                    name = model.get("name")
                    if name:
                        model_names.append(str(name))
    if model_ids:
        model_id = model_ids[0]
    if not model_names and model_ids:
        model_names = model_ids[:]

    display_name = sample_name
    if isinstance(display_name, str):
        display_name = display_name.strip() or None
    if not display_name:
        display_name = f"Sampling {datetime.utcnow().strftime('%Y%m%d %H:%M')}"

    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []

    existing = next((s for s in samples if s.get("sample_id") == resolved_sample_id), None)
    summary = {}
    if metadata_path and Path(metadata_path).exists():
        try:
            meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        args = meta.get("args") if isinstance(meta, dict) else None
        if isinstance(args, dict):
            summary = {
                "sampling_method": args.get("sampling_method"),
                "gibbs_method": args.get("gibbs_method"),
                "beta": args.get("beta"),
                "gibbs_samples": args.get("gibbs_samples"),
                "gibbs_burnin": args.get("gibbs_burnin"),
                "gibbs_thin": args.get("gibbs_thin"),
                "gibbs_chains": args.get("gibbs_chains"),
                "rex_rounds": args.get("rex_rounds"),
                "rex_burnin_rounds": args.get("rex_burnin_rounds"),
                "rex_thin_rounds": args.get("rex_thin_rounds"),
                "rex_n_replicas": args.get("rex_n_replicas"),
                "rex_beta_min": args.get("rex_beta_min"),
                "rex_beta_max": args.get("rex_beta_max"),
                "rex_spacing": args.get("rex_spacing"),
                "rex_betas": args.get("rex_betas"),
                "sa_reads": args.get("sa_reads"),
                "sa_sweeps": args.get("sa_sweeps"),
                "sa_beta_hot": args.get("sa_beta_hot"),
                "sa_beta_cold": args.get("sa_beta_cold"),
                "sa_beta_schedule": args.get("sa_beta_schedule"),
                "beta_eff": meta.get("beta_eff"),
                "beta_eff_by_schedule": meta.get("beta_eff_by_schedule"),
            }
        if "beta_eff" in meta and meta.get("beta_eff") is not None:
            summary["beta_eff"] = meta.get("beta_eff")

    entry_payload = {
        "sample_id": resolved_sample_id,
        "name": display_name,
        "type": sample_type,
        "method": method,
        "model_id": model_id,
        "model_ids": model_ids or None,
        "model_names": model_names or None,
        "created_at": datetime.utcnow().isoformat(),
        "path": primary_path,
        "paths": paths,
        "params": params,
        "summary": summary,
    }
    if existing is not None:
        existing.update(entry_payload)
    else:
        samples.append(entry_payload)
    entry["samples"] = samples
    store.save_system(system_meta)
    return resolved_sample_id
