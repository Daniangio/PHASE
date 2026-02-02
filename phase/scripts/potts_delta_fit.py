from __future__ import annotations

from datetime import datetime
import argparse
import os
from pathlib import Path
import re
import shutil
import uuid

from phase.services.project_store import ProjectStore
from phase.potts import delta_fit


def _sanitize_model_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    safe = safe.strip("._-")
    return safe or "potts_model"


def _persist_model(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_path: Path,
    model_name: str | None,
    params: dict,
    source: str = "offline_delta_fit",
) -> tuple[str | None, str | None]:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    try:
        system_meta = store.get_system(project_id, system_id)
    except FileNotFoundError:
        print(f"[potts_delta_fit] warning: system {project_id}/{system_id} not found; leaving model in place.")
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
            cluster_name = entry.get("name")
            if isinstance(cluster_name, str) and cluster_name.strip():
                display_name = f"{cluster_name} Potts Model"
    if not display_name:
        display_name = f"{cluster_id} Potts Model"

    model_id = str(uuid.uuid4())
    base_name = _sanitize_model_filename(display_name)
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
        print("[potts_delta_fit] warning: cluster entry not found; model copied without metadata update.")

    return rel_path, model_id


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="")
    parser.add_argument("--system-id", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--model-source", default="offline_delta_fit")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def main(argv: list[str] | None = None) -> int:
    args, remaining = _parse_args(argv)
    delta_args = delta_fit._parse_args(remaining)

    exit_code = delta_fit.main(remaining)
    if exit_code:
        return int(exit_code)

    project_id = (args.project_id or "").strip()
    system_id = (args.system_id or "").strip()
    cluster_id = (args.cluster_id or "").strip()
    if not project_id or not system_id or not cluster_id:
        print("[potts_delta_fit] skipping metadata update (missing project/system/cluster ids).")
        return 0

    results_dir = Path(delta_args.results_dir)
    if not results_dir.exists():
        print(f"[potts_delta_fit] results dir not found: {results_dir}")
        return 1

    base_label = (args.model_name or "").strip()
    if not base_label:
        base_label = f"{Path(delta_args.base_model).stem} Delta"
    state_ids = []
    if getattr(delta_args, "state_ids", None):
        state_ids = [s.strip() for s in str(delta_args.state_ids).split(",") if s.strip()]
    if state_ids:
        base_label = f"{base_label} ({','.join(state_ids)})"

    params = vars(delta_args)
    params["fit_mode"] = "delta"

    outputs = [
        ("delta_patch", results_dir / "delta_model.npz", f"{base_label} (delta)"),
        ("model_patch", results_dir / "model_combined.npz", f"{base_label} (combined)"),
        ("delta_active", results_dir / "delta_active.npz", f"{base_label} (delta active)"),
        ("delta_inactive", results_dir / "delta_inactive.npz", f"{base_label} (delta inactive)"),
        ("model_active", results_dir / "model_active.npz", f"{base_label} (combined active)"),
        ("model_inactive", results_dir / "model_inactive.npz", f"{base_label} (combined inactive)"),
    ]

    for kind, path, name in outputs:
        if not path.exists():
            continue
        model_params = dict(params)
        model_params["delta_kind"] = kind
        rel_path, model_id = _persist_model(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_path=path,
            model_name=name,
            params=model_params,
            source=args.model_source or "offline_delta_fit",
        )
        if rel_path and model_id:
            print(f"[potts_delta_fit] saved {kind} model {model_id}: {rel_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
