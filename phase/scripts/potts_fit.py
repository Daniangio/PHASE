from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import uuid

from phase.services.project_store import ProjectStore
from phase.potts import pipeline as sim_main


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
    source: str = "offline",
) -> str | None:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    try:
        system_meta = store.get_system(project_id, system_id)
    except FileNotFoundError:
        print(f"[potts_fit] warning: system {project_id}/{system_id} not found; leaving model in place.")
        return None

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
        print("[potts_fit] warning: cluster entry not found; model copied without metadata update.")

    return rel_path


def main(argv: list[str] | None = None) -> int:
    parser = sim_main._build_arg_parser()
    parser.add_argument("--project-id", default="")
    parser.add_argument("--system-id", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--model-source", default="offline")
    args = parser.parse_args(argv)
    args.fit_only = True
    project_id = (args.project_id or "").strip()
    system_id = (args.system_id or "").strip()
    cluster_id = (args.cluster_id or "").strip()
    if project_id and system_id and cluster_id and not args.model_out:
        data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
        store = ProjectStore(base_dir=data_root / "projects")
        try:
            system_meta = store.get_system(project_id, system_id)
        except FileNotFoundError:
            system_meta = None
        entry = None
        if system_meta:
            entry = next(
                (c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id),
                None,
            )
        display_name = args.model_name or None
        if not display_name and isinstance(entry, dict):
            cluster_name = entry.get("name")
            if not display_name and isinstance(cluster_name, str) and cluster_name.strip():
                display_name = f"{cluster_name} Potts Model"
        if not display_name:
            display_name = f"{cluster_id} Potts Model"
        model_id = str(uuid.uuid4())
        model_dir = store.ensure_cluster_directories(project_id, system_id, cluster_id)["potts_models_dir"] / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        args.model_out = str(model_dir / f"{_sanitize_model_filename(display_name)}.npz")
    try:
        results = sim_main.run_pipeline(args, parser=parser)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    if project_id and system_id and cluster_id:
        model_path = Path(results.get("model_path")) if results else None
        if model_path:
            params = vars(args) if hasattr(args, "__dict__") else {}
            _persist_model(
                project_id=project_id,
                system_id=system_id,
                cluster_id=cluster_id,
                model_path=model_path,
                model_name=args.model_name or None,
                params=params,
                source=args.model_source or "offline",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
