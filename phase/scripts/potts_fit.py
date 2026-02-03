from __future__ import annotations

import os
from pathlib import Path
import uuid

from phase.services.project_store import ProjectStore
from phase.potts import pipeline as sim_main
from phase.scripts.potts_utils import persist_model, sanitize_model_filename


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
        args.model_out = str(model_dir / f"{sanitize_model_filename(display_name)}.npz")
    try:
        results = sim_main.run_pipeline(args, parser=parser)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    if project_id and system_id and cluster_id:
        model_path = Path(results.get("model_path")) if results else None
        if model_path:
            params = vars(args) if hasattr(args, "__dict__") else {}
            params.setdefault("fit_mode", "standard")
            store = ProjectStore(base_dir=Path(os.getenv("PHASE_DATA_ROOT", "/app/data")) / "projects")
            resume_path = None
            if args.plm_resume_model:
                resume_path = Path(args.plm_resume_model)
                if not resume_path.is_absolute():
                    resume_path = store.resolve_path(project_id, system_id, str(args.plm_resume_model))
            if resume_path and resume_path.resolve() == model_path.resolve():
                try:
                    system_meta = store.get_system(project_id, system_id)
                except FileNotFoundError:
                    system_meta = None
                if system_meta:
                    entry = next(
                        (c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id),
                        None,
                    )
                    if isinstance(entry, dict):
                        models = entry.get("potts_models") or []
                        updated = False
                        for model in models:
                            rel_path = model.get("path")
                            if not rel_path:
                                continue
                            abs_path = store.resolve_path(project_id, system_id, rel_path)
                            if abs_path.resolve() == resume_path.resolve():
                                model["params"] = params
                                updated = True
                                break
                        if updated:
                            entry["potts_models"] = models
                            store.save_system(system_meta)
                            return 0
                return 0
            persist_model(
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
