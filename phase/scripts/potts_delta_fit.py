from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.services.project_store import ProjectStore
from phase.potts import delta_fit
from phase.scripts.potts_utils import persist_model


def _update_existing_model(
    store: ProjectStore,
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_path: Path,
    params: dict,
    source: str,
) -> bool:
    try:
        system_meta = store.get_system(project_id, system_id)
    except FileNotFoundError:
        return False

    entry = next(
        (c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id),
        None,
    )
    if not isinstance(entry, dict):
        return False
    models = entry.get("potts_models") or []
    updated = False
    for model in models:
        rel_path = model.get("path")
        if not rel_path:
            continue
        abs_path = store.resolve_path(project_id, system_id, rel_path)
        if abs_path.resolve() == model_path.resolve():
            model["params"] = params
            if source:
                model["source"] = source
            updated = True
            break
    if updated:
        entry["potts_models"] = models
        store.save_system(system_meta)
    return updated


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
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")

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

    resume_path = None
    resume_raw = getattr(delta_args, "resume_model", None)
    if resume_raw:
        resume_path = Path(str(resume_raw))
        if not resume_path.is_absolute():
            resume_path = store.resolve_path(project_id, system_id, str(resume_raw))

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
        if resume_path and resume_path.resolve() == path.resolve():
            updated = _update_existing_model(
                store,
                project_id=project_id,
                system_id=system_id,
                cluster_id=cluster_id,
                model_path=path,
                params=model_params,
                source=args.model_source or "offline_delta_fit",
            )
            if updated:
                print(f"[potts_delta_fit] updated {kind} model metadata: {path}")
            else:
                print("[potts_delta_fit] warning: resume model not found in metadata; skipping new entry.")
            continue
        rel_path, model_id = persist_model(
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
