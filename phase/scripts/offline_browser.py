from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from backend.services.project_store import ProjectStore, ProjectMetadata, SystemMetadata


def _project_meta_path(root: Path, project_id: str) -> Path:
    return root / project_id / "project.json"


def _system_meta_path(root: Path, project_id: str, system_id: str) -> Path:
    return root / project_id / "systems" / system_id / "system.json"


def _load_project_meta(root: Path, project_id: str) -> ProjectMetadata:
    path = _project_meta_path(root, project_id)
    data = json.loads(path.read_text())
    return ProjectMetadata(**data)


def _load_system_meta(root: Path, project_id: str, system_id: str) -> SystemMetadata:
    path = _system_meta_path(root, project_id, system_id)
    data = json.loads(path.read_text())
    return SystemMetadata(**data)


def _iter_project_ids(root: Path) -> Iterable[str]:
    if not root.exists():
        return []
    return [p.name for p in root.iterdir() if p.is_dir() and (p / "project.json").exists()]


def _print_rows(rows: List[List[str]]) -> None:
    for row in rows:
        print("|".join(row))


def list_projects(root: Path) -> None:
    rows = []
    for project_id in _iter_project_ids(root):
        meta = _load_project_meta(root, project_id)
        rows.append([project_id, meta.name or project_id])
    _print_rows(rows)


def list_systems(root: Path, project_id: str) -> None:
    meta = _load_project_meta(root, project_id)
    rows = []
    for system_id in meta.systems or []:
        sys_meta = _load_system_meta(root, project_id, system_id)
        rows.append([system_id, sys_meta.name or system_id])
    _print_rows(rows)


def list_states(root: Path, project_id: str, system_id: str) -> None:
    sys_meta = _load_system_meta(root, project_id, system_id)
    rows = []
    for state_id, state in (sys_meta.states or {}).items():
        if isinstance(state, dict):
            name = state.get("name") or state_id
            pdb_file = state.get("pdb_file") or ""
            traj_file = state.get("trajectory_file") or ""
            desc_file = state.get("descriptor_file") or ""
        else:
            name = getattr(state, "name", None) or state_id
            pdb_file = getattr(state, "pdb_file", "") or ""
            traj_file = getattr(state, "trajectory_file", "") or ""
            desc_file = getattr(state, "descriptor_file", "") or ""
        rows.append([state_id, name, pdb_file, traj_file, desc_file])
    _print_rows(rows)


def list_analysis_states(root: Path, project_id: str, system_id: str) -> None:
    sys_meta = _load_system_meta(root, project_id, system_id)
    rows = []
    analysis_states = getattr(sys_meta, "analysis_states", None) or sys_meta.__dict__.get("analysis_states") or []
    if not analysis_states:
        analysis_states = []
        for state_id, state in (sys_meta.states or {}).items():
            if isinstance(state, dict):
                analysis_states.append({**state, "state_id": state_id, "kind": "macro"})
            else:
                analysis_states.append(
                    {
                        "state_id": state_id,
                        "name": getattr(state, "name", None) or state_id,
                        "kind": "macro",
                        "descriptor_file": getattr(state, "descriptor_file", None),
                    }
                )
        for meta in sys_meta.metastable_states or []:
            meta_id = meta.get("metastable_id") or meta.get("id")
            if not meta_id:
                continue
            analysis_states.append(
                {
                    "state_id": str(meta_id),
                    "name": meta.get("name") or meta.get("default_name") or str(meta_id),
                    "kind": "metastable",
                    "descriptor_file": None,
                }
            )
    for item in analysis_states:
        if not isinstance(item, dict):
            continue
        state_id = item.get("state_id") or item.get("metastable_id") or item.get("id")
        if not state_id:
            continue
        label = item.get("name") or str(state_id)
        kind = item.get("kind") or ("metastable" if item.get("metastable_id") else "macro")
        desc = item.get("descriptor_file") or ""
        rows.append([str(state_id), f"[{kind}] {label}", desc])
    _print_rows(rows)


def list_clusters(root: Path, project_id: str, system_id: str) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    rows = []
    for entry in sys_meta.metastable_clusters or []:
        cluster_id = entry.get("cluster_id") or ""
        name = entry.get("name") or cluster_id
        rel_path = entry.get("path") or ""
        abs_path = store.resolve_path(project_id, system_id, rel_path) if rel_path else Path("")
        rows.append([cluster_id, name, str(abs_path) if rel_path else ""])
    _print_rows(rows)


def list_models(root: Path, project_id: str, system_id: str) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    rows = []
    for entry in sys_meta.metastable_clusters or []:
        cluster_id = entry.get("cluster_id") or ""
        model_rel = entry.get("potts_model_path")
        if not model_rel:
            continue
        name = entry.get("potts_model_name") or Path(model_rel).stem
        abs_path = store.resolve_path(project_id, system_id, model_rel)
        if not abs_path.exists():
            continue
        rows.append([cluster_id, name, str(abs_path)])
    _print_rows(rows)


def prune_models(root: Path, project_id: str, system_id: str, *, dry_run: bool) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    clusters = sys_meta.metastable_clusters or []
    removed = []
    for entry in clusters:
        if not isinstance(entry, dict):
            continue
        model_rel = entry.get("potts_model_path")
        if not model_rel:
            continue
        abs_path = store.resolve_path(project_id, system_id, model_rel)
        if abs_path.exists():
            continue
        removed.append([
            entry.get("cluster_id") or "",
            entry.get("potts_model_name") or Path(model_rel).stem,
            str(abs_path),
        ])
        if not dry_run:
            entry.pop("potts_model_path", None)
            entry.pop("potts_model_name", None)
            entry.pop("potts_model_id", None)
            entry.pop("potts_model_source", None)
            entry.pop("potts_model_params", None)
            entry.pop("potts_model_updated_at", None)
    if not dry_run:
        sys_meta.metastable_clusters = clusters
        store.save_system(sys_meta)
    _print_rows(removed)


def list_descriptors(root: Path, project_id: str, system_id: str) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    rows = []
    for state_id, state in (sys_meta.states or {}).items():
        if not state.descriptor_file:
            continue
        abs_path = store.resolve_path(project_id, system_id, state.descriptor_file)
        rows.append([state_id, state.name or state_id, str(abs_path)])
    _print_rows(rows)


def list_pdbs(root: Path, project_id: str, system_id: str) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    rows = []
    for state_id, state in (sys_meta.states or {}).items():
        if not state.pdb_file:
            continue
        abs_path = store.resolve_path(project_id, system_id, state.pdb_file)
        rows.append([state_id, state.name or state_id, str(abs_path)])
    _print_rows(rows)


def list_trajectories(root: Path, project_id: str, system_id: str) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    rows = []
    for state_id, state in (sys_meta.states or {}).items():
        if not state.trajectory_file:
            continue
        abs_path = store.resolve_path(project_id, system_id, state.trajectory_file)
        rows.append([state_id, state.name or state_id, str(abs_path)])
    _print_rows(rows)


def list_sampling(root: Path, project_id: str, system_id: str) -> None:
    system_dir = root / project_id / "systems" / system_id
    results_dir = root.parent / "results"
    rows = []
    if not results_dir.exists():
        _print_rows(rows)
        return
    for meta_path in results_dir.rglob("run_metadata.json"):
        try:
            payload = json.loads(meta_path.read_text())
        except Exception:
            continue
        data_npz = payload.get("data_npz") or payload.get("args", {}).get("npz")
        if not data_npz:
            continue
        npz_path = Path(data_npz)
        if not npz_path.is_absolute():
            npz_path = (meta_path.parent / npz_path).resolve()
        try:
            matches_system = system_dir in npz_path.parents
        except Exception:
            matches_system = False
        if not matches_system:
            continue
        run_dir = meta_path.parent
        summary_path = run_dir / payload.get("summary_file", "run_summary.npz")
        rows.append([run_dir.name, run_dir.name, str(summary_path), str(npz_path)])
    _print_rows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Offline data root (contains projects/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-projects")

    sp = sub.add_parser("list-systems")
    sp.add_argument("--project-id", required=True)

    sp = sub.add_parser("list-states")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    sp = sub.add_parser("list-analysis-states")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    sp = sub.add_parser("list-clusters")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    sp = sub.add_parser("list-models")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    sp = sub.add_parser("prune-models")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("list-descriptors")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    sp = sub.add_parser("list-pdbs")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    sp = sub.add_parser("list-trajectories")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    sp = sub.add_parser("list-sampling")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)

    args = ap.parse_args(argv)
    root = Path(args.root).expanduser().resolve() / "projects"

    if args.cmd == "list-projects":
        list_projects(root)
    elif args.cmd == "list-systems":
        list_systems(root, args.project_id)
    elif args.cmd == "list-states":
        list_states(root, args.project_id, args.system_id)
    elif args.cmd == "list-analysis-states":
        list_analysis_states(root, args.project_id, args.system_id)
    elif args.cmd == "list-clusters":
        list_clusters(root, args.project_id, args.system_id)
    elif args.cmd == "list-models":
        list_models(root, args.project_id, args.system_id)
    elif args.cmd == "prune-models":
        prune_models(root, args.project_id, args.system_id, dry_run=args.dry_run)
    elif args.cmd == "list-descriptors":
        list_descriptors(root, args.project_id, args.system_id)
    elif args.cmd == "list-pdbs":
        list_pdbs(root, args.project_id, args.system_id)
    elif args.cmd == "list-trajectories":
        list_trajectories(root, args.project_id, args.system_id)
    elif args.cmd == "list-sampling":
        list_sampling(root, args.project_id, args.system_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
