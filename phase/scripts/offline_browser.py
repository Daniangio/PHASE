from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import List

from phase.services.project_store import ProjectStore


def _print_rows(rows: List[List[str]]) -> None:
    for row in rows:
        print("|".join(row))


def list_projects(root: Path) -> None:
    store = ProjectStore(base_dir=root)
    rows = []
    for meta in store.list_projects():
        rows.append([meta.project_id, meta.name or meta.project_id])
    _print_rows(rows)


def list_systems(root: Path, project_id: str) -> None:
    store = ProjectStore(base_dir=root)
    rows = []
    for sys_meta in store.list_systems(project_id):
        rows.append([sys_meta.system_id, sys_meta.name or sys_meta.system_id])
    _print_rows(rows)


def list_states(root: Path, project_id: str, system_id: str) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
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
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
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
        cluster_name = entry.get("name") or cluster_id
        for model in entry.get("potts_models") or []:
            model_rel = model.get("path")
            if not model_rel:
                continue
            abs_path = store.resolve_path(project_id, system_id, model_rel)
            if not abs_path.exists():
                continue
            model_name = model.get("name") or Path(model_rel).stem
            params = model.get("params") if isinstance(model.get("params"), dict) else {}
            delta_kind = str(params.get("delta_kind") or "")
            fit_mode = str(params.get("fit_mode") or "")
            rows.append([
                model.get("model_id") or "",
                f"{cluster_name} :: {model_name}",
                str(abs_path),
                cluster_id,
                delta_kind,
                fit_mode,
            ])
    _print_rows(rows)


def prune_models(root: Path, project_id: str, system_id: str, *, dry_run: bool) -> None:
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    clusters = sys_meta.metastable_clusters or []
    removed = []
    for entry in clusters:
        if not isinstance(entry, dict):
            continue
        models = entry.get("potts_models") or []
        keep = []
        for model in models:
            model_rel = model.get("path")
            if not model_rel:
                continue
            abs_path = store.resolve_path(project_id, system_id, model_rel)
            if abs_path.exists():
                keep.append(model)
                continue
            removed.append([
                entry.get("cluster_id") or "",
                model.get("model_id") or "",
                model.get("name") or Path(model_rel).stem,
                str(abs_path),
            ])
        if not dry_run:
            entry["potts_models"] = keep
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
    store = ProjectStore(base_dir=root)
    sys_meta = store.get_system(project_id, system_id)
    rows = []
    for entry in sys_meta.metastable_clusters or []:
        cluster_id = entry.get("cluster_id") or ""
        cluster_name = entry.get("name") or cluster_id
        for sample in entry.get("samples") or []:
            sample_id = sample.get("sample_id") or ""
            name = sample.get("name") or sample_id
            sample_type = sample.get("type") or ""
            paths = sample.get("paths") or {}
            sample_path = sample.get("path")
            if isinstance(paths, dict) and paths:
                candidate = (
                    paths.get("summary_npz")
                    or paths.get("metadata_json")
                    or paths.get("sampling_report")
                    or paths.get("cross_likelihood_report")
                    or paths.get("marginals_plot")
                )
                if candidate:
                    sample_path = candidate
            abs_path = ""
            if sample_path:
                abs_path = str(store.resolve_path(project_id, system_id, sample_path))
            rows.append([cluster_id, f"{cluster_name} :: {name}", sample_type, abs_path])
    _print_rows(rows)


def list_cluster_samples(root: Path, project_id: str, system_id: str, cluster_id: str) -> None:
    store = ProjectStore(base_dir=root)
    rows = []
    for sample in store.list_samples(project_id, system_id, cluster_id):
        sample_id = str(sample.get("sample_id") or "")
        name = str(sample.get("name") or sample_id)
        typ = str(sample.get("type") or "")
        path = sample.get("path") or ""
        rows.append([sample_id, name, typ, str(path)])
    _print_rows(rows)


def list_analyses(root: Path, project_id: str, system_id: str, cluster_id: str, analysis_type: str) -> None:
    store = ProjectStore(base_dir=root)
    dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    kind_root = dirs["cluster_dir"] / "analyses" / analysis_type
    rows = []
    if not kind_root.exists():
        _print_rows(rows)
        return
    for analysis_dir in sorted((p for p in kind_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        meta_path = analysis_dir / "analysis_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mode = str(meta.get("mode") or analysis_type)
        if mode == "single":
            label = str(meta.get("state_name") or meta.get("state_id") or analysis_dir.name)
        elif mode == "pair":
            label = f"{meta.get('state_a_name') or meta.get('state_a_id')} -> {meta.get('state_b_name') or meta.get('state_b_id')}"
        elif mode == "intersection":
            label = (
                f"{meta.get('single_state_name') or meta.get('single_analysis_id')} x "
                f"{meta.get('pair_state_a_name') or meta.get('pair_state_a_id')} -> {meta.get('pair_state_b_name') or meta.get('pair_state_b_id')}"
            )
        else:
            label = str(meta.get("name") or analysis_dir.name)
        updated = str(meta.get("updated_at") or meta.get("created_at") or "")
        rows.append([analysis_dir.name, label, analysis_type, updated])
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

    sp = sub.add_parser("list-cluster-samples")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)
    sp.add_argument("--cluster-id", required=True)

    sp = sub.add_parser("list-analyses")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)
    sp.add_argument("--cluster-id", required=True)
    sp.add_argument("--analysis-type", required=True)

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
    elif args.cmd == "list-cluster-samples":
        list_cluster_samples(root, args.project_id, args.system_id, args.cluster_id)
    elif args.cmd == "list-analyses":
        list_analyses(root, args.project_id, args.system_id, args.cluster_id, args.analysis_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
