from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

from phase.services.project_store import ProjectStore, SystemMetadata
from phase.workflows.macro_states import add_state as add_macro_state, refresh_system_metadata


def _root_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned


def _unique_id(base: str, existing: set[str]) -> str:
    candidate = base or str(uuid.uuid4())
    if candidate not in existing:
        return candidate
    idx = 2
    while f"{candidate}-{idx}" in existing:
        idx += 1
    return f"{candidate}-{idx}"


def init_project(
    store: ProjectStore,
    name: str,
    description: str | None,
    *,
    use_slug_ids: bool,
) -> str:
    project_id = None
    if use_slug_ids:
        slug = _slugify(name)
        existing = {p.project_id for p in store.list_projects()}
        project_id = _unique_id(slug, existing)
    project = store.create_project(name=name, description=description, project_id=project_id)
    print(project.project_id)
    return project.project_id


def create_system(
    store: ProjectStore,
    project_id: str,
    name: str | None,
    description: str | None,
    *,
    use_slug_ids: bool,
) -> str:
    system_id = None
    if use_slug_ids:
        slug = _slugify(name or "")
        project_meta = store.get_project(project_id)
        existing = set(project_meta.systems or [])
        system_id = _unique_id(slug, existing)
    system = store.create_system(project_id=project_id, name=name, description=description, system_id=system_id)
    print(system.system_id)
    return system.system_id


def add_state(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    state_id: str,
    name: str | None,
    pdb_path: Path,
    traj_path: Path,
    residue_selection: str | None,
    copy_traj: bool,
    build_descriptors: bool,
    slice_spec: str | None,
    resid_shift: int,
) -> None:
    try:
        add_macro_state(
            store,
            project_id,
            system_id,
            state_id,
            name,
            pdb_path,
            traj_path,
            residue_selection,
            copy_traj,
            build_descriptors,
            slice_spec,
            resid_shift,
        )
    except Exception as exc:
        print(f"Failed to add state '{state_id}': {exc}")
        raise

    print(state_id)


def delete_state(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    state_id: str,
) -> None:
    system = store.get_system(project_id, system_id)
    state = (system.states or {}).get(state_id)
    if state is None:
        raise FileNotFoundError(f"State '{state_id}' not found in system '{system_id}'.")

    system_dir = store.ensure_directories(project_id, system_id)["system_dir"].resolve()
    for field in ("descriptor_file", "descriptor_metadata_file", "trajectory_file", "pdb_file"):
        rel_path = getattr(state, field, None)
        if not rel_path:
            continue
        try:
            abs_path = store.resolve_path(project_id, system_id, rel_path).resolve()
            abs_path.relative_to(system_dir)
        except Exception:
            # Never delete external/original source files.
            continue
        try:
            abs_path.unlink(missing_ok=True)
        except Exception:
            pass

    system.states.pop(state_id, None)
    refresh_system_metadata(system)
    store.save_system(system)
    print(state_id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline system/project setup for PHASE.")
    ap.add_argument("--root", required=True, help="Offline data root (will create projects/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init-project")
    sp.add_argument("--name", required=True)
    sp.add_argument("--description")
    sp.add_argument("--use-slug-ids", action="store_true")

    sp = sub.add_parser("create-system")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--name")
    sp.add_argument("--description")
    sp.add_argument("--use-slug-ids", action="store_true")

    sp = sub.add_parser("add-state")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)
    sp.add_argument("--state-id", required=True)
    sp.add_argument("--name")
    sp.add_argument("--pdb", required=True)
    sp.add_argument("--traj", required=True)
    sp.add_argument("--residue-selection")
    sp.add_argument("--copy-traj", action="store_true")
    sp.add_argument("--slice-spec")
    sp.add_argument("--resid-shift", type=int, default=0)

    sp = sub.add_parser("delete-state")
    sp.add_argument("--project-id", required=True)
    sp.add_argument("--system-id", required=True)
    sp.add_argument("--state-id", required=True)

    args = ap.parse_args(argv)
    root = _root_path(args.root) / "projects"
    store = ProjectStore(base_dir=root)

    if args.cmd == "init-project":
        init_project(store, args.name, args.description, use_slug_ids=args.use_slug_ids)
    elif args.cmd == "create-system":
        create_system(store, args.project_id, args.name, args.description, use_slug_ids=args.use_slug_ids)
    elif args.cmd == "add-state":
        add_state(
            store,
            args.project_id,
            args.system_id,
            args.state_id,
            args.name,
            Path(args.pdb).expanduser().resolve(),
            Path(args.traj).expanduser().resolve(),
            args.residue_selection,
            args.copy_traj,
            True,
            args.slice_spec,
            args.resid_shift,
        )
    elif args.cmd == "delete-state":
        delete_state(
            store,
            args.project_id,
            args.system_id,
            args.state_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
