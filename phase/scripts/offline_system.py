from __future__ import annotations

import argparse
import json
import shutil
import re
import uuid
from pathlib import Path

from backend.services.selection_utils import build_residue_selection_config
from backend.services.descriptors import save_descriptor_npz
from backend.services.preprocessing import DescriptorPreprocessor
from backend.services.project_store import DescriptorState, ProjectStore, SystemMetadata
from backend.services.state_utils import build_analysis_states


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
) -> None:
    system = store.get_system(project_id, system_id)
    dirs = store.ensure_directories(project_id, system_id)

    if state_id in system.states:
        raise ValueError(f"State '{state_id}' already exists.")

    pdb_ext = pdb_path.suffix or ".pdb"
    traj_ext = traj_path.suffix or ".xtc"

    pdb_dest = dirs["structures_dir"] / f"{state_id}{pdb_ext}"
    traj_dest = dirs["trajectories_dir"] / f"{state_id}{traj_ext}"

    pdb_dest.parent.mkdir(parents=True, exist_ok=True)
    traj_dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(pdb_path, pdb_dest)
        traj_value = str(traj_path)
        if copy_traj:
            shutil.copy2(traj_path, traj_dest)
            traj_value = str(traj_dest.relative_to(dirs["system_dir"]))

        slice_value = slice_spec.strip() if slice_spec else None
        stride_val = 1
        if slice_value:
            from backend.services.slice_utils import parse_slice_spec

            slice_value, stride_val = parse_slice_spec(slice_value)
        state = DescriptorState(
            state_id=state_id,
            name=name or state_id,
            pdb_file=str(pdb_dest.relative_to(dirs["system_dir"])),
            trajectory_file=traj_value,
            residue_selection=residue_selection,
            slice_spec=slice_value,
            stride=stride_val,
        )
        system.states[state_id] = state
        _refresh_system_metadata(system)
        store.save_system(system)
        if build_descriptors:
            _build_state_descriptors(store, project_id, system, state, residue_selection)
    except Exception as exc:
        system.states.pop(state_id, None)
        try:
            if pdb_dest.exists():
                pdb_dest.unlink()
        except Exception:
            pass
        if copy_traj:
            try:
                if traj_dest.exists():
                    traj_dest.unlink()
            except Exception:
                pass
        store.save_system(system)
        print(f"Failed to add state '{state_id}': {exc}")
        raise

    print(state_id)


def _refresh_system_metadata(system_meta: SystemMetadata) -> None:
    all_keys = set()
    for state in system_meta.states.values():
        all_keys.update(state.residue_keys or [])
    system_meta.descriptor_keys = sorted(all_keys)
    system_meta.analysis_states = build_analysis_states(system_meta)
    _update_system_status(system_meta)


def _update_system_status(system_meta: SystemMetadata) -> None:
    descriptors_ready = [s for s in system_meta.states.values() if s.descriptor_file]
    trajectories_uploaded = [s for s in system_meta.states.values() if s.trajectory_file]
    if len(descriptors_ready) >= 2:
        system_meta.status = "ready"
    elif descriptors_ready:
        system_meta.status = "single-ready"
    elif trajectories_uploaded:
        system_meta.status = "awaiting-descriptor"
    elif system_meta.states:
        system_meta.status = "pdb-only"
    else:
        system_meta.status = "empty"

def _build_state_descriptors(
    store: ProjectStore,
    project_id: str,
    system_meta: SystemMetadata,
    state_meta: DescriptorState,
    residue_filter: str | None,
) -> None:
    if not state_meta.trajectory_file or not state_meta.pdb_file:
        raise ValueError("State must include trajectory and PDB to build descriptors.")

    dirs = store.ensure_directories(project_id, system_meta.system_id)
    system_dir = dirs["system_dir"]
    descriptors_dir = dirs["descriptors_dir"]

    traj_path = Path(state_meta.trajectory_file)
    if not traj_path.is_absolute():
        traj_path = store.resolve_path(project_id, system_meta.system_id, state_meta.trajectory_file)
    pdb_path = store.resolve_path(project_id, system_meta.system_id, state_meta.pdb_file)

    if not traj_path.exists():
        raise ValueError(f"Trajectory file not found: {traj_path}")
    if not pdb_path.exists():
        raise ValueError(f"PDB file not found: {pdb_path}")

    selection_used = "protein"
    if residue_filter is not None and residue_filter.strip():
        selection_used = f"protein and ({residue_filter.strip()})"
    elif system_meta.residue_selections:
        selection_used = "system_selections"
    selections_config = build_residue_selection_config(
        base_selections=system_meta.residue_selections,
        residue_filter=residue_filter,
    )

    preprocessor = DescriptorPreprocessor(residue_selections=selections_config)
    build_result = preprocessor.build_single(str(traj_path), str(pdb_path), state_meta.slice_spec)

    npz_path = descriptors_dir / f"{state_meta.state_id}_descriptors.npz"
    meta_path = descriptors_dir / f"{state_meta.state_id}_descriptor_metadata.json"
    save_descriptor_npz(npz_path, build_result.features)
    meta_path.write_text(
        json.dumps(
            {
                "descriptor_keys": build_result.residue_keys,
                "residue_mapping": build_result.residue_mapping,
                "n_frames": build_result.n_frames,
                "residue_selection": selection_used,
            },
            indent=2,
        )
    )

    state_meta.descriptor_file = str(npz_path.relative_to(system_dir))
    state_meta.descriptor_metadata_file = str(meta_path.relative_to(system_dir))
    state_meta.n_frames = build_result.n_frames
    state_meta.residue_keys = build_result.residue_keys
    state_meta.residue_mapping = build_result.residue_mapping
    state_meta.residue_selection = residue_filter.strip() if residue_filter else None

    _refresh_system_metadata(system_meta)
    store.save_system(system_meta)


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
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
