from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from phase.common.selection_utils import build_residue_selection_config
from phase.common.slice_utils import parse_slice_spec
from phase.io.descriptors import save_descriptor_npz
from phase.services.project_store import DescriptorState, ProjectStore, SystemMetadata
from phase.services.state_utils import build_analysis_states
from phase.workflows.descriptors import DescriptorPreprocessor


def update_system_status(system_meta: SystemMetadata) -> None:
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


def refresh_system_metadata(system_meta: SystemMetadata) -> None:
    all_keys = set()
    for state in system_meta.states.values():
        all_keys.update(state.residue_keys or [])
    system_meta.descriptor_keys = sorted(all_keys)
    system_meta.analysis_states = build_analysis_states(system_meta)
    update_system_status(system_meta)


def _resolve_selection_config(
    system_meta: SystemMetadata,
    residue_filter: Optional[str],
) -> Tuple[str, Optional[Dict[str, str] | list[str]]]:
    selection_used = "protein"
    if residue_filter is not None and residue_filter.strip():
        selection_used = f"protein and ({residue_filter.strip()})"
    elif system_meta.residue_selections:
        selection_used = "system_selections"
    selections_config = build_residue_selection_config(
        base_selections=system_meta.residue_selections,
        residue_filter=residue_filter,
    )
    return selection_used, selections_config


def _build_state_artifacts(
    preprocessor: DescriptorPreprocessor,
    *,
    traj_path: Path,
    pdb_path: Path,
    descriptors_dir: Path,
    slice_spec: Optional[str],
    state_id: str,
    selection_used: str,
) -> Tuple[Any, Dict[str, Path]]:
    build_result = preprocessor.build_single(str(traj_path), str(pdb_path), slice_spec)
    artifact_paths = {
        "npz": descriptors_dir / f"{state_id}_descriptors.npz",
        "metadata": descriptors_dir / f"{state_id}_descriptor_metadata.json",
    }
    save_descriptor_npz(artifact_paths["npz"], build_result.features)
    metadata_payload = {
        "descriptor_keys": build_result.residue_keys,
        "residue_mapping": build_result.residue_mapping,
        "n_frames": build_result.n_frames,
        "residue_selection": selection_used,
    }
    artifact_paths["metadata"].write_text(json.dumps(metadata_payload, indent=2))
    return build_result, artifact_paths


def build_state_descriptors(
    store: ProjectStore,
    project_id: str,
    system_meta: SystemMetadata,
    state_meta: DescriptorState,
    *,
    residue_filter: Optional[str] = None,
    traj_path_override: Optional[Path] = None,
) -> SystemMetadata:
    if not state_meta.trajectory_file and traj_path_override is None:
        raise ValueError("No trajectory stored for this state.")
    if not state_meta.pdb_file:
        raise ValueError("No PDB stored for this state.")

    dirs = store.ensure_directories(project_id, system_meta.system_id)
    system_dir = dirs["system_dir"]
    descriptors_dir = dirs["descriptors_dir"]

    if traj_path_override is not None:
        traj_path = traj_path_override
    else:
        traj_path = store.resolve_path(project_id, system_meta.system_id, state_meta.trajectory_file)
    pdb_path = store.resolve_path(project_id, system_meta.system_id, state_meta.pdb_file)

    if not traj_path.exists():
        raise FileNotFoundError("Stored trajectory file missing on disk.")
    if not pdb_path.exists():
        raise FileNotFoundError("Stored PDB file missing on disk.")

    selection_used, selections_config = _resolve_selection_config(system_meta, residue_filter)
    preprocessor = DescriptorPreprocessor(residue_selections=selections_config)
    build_result, artifact_paths = _build_state_artifacts(
        preprocessor,
        traj_path=traj_path,
        pdb_path=pdb_path,
        descriptors_dir=descriptors_dir,
        slice_spec=state_meta.slice_spec,
        state_id=state_meta.state_id,
        selection_used=selection_used,
    )

    rel_npz = str(artifact_paths["npz"].relative_to(system_dir))
    rel_meta = str(artifact_paths["metadata"].relative_to(system_dir))

    state_meta.descriptor_file = rel_npz
    state_meta.descriptor_metadata_file = rel_meta
    state_meta.n_frames = build_result.n_frames
    state_meta.residue_keys = build_result.residue_keys
    state_meta.residue_mapping = build_result.residue_mapping
    state_meta.residue_selection = residue_filter.strip() if residue_filter else None

    refresh_system_metadata(system_meta)
    store.save_system(system_meta)
    return system_meta


def add_state(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    state_id: str,
    name: Optional[str],
    pdb_path: Path,
    traj_path: Path,
    residue_selection: Optional[str],
    copy_traj: bool,
    build_descriptors: bool,
    slice_spec: Optional[str],
) -> DescriptorState:
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
        refresh_system_metadata(system)
        store.save_system(system)
        if build_descriptors:
            build_state_descriptors(
                store,
                project_id,
                system,
                state,
                residue_filter=residue_selection,
            )
    except Exception:
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
        raise

    return state


def register_state_from_pdb(
    store: ProjectStore,
    project_id: str,
    system_meta: SystemMetadata,
    *,
    state_id: str,
    name: str,
    pdb_path: Path,
    stride: int = 1,
) -> DescriptorState:
    if state_id in system_meta.states:
        raise ValueError(f"State '{state_id}' already exists.")
    dirs = store.ensure_directories(project_id, system_meta.system_id)
    system_dir = dirs["system_dir"]
    rel_pdb = str(pdb_path)
    if pdb_path.is_absolute():
        rel_pdb = str(pdb_path.relative_to(system_dir))
    state = DescriptorState(
        state_id=state_id,
        name=name,
        pdb_file=rel_pdb,
        stride=stride,
    )
    system_meta.states[state_id] = state
    refresh_system_metadata(system_meta)
    store.save_system(system_meta)
    return state
