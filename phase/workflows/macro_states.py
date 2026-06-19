from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from phase.common.selection_utils import build_residue_selection_config
from phase.common.slice_utils import parse_slice_spec
from phase.io.descriptors import save_descriptor_npz
from phase.services.project_store import DescriptorState, ProjectStore, SystemMetadata
from phase.services.state_utils import build_analysis_states
from phase.workflows.descriptors import DescriptorPreprocessor


_RESIDUE_KEY_PATTERN = re.compile(r"^res_(-?\d+(?:_-?\d+)*)$")
_RESID_SELECTION_PATTERN = re.compile(r"\bresid\s+((?:-?\d+\s*)+)")


def _slugify_state_storage_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def allocate_state_storage_key(system_meta: SystemMetadata, state_name: str, fallback: str) -> str:
    existing = {
        str(getattr(state, "storage_key", None) or getattr(state, "state_id", sid))
        for sid, state in (system_meta.states or {}).items()
    }
    base = _slugify_state_storage_key(state_name) or _slugify_state_storage_key(fallback) or str(fallback)
    candidate = base
    idx = 2
    while candidate in existing:
        candidate = f"{base}_{idx}"
        idx += 1
    return candidate


def _shift_residue_key(key: str, resid_shift: int) -> str:
    if resid_shift == 0:
        return key
    match = _RESIDUE_KEY_PATTERN.match(str(key))
    if not match:
        return key
    parts = match.group(1).split("_")
    try:
        shifted = [str(int(part) + resid_shift) for part in parts]
    except ValueError:
        return key
    return f"res_{'_'.join(shifted)}"


def _shift_residue_key_from_mapping(key: str, selection: Optional[str], resid_shift: int) -> str:
    if resid_shift == 0 or not isinstance(selection, str) or not selection.strip():
        return _shift_residue_key(key, resid_shift)
    match = _RESID_SELECTION_PATTERN.search(selection)
    if not match:
        return _shift_residue_key(key, resid_shift)
    try:
        numbers = [int(v) for v in re.findall(r"-?\d+", match.group(1))]
    except ValueError:
        return _shift_residue_key(key, resid_shift)
    if not numbers:
        return _shift_residue_key(key, resid_shift)
    shifted = [str(v + resid_shift) for v in numbers]
    return f"res_{'_'.join(shifted)}"


def _apply_residue_shift(
    *,
    features: Dict[str, Any],
    residue_keys: list[str],
    residue_mapping: Dict[str, str],
    resid_shift: int,
) -> tuple[Dict[str, Any], list[str], Dict[str, str]]:
    if resid_shift == 0:
        return features, residue_keys, residue_mapping

    shifted_features: Dict[str, Any] = {}
    shifted_mapping: Dict[str, str] = {}
    key_translation: Dict[str, str] = {}

    for key in residue_keys:
        old_key = str(key)
        shifted_key = _shift_residue_key_from_mapping(
            old_key,
            (residue_mapping or {}).get(old_key),
            resid_shift,
        )
        if shifted_key in key_translation.values() and key_translation.get(old_key) != shifted_key:
            raise ValueError(f"Residue shift created duplicate residue key '{shifted_key}'.")
        key_translation[old_key] = shifted_key

    for key, value in features.items():
        old_key = str(key)
        shifted_key = key_translation.get(old_key, _shift_residue_key(old_key, resid_shift))
        if shifted_key in shifted_features and shifted_key != old_key:
            raise ValueError(f"Residue shift created duplicate feature key '{shifted_key}'.")
        shifted_features[shifted_key] = value

    for key, selection in (residue_mapping or {}).items():
        old_key = str(key)
        shifted_key = key_translation.get(old_key, _shift_residue_key(old_key, resid_shift))
        shifted_mapping[shifted_key] = str(selection)

    shifted_keys = [key_translation.get(str(key), _shift_residue_key(str(key), resid_shift)) for key in residue_keys]

    return shifted_features, shifted_keys, shifted_mapping


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
    state_dir: Path,
    slice_spec: Optional[str],
    resid_shift: int,
    state_id: str,
    state_name: Optional[str],
    selection_used: str,
) -> Tuple[Any, Dict[str, Path]]:
    build_result = preprocessor.build_single(str(traj_path), str(pdb_path), slice_spec)
    (
        build_result.features,
        build_result.residue_keys,
        build_result.residue_mapping,
    ) = _apply_residue_shift(
        features=build_result.features,
        residue_keys=list(build_result.residue_keys),
        residue_mapping=dict(build_result.residue_mapping),
        resid_shift=resid_shift,
    )
    artifact_paths = {
        "npz": state_dir / "descriptors.npz",
        "metadata": state_dir / "descriptor_metadata.json",
    }
    save_descriptor_npz(artifact_paths["npz"], build_result.features)
    metadata_payload = {
        "state_id": state_id,
        "state_name": state_name or state_id,
        "descriptor_keys": build_result.residue_keys,
        "residue_mapping": build_result.residue_mapping,
        "n_frames": build_result.n_frames,
        "residue_selection": selection_used,
        "resid_shift": int(resid_shift),
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
    resid_shift: Optional[int] = None,
    traj_path_override: Optional[Path] = None,
) -> SystemMetadata:
    if not state_meta.pdb_file:
        raise ValueError("No PDB stored for this state.")

    dirs = store.ensure_directories(project_id, system_meta.system_id)
    system_dir = dirs["system_dir"]
    state_dirs = store.ensure_state_directories(
        project_id,
        system_meta.system_id,
        state_meta.state_id,
        storage_key=state_meta.storage_key or state_meta.state_id,
    )
    state_dir = state_dirs["state_dir"]

    pdb_path = store.resolve_path(project_id, system_meta.system_id, state_meta.pdb_file)
    if traj_path_override is not None:
        traj_path = traj_path_override
    elif state_meta.trajectory_file:
        traj_path = store.resolve_path(project_id, system_meta.system_id, state_meta.trajectory_file)
    else:
        # PDB-only state: treat the structure as a one-frame trajectory.
        traj_path = pdb_path

    if not pdb_path.exists():
        raise FileNotFoundError("Stored PDB file missing on disk.")
    if not traj_path.exists():
        raise FileNotFoundError("Stored trajectory file missing on disk.")

    selection_used, selections_config = _resolve_selection_config(system_meta, residue_filter)
    preprocessor = DescriptorPreprocessor(residue_selections=selections_config)
    shift_value = int(state_meta.resid_shift if resid_shift is None else resid_shift)
    build_result, artifact_paths = _build_state_artifacts(
        preprocessor,
        traj_path=traj_path,
        pdb_path=pdb_path,
        state_dir=state_dir,
        slice_spec=state_meta.slice_spec,
        resid_shift=shift_value,
        state_id=state_meta.state_id,
        state_name=state_meta.name,
        selection_used=selection_used,
    )

    rel_npz = str(artifact_paths["npz"].relative_to(system_dir))
    rel_meta = str(artifact_paths["metadata"].relative_to(system_dir))

    state_meta.descriptor_file = rel_npz
    state_meta.descriptor_metadata_file = rel_meta
    state_meta.n_frames = build_result.n_frames
    state_meta.residue_keys = build_result.residue_keys
    state_meta.residue_mapping = build_result.residue_mapping
    state_meta.resid_shift = shift_value
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
    traj_path: Optional[Path],
    residue_selection: Optional[str],
    copy_traj: bool,
    build_descriptors: bool,
    slice_spec: Optional[str],
    resid_shift: int = 0,
) -> DescriptorState:
    system = store.get_system(project_id, system_id)

    if state_id in system.states:
        raise ValueError(f"State '{state_id}' already exists.")

    state_name = name or state_id
    storage_key = allocate_state_storage_key(system, state_name, state_id)
    dirs = store.ensure_state_directories(project_id, system_id, state_id, storage_key=storage_key)
    pdb_ext = pdb_path.suffix or ".pdb"
    traj_ext = traj_path.suffix if traj_path is not None else ".xtc"

    pdb_dest = dirs["state_dir"] / f"structure{pdb_ext}"
    traj_dest = dirs["state_dir"] / f"trajectory{traj_ext}"

    pdb_dest.parent.mkdir(parents=True, exist_ok=True)
    traj_dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(pdb_path, pdb_dest)
        traj_value = None
        if traj_path is not None:
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
            name=state_name,
            pdb_file=str(pdb_dest.relative_to(dirs["system_dir"])),
            trajectory_file=traj_value,
            residue_selection=residue_selection,
            slice_spec=slice_value,
            stride=stride_val,
            resid_shift=int(resid_shift),
            storage_key=storage_key,
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
                resid_shift=int(resid_shift),
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
    resid_shift: int = 0,
    storage_key: Optional[str] = None,
) -> DescriptorState:
    if state_id in system_meta.states:
        raise ValueError(f"State '{state_id}' already exists.")
    storage_key = storage_key or allocate_state_storage_key(system_meta, name, state_id)
    dirs = store.ensure_state_directories(project_id, system_meta.system_id, state_id, storage_key=storage_key)
    system_dir = dirs["system_dir"]
    rel_pdb = str(pdb_path)
    if pdb_path.is_absolute():
        rel_pdb = str(pdb_path.relative_to(system_dir))
    state = DescriptorState(
        state_id=state_id,
        name=name,
        pdb_file=rel_pdb,
        stride=stride,
        resid_shift=int(resid_shift),
        storage_key=storage_key,
    )
    system_meta.states[state_id] = state
    refresh_system_metadata(system_meta)
    store.save_system(system_meta)
    return state
