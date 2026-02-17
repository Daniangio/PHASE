import json
import os
import shutil
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.v1.common import (
    build_state_descriptors,
    ensure_not_macro_locked,
    get_state_or_404,
    normalize_stride,
    project_store,
    refresh_system_metadata,
    serialize_system,
    stream_upload,
    stride_to_slice,
)
from phase.workflows.macro_states import register_state_from_pdb
from phase.common.slice_utils import parse_slice_spec
from backend.services.project_store import DescriptorState


router = APIRouter()


@router.get(
    "/projects/{project_id}/systems/{system_id}/structures/{state_id}",
    summary="Download the stored PDB file for a system state",
)
async def download_structure(project_id: str, system_id: str, state_id: str):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = get_state_or_404(system, state_id)
    if not state_meta.pdb_file:
        raise HTTPException(status_code=404, detail=f"No PDB stored for state '{state_id}'.")

    file_path = project_store.resolve_path(project_id, system_id, state_meta.pdb_file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored PDB file is missing on disk.")

    download_name = f"{state_meta.name}.pdb" if state_meta.name else os.path.basename(file_path)

    return FileResponse(
        file_path,
        filename=download_name,
        media_type="chemical/x-pdb",
    )


@router.get(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/descriptors/npz",
    summary="Download the descriptor NPZ for a system state",
)
async def download_state_descriptors(project_id: str, system_id: str, state_id: str):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = get_state_or_404(system, state_id)
    if not state_meta.descriptor_file:
        raise HTTPException(status_code=404, detail=f"No descriptors stored for state '{state_id}'.")

    file_path = project_store.resolve_path(project_id, system_id, state_meta.descriptor_file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Descriptor NPZ file is missing on disk.")

    base_name = state_meta.name or state_id
    download_name = f"{base_name}_descriptors.npz"

    return FileResponse(
        file_path,
        filename=download_name,
        media_type="application/octet-stream",
    )


@router.post(
    "/projects/{project_id}/systems/{system_id}/states",
    summary="Add a new state to an existing system",
)
async def add_system_state(
    project_id: str,
    system_id: str,
    name: str = Form(...),
    pdb: Optional[UploadFile] = File(None),
    source_state_id: Optional[str] = Form(None),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")
    ensure_not_macro_locked(system_meta)

    state_name = name.strip()
    if not state_name:
        raise HTTPException(status_code=400, detail="State name is required.")

    if not pdb and not source_state_id:
        raise HTTPException(status_code=400, detail="Provide a PDB file or choose an existing state to copy from.")

    dirs = project_store.ensure_directories(project_id, system_id)
    system_dir = dirs["system_dir"]
    structures_dir = dirs["structures_dir"]

    state_id = str(uuid.uuid4())
    pdb_path = structures_dir / f"{state_id}.pdb"

    if pdb:
        await stream_upload(pdb, pdb_path)
    else:
        source_state = get_state_or_404(system_meta, source_state_id)
        if not source_state.pdb_file:
            raise HTTPException(status_code=400, detail="Source state has no stored PDB to copy.")
        source_path = project_store.resolve_path(project_id, system_id, source_state.pdb_file)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source PDB file missing on disk.")
        shutil.copy(source_path, pdb_path)

    register_state_from_pdb(
        project_store,
        project_id,
        system_meta,
        state_id=state_id,
        name=state_name,
        pdb_path=pdb_path,
        stride=1,
    )
    return serialize_system(system_meta)


@router.post(
    "/projects/{project_id}/systems/{system_id}/states/rescan",
    summary="Rescan structures/descriptors on disk and sync states into system metadata",
)
async def rescan_states_from_disk(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    dirs = project_store.ensure_directories(project_id, system_id)
    system_dir = dirs["system_dir"]
    structures_dir = dirs["structures_dir"]
    descriptors_dir = dirs["descriptors_dir"]
    trajectories_dir = dirs["trajectories_dir"]

    states = dict(system_meta.states or {})
    discovered_ids: set[str] = set()
    added = 0
    updated = 0

    for pdb_path in sorted(structures_dir.glob("*.pdb")):
        state_id = pdb_path.stem
        discovered_ids.add(state_id)
        state_meta = states.get(state_id)
        is_new = False
        if not state_meta:
            state_meta = DescriptorState(state_id=state_id, name=state_id)
            states[state_id] = state_meta
            is_new = True

        new_pdb_rel = str(pdb_path.relative_to(system_dir))
        changed = bool(state_meta.pdb_file != new_pdb_rel)
        state_meta.pdb_file = new_pdb_rel
        if not state_meta.name:
            state_meta.name = state_id
            changed = True

        desc_npz = descriptors_dir / f"{state_id}_descriptors.npz"
        if desc_npz.exists():
            rel = str(desc_npz.relative_to(system_dir))
            if state_meta.descriptor_file != rel:
                state_meta.descriptor_file = rel
                changed = True

        desc_meta = descriptors_dir / f"{state_id}_descriptor_metadata.json"
        if desc_meta.exists():
            rel = str(desc_meta.relative_to(system_dir))
            if state_meta.descriptor_metadata_file != rel:
                state_meta.descriptor_metadata_file = rel
                changed = True
            try:
                payload = json.loads(desc_meta.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            keys = payload.get("descriptor_keys")
            if isinstance(keys, list):
                clean_keys = [str(v) for v in keys]
                if clean_keys != list(state_meta.residue_keys or []):
                    state_meta.residue_keys = clean_keys
                    changed = True
            mapping = payload.get("residue_mapping")
            if isinstance(mapping, dict):
                clean_map = {str(k): str(v) for k, v in mapping.items()}
                if clean_map != dict(state_meta.residue_mapping or {}):
                    state_meta.residue_mapping = clean_map
                    changed = True
            n_frames = payload.get("n_frames")
            if isinstance(n_frames, (int, float)):
                nf = int(n_frames)
                if nf >= 0 and state_meta.n_frames != nf:
                    state_meta.n_frames = nf
                    changed = True
            selection = payload.get("residue_selection")
            if isinstance(selection, str) and selection.strip() and not state_meta.residue_selection:
                state_meta.residue_selection = selection.strip()
                changed = True

        if not state_meta.trajectory_file:
            traj_candidates = sorted(p for p in trajectories_dir.glob(f"{state_id}.*") if p.is_file())
            if traj_candidates:
                rel = str(traj_candidates[0].relative_to(system_dir))
                state_meta.trajectory_file = rel
                state_meta.source_traj = traj_candidates[0].name
                changed = True

        if is_new:
            added += 1
        elif changed:
            updated += 1

    # If new states appeared while macro was locked, require reconfirmation.
    if added > 0 and getattr(system_meta, "macro_locked", False):
        system_meta.macro_locked = False
        system_meta.metastable_locked = False
        system_meta.analysis_mode = None

    system_meta.states = states
    refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)

    return {
        **serialize_system(system_meta),
        "rescan_summary": {
            "states_discovered_from_structures": len(discovered_ids),
            "states_added": added,
            "states_updated": updated,
        },
    }


@router.post(
    "/projects/{project_id}/systems/{system_id}/states/unlock-editing",
    summary="Unlock macro state editing and reset downstream locks",
)
async def unlock_macro_state_editing(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    system_meta.macro_locked = False
    system_meta.metastable_locked = False
    system_meta.analysis_mode = None
    project_store.save_system(system_meta)
    return serialize_system(system_meta)


@router.post(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory",
    summary="Upload/replace a trajectory for a state and rebuild descriptors",
)
async def upload_state_trajectory(
    project_id: str,
    system_id: str,
    state_id: str,
    trajectory: UploadFile = File(...),
    stride: int = Form(1),
    slice_spec: str = Form(None),
    residue_selection: str = Form(None),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")
    ensure_not_macro_locked(system_meta)

    state_meta = get_state_or_404(system_meta, state_id)
    if slice_spec:
        try:
            slice_spec, stride_val = parse_slice_spec(slice_spec)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid slice spec: {exc}") from exc
    else:
        stride_val = normalize_stride(state_meta.name, stride)
        slice_spec = stride_to_slice(stride_val)

    dirs = project_store.ensure_directories(project_id, system_id)
    system_dir = dirs["system_dir"]
    tmp_dir = dirs["tmp_dir"]

    traj_path = tmp_dir / f"{state_id}_{trajectory.filename or 'traj'}"
    await stream_upload(trajectory, traj_path)

    if state_meta.trajectory_file:
        old_traj = project_store.resolve_path(project_id, system_id, state_meta.trajectory_file)
        try:
            old_traj.unlink(missing_ok=True)
        except Exception:
            pass
        state_meta.trajectory_file = None
    state_meta.source_traj = trajectory.filename
    state_meta.stride = stride_val
    state_meta.slice_spec = slice_spec
    state_meta.residue_selection = residue_selection.strip() if residue_selection else None
    if state_meta.descriptor_file:
        old_descriptor = project_store.resolve_path(project_id, system_id, state_meta.descriptor_file)
        try:
            old_descriptor.unlink(missing_ok=True)
        except Exception:
            pass
    if state_meta.descriptor_metadata_file:
        old_meta = project_store.resolve_path(project_id, system_id, state_meta.descriptor_metadata_file)
        try:
            old_meta.unlink(missing_ok=True)
        except Exception:
            pass
    state_meta.descriptor_file = None
    state_meta.descriptor_metadata_file = None
    state_meta.residue_keys = []
    state_meta.residue_mapping = {}
    state_meta.n_frames = 0

    if not state_meta.pdb_file:
        raise HTTPException(status_code=400, detail="No stored PDB for this state. Upload PDB first.")

    project_store.save_system(system_meta)

    try:
        await build_state_descriptors(
            project_id,
            system_meta,
            state_meta,
            residue_filter=residue_selection,
            traj_path_override=traj_path,
        )
    except Exception as exc:
        system_meta.status = "failed"
        project_store.save_system(system_meta)
        raise HTTPException(status_code=500, detail=f"Descriptor build failed after upload: {exc}") from exc
    finally:
        try:
            traj_path.unlink(missing_ok=True)
        except Exception:
            pass

    return serialize_system(system_meta)


@router.delete(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory",
    summary="Delete the trajectory and descriptors for a state",
)
async def delete_state_trajectory(project_id: str, system_id: str, state_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")
    ensure_not_macro_locked(system_meta)

    state_meta = get_state_or_404(system_meta, state_id)

    if state_meta.descriptor_file:
        abs_path = project_store.resolve_path(project_id, system_id, state_meta.descriptor_file)
        try:
            abs_path.unlink(missing_ok=True)
        except Exception:
            pass
        state_meta.descriptor_file = None
        state_meta.n_frames = 0
    if state_meta.descriptor_metadata_file:
        meta_path = project_store.resolve_path(project_id, system_id, state_meta.descriptor_metadata_file)
        try:
            meta_path.unlink(missing_ok=True)
        except Exception:
            pass
        state_meta.descriptor_metadata_file = None
    if state_meta.trajectory_file:
        traj_path = project_store.resolve_path(project_id, system_id, state_meta.trajectory_file)
        try:
            traj_path.unlink(missing_ok=True)
        except Exception:
            pass
        state_meta.trajectory_file = None
        state_meta.source_traj = None

    state_meta.residue_keys = []
    state_meta.residue_mapping = {}
    state_meta.slice_spec = None
    state_meta.stride = 1

    refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)

    return serialize_system(system_meta)


@router.delete(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}",
    summary="Delete a state and its stored files",
)
async def delete_state(project_id: str, system_id: str, state_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")
    ensure_not_macro_locked(system_meta)

    state_meta = get_state_or_404(system_meta, state_id)

    for field in ("descriptor_file", "descriptor_metadata_file", "trajectory_file", "pdb_file"):
        rel_path = getattr(state_meta, field, None)
        if not rel_path:
            continue
        abs_path = project_store.resolve_path(project_id, system_id, rel_path)
        try:
            abs_path.unlink(missing_ok=True)
        except Exception:
            pass

    system_meta.states.pop(state_id, None)
    refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)
    return serialize_system(system_meta)


@router.patch(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}",
    summary="Rename a system state",
)
async def rename_system_state(
    project_id: str,
    system_id: str,
    state_id: str,
    payload: Dict[str, Any],
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")
    ensure_not_macro_locked(system_meta)

    new_name = (payload or {}).get("name")
    if not new_name or not str(new_name).strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    new_name = str(new_name).strip()

    state_meta = get_state_or_404(system_meta, state_id)
    state_meta.name = new_name

    project_store.save_system(system_meta)
    return serialize_system(system_meta)
