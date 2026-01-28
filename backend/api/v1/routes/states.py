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
from backend.services.slice_utils import parse_slice_spec
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

    system_meta.states[state_id] = DescriptorState(
        state_id=state_id,
        name=state_name,
        pdb_file=str(pdb_path.relative_to(system_dir)),
        stride=1,
    )
    refresh_system_metadata(system_meta)
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
    traj_dir = dirs["trajectories_dir"]
    system_dir = dirs["system_dir"]

    traj_path = traj_dir / f"{state_id}_{trajectory.filename or 'traj'}"
    await stream_upload(trajectory, traj_path)

    state_meta.trajectory_file = str(traj_path.relative_to(system_dir))
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
        )
    except Exception as exc:
        system_meta.status = "failed"
        project_store.save_system(system_meta)
        raise HTTPException(status_code=500, detail=f"Descriptor build failed after upload: {exc}") from exc

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
