import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from backend.api.v1.analysis_cleanup import (
    cleanup_results_for_samples,
    cleanup_state_linked_artifacts,
    remove_md_samples_for_states,
)
from backend.api.v1.common import (
    build_state_descriptors,
    get_state_or_404,
    normalize_stride,
    project_store,
    refresh_system_metadata,
    serialize_system,
    stream_upload,
    stride_to_slice,
)
from phase.workflows.macro_states import allocate_state_storage_key, register_state_from_pdb
from phase.common.slice_utils import parse_slice_spec
from backend.services.project_store import DescriptorState


router = APIRouter()


def _unlink_if_inside_system(project_id: str, system_id: str, rel_path: Optional[str]) -> None:
    """Delete file only if it resolves under the system directory."""
    if not rel_path:
        return
    try:
        abs_path = project_store.resolve_path(project_id, system_id, rel_path).resolve()
    except Exception:
        return
    system_dir = project_store.ensure_directories(project_id, system_id)["system_dir"].resolve()
    try:
        abs_path.relative_to(system_dir)
    except ValueError:
        return
    try:
        abs_path.unlink(missing_ok=True)
    except Exception:
        pass


def _drop_state_metastable_data(system_meta, state_id: str) -> None:
    system_meta.metastable_states = [
        meta
        for meta in (system_meta.metastable_states or [])
        if str(meta.get("macro_state_id") or "") != str(state_id)
    ]
    if not system_meta.metastable_states:
        system_meta.metastable_locked = False
        system_meta.analysis_mode = "macro" if system_meta.macro_locked else None


def _resolve_state_file(project_id: str, system_id: str, file_ref: str):
    path = Path(file_ref)
    if path.is_absolute():
        return path
    return project_store.resolve_path(project_id, system_id, file_ref)


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
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory/frame",
    summary="Extract one stored state trajectory frame as PDB for 3D visualization",
)
async def download_state_trajectory_frame(project_id: str, system_id: str, state_id: str, frame: int = 0):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = get_state_or_404(system, state_id)
    if not state_meta.pdb_file:
        raise HTTPException(status_code=404, detail=f"No PDB stored for state '{state_id}'.")

    top_path = project_store.resolve_path(project_id, system_id, state_meta.pdb_file)
    if not top_path.exists():
        raise HTTPException(status_code=404, detail="Stored topology file is missing on disk.")
    traj_rel = state_meta.trajectory_file or state_meta.pdb_file
    traj_path = project_store.resolve_path(project_id, system_id, traj_rel)
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Stored trajectory file is missing on disk.")

    try:
        import tempfile
        import MDAnalysis as mda

        universe = mda.Universe(str(top_path), str(traj_path))
        n_frames = len(universe.trajectory)
        if n_frames <= 0:
            raise ValueError("Trajectory has no frames.")
        frame_i = max(0, min(int(frame), n_frames - 1))
        universe.trajectory[frame_i]
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            atoms = universe.atoms
            atoms.write(tmp_path)
            text = open(tmp_path, "r", encoding="utf-8", errors="replace").read()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract trajectory frame: {exc}") from exc

    return Response(
        content=text,
        media_type="chemical/x-pdb",
        headers={"X-PHASE-Frame": str(frame_i), "X-PHASE-Frame-Count": str(n_frames)},
    )


@router.get(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory/raw",
    summary="Download the raw stored state trajectory for Mol* trajectory playback",
)
async def download_state_trajectory_raw(project_id: str, system_id: str, state_id: str):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = get_state_or_404(system, state_id)
    if not state_meta.trajectory_file:
        raise HTTPException(status_code=404, detail=f"No trajectory stored for state '{state_id}'.")

    file_path = _resolve_state_file(project_id, system_id, state_meta.trajectory_file)
    if not file_path.exists():
        if Path(state_meta.trajectory_file).is_absolute():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Stored trajectory points to an absolute host path that is not visible inside the backend "
                    "container. Re-upload the trajectory from the System page so it is copied into the shared "
                    "PHASE data root, or bind-mount that host path into the backend/worker containers."
                ),
            )
        raise HTTPException(status_code=404, detail="Stored trajectory file is missing on disk.")

    download_name = state_meta.source_traj or os.path.basename(file_path)
    return FileResponse(
        file_path,
        filename=download_name,
        media_type="application/octet-stream",
    )


@router.get(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory/raw/info",
    summary="Check whether the raw stored state trajectory is reachable by the backend",
)
async def get_state_trajectory_raw_info(project_id: str, system_id: str, state_id: str):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = get_state_or_404(system, state_id)
    if not state_meta.trajectory_file:
        return {
            "available": False,
            "detail": f"No trajectory stored for state '{state_id}'.",
            "trajectory_file": None,
            "absolute_path": False,
        }

    file_path = _resolve_state_file(project_id, system_id, state_meta.trajectory_file)
    exists = file_path.exists()
    detail = "Trajectory is reachable by the backend."
    if not exists and Path(state_meta.trajectory_file).is_absolute():
        detail = (
            "Stored trajectory points to an absolute host path that is not visible inside the backend container. "
            "Re-upload the trajectory from the System page so it is copied into the shared PHASE data root, "
            "or bind-mount that host path into the backend/worker containers."
        )
    elif not exists:
        detail = "Stored trajectory file is missing on disk."

    return {
        "available": bool(exists),
        "detail": detail,
        "trajectory_file": state_meta.trajectory_file,
        "source_traj": state_meta.source_traj,
        "absolute_path": Path(state_meta.trajectory_file).is_absolute(),
        "size_bytes": file_path.stat().st_size if exists else None,
    }


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
    resid_shift: Optional[int] = Form(None),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_name = name.strip()
    if not state_name:
        raise HTTPException(status_code=400, detail="State name is required.")

    if not pdb and not source_state_id:
        raise HTTPException(status_code=400, detail="Provide a PDB file or choose an existing state to copy from.")

    state_id = str(uuid.uuid4())
    storage_key = allocate_state_storage_key(system_meta, state_name, state_id)
    state_dirs = project_store.ensure_state_directories(project_id, system_id, state_id, storage_key=storage_key)
    shift_value = int(resid_shift or 0)
    if pdb:
        pdb_ext = os.path.splitext(pdb.filename or "state.pdb")[1] or ".pdb"
        pdb_path = state_dirs["state_dir"] / f"structure{pdb_ext}"
        await stream_upload(pdb, pdb_path)
    else:
        source_state = get_state_or_404(system_meta, source_state_id)
        if not source_state.pdb_file:
            raise HTTPException(status_code=400, detail="Source state has no stored PDB to copy.")
        source_path = project_store.resolve_path(project_id, system_id, source_state.pdb_file)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source PDB file missing on disk.")
        pdb_ext = source_path.suffix or ".pdb"
        pdb_path = state_dirs["state_dir"] / f"structure{pdb_ext}"
        shutil.copy(source_path, pdb_path)
        if resid_shift is None:
            shift_value = int(getattr(source_state, "resid_shift", 0) or 0)

    register_state_from_pdb(
        project_store,
        project_id,
        system_meta,
        state_id=state_id,
        name=state_name,
        pdb_path=pdb_path,
        stride=1,
        resid_shift=shift_value,
        storage_key=storage_key,
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
    current_ids = set((system_meta.states or {}).keys())
    refreshed = project_store.get_system(project_id, system_id)
    discovered_ids = set((refreshed.states or {}).keys())
    added = len(discovered_ids - current_ids)
    updated = len(discovered_ids & current_ids)

    if added > 0 and getattr(refreshed, "macro_locked", False):
        refreshed.macro_locked = False
        refreshed.metastable_locked = False
        refreshed.analysis_mode = None

    refresh_system_metadata(refreshed)
    project_store.save_system(refreshed)

    return {
        **serialize_system(refreshed),
        "rescan_summary": {
            "states_discovered_from_structures": len(discovered_ids),
            "states_added": added,
            "states_updated": updated,
        },
    }


@router.post(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory",
    summary="Upload/replace a trajectory for a state and rebuild descriptors",
)
async def upload_state_trajectory(
    project_id: str,
    system_id: str,
    state_id: str,
    trajectory: Optional[UploadFile] = File(None),
    stride: int = Form(1),
    slice_spec: str = Form(None),
    residue_selection: str = Form(None),
    resid_shift: Optional[int] = Form(None),
    build_descriptors_after_upload: bool = Form(True),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = get_state_or_404(system_meta, state_id)
    if slice_spec:
        try:
            slice_spec, stride_val = parse_slice_spec(slice_spec)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid slice spec: {exc}") from exc
    else:
        stride_val = normalize_stride(state_meta.name, stride)
        slice_spec = stride_to_slice(stride_val)

    state_dirs = project_store.ensure_state_directories(
        project_id,
        system_id,
        state_meta.state_id,
        storage_key=state_meta.storage_key or state_meta.state_id,
    )
    system_dir = state_dirs["system_dir"]
    traj_path_override = None
    if trajectory is not None:
        traj_ext = os.path.splitext(trajectory.filename or "traj.xtc")[1] or ".xtc"
        traj_path = state_dirs["state_dir"] / f"trajectory{traj_ext}"
        await stream_upload(trajectory, traj_path)

        if state_meta.trajectory_file:
            _unlink_if_inside_system(project_id, system_id, state_meta.trajectory_file)
        state_meta.source_traj = trajectory.filename
        state_meta.trajectory_file = str(traj_path.relative_to(system_dir))
        traj_path_override = traj_path
    else:
        if not state_meta.pdb_file:
            raise HTTPException(status_code=400, detail="No stored PDB for this state. Upload PDB first.")
        traj_path_override = project_store.resolve_path(project_id, system_id, state_meta.pdb_file)
        if not traj_path_override.exists():
            raise HTTPException(status_code=404, detail="Stored PDB file is missing on disk.")
    state_meta.stride = stride_val
    state_meta.slice_spec = slice_spec
    state_meta.resid_shift = int(state_meta.resid_shift if resid_shift is None else resid_shift)
    state_meta.residue_selection = residue_selection.strip() if residue_selection else None

    if build_descriptors_after_upload:
        if state_meta.descriptor_file:
            _unlink_if_inside_system(project_id, system_id, state_meta.descriptor_file)
        if state_meta.descriptor_metadata_file:
            _unlink_if_inside_system(project_id, system_id, state_meta.descriptor_metadata_file)
        state_meta.descriptor_file = None
        state_meta.descriptor_metadata_file = None
        if state_meta.metastable_metadata_file:
            _unlink_if_inside_system(project_id, system_id, state_meta.metastable_metadata_file)
            state_meta.metastable_metadata_file = None
        if state_meta.metastable_labels_file:
            _unlink_if_inside_system(project_id, system_id, state_meta.metastable_labels_file)
            state_meta.metastable_labels_file = None
        _drop_state_metastable_data(system_meta, state_id)
        state_meta.residue_keys = []
        state_meta.residue_mapping = {}
        state_meta.n_frames = 0

    if not state_meta.pdb_file:
        raise HTTPException(status_code=400, detail="No stored PDB for this state. Upload PDB first.")

    project_store.save_system(system_meta)

    if not build_descriptors_after_upload:
        refresh_system_metadata(system_meta)
        project_store.save_system(system_meta)
        return serialize_system(system_meta)

    try:
        await build_state_descriptors(
            project_id,
            system_meta,
            state_meta,
            residue_filter=residue_selection,
            resid_shift=state_meta.resid_shift,
            traj_path_override=traj_path_override,
        )
    except Exception as exc:
        system_meta.status = "failed"
        project_store.save_system(system_meta)
        action = "upload" if trajectory is not None else "PDB-only build"
        raise HTTPException(status_code=500, detail=f"Descriptor build failed after {action}: {exc}") from exc
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

    state_meta = get_state_or_404(system_meta, state_id)

    if state_meta.descriptor_file:
        _unlink_if_inside_system(project_id, system_id, state_meta.descriptor_file)
        state_meta.descriptor_file = None
        state_meta.n_frames = 0
    if state_meta.descriptor_metadata_file:
        _unlink_if_inside_system(project_id, system_id, state_meta.descriptor_metadata_file)
        state_meta.descriptor_metadata_file = None
    if state_meta.trajectory_file:
        _unlink_if_inside_system(project_id, system_id, state_meta.trajectory_file)
        state_meta.trajectory_file = None
        state_meta.source_traj = None
    if state_meta.metastable_metadata_file:
        _unlink_if_inside_system(project_id, system_id, state_meta.metastable_metadata_file)
        state_meta.metastable_metadata_file = None
    if state_meta.metastable_labels_file:
        _unlink_if_inside_system(project_id, system_id, state_meta.metastable_labels_file)
        state_meta.metastable_labels_file = None
    _drop_state_metastable_data(system_meta, state_id)

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

    state_meta = get_state_or_404(system_meta, state_id)
    linked_state_ids = {str(state_id)}
    linked_state_ids.update(
        str(meta.get("metastable_id") or meta.get("id"))
        for meta in (system_meta.metastable_states or [])
        if isinstance(meta, dict)
        and str(meta.get("macro_state_id") or "") == str(state_id)
        and (meta.get("metastable_id") or meta.get("id"))
    )

    for field in ("descriptor_file", "descriptor_metadata_file", "trajectory_file", "pdb_file"):
        _unlink_if_inside_system(project_id, system_id, getattr(state_meta, field, None))
    if state_meta.metastable_metadata_file:
        _unlink_if_inside_system(project_id, system_id, state_meta.metastable_metadata_file)
    if state_meta.metastable_labels_file:
        _unlink_if_inside_system(project_id, system_id, state_meta.metastable_labels_file)

    try:
        state_dir = project_store.ensure_state_directories(
            project_id,
            system_id,
            state_meta.state_id,
            storage_key=state_meta.storage_key or state_meta.state_id,
        )["state_dir"]
        shutil.rmtree(state_dir, ignore_errors=True)
    except Exception:
        pass

    _drop_state_metastable_data(system_meta, state_id)
    system_meta.states.pop(state_id, None)
    refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)
    md_cleanup = remove_md_samples_for_states(project_id, system_id, linked_state_ids)
    sample_results_removed = cleanup_results_for_samples(
        project_id,
        system_id,
        md_cleanup.get("removed_md_sample_ids") or [],
    )
    cleanup = cleanup_state_linked_artifacts(project_id, system_id)
    return {
        **serialize_system(system_meta),
        "cleanup_summary": {
            **cleanup,
            **md_cleanup,
            "sample_results_removed": sample_results_removed,
        },
    }


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

    new_name = (payload or {}).get("name")
    if not new_name or not str(new_name).strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    new_name = str(new_name).strip()

    state_meta = get_state_or_404(system_meta, state_id)
    state_meta.name = new_name

    project_store.save_system(system_meta)
    return serialize_system(system_meta)
