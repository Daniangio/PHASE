import shutil
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.api.v1.common import (
    ensure_not_metastable_locked,
    project_store,
    serialize_system,
)
from backend.services.metastable import recompute_metastable_states


router = APIRouter()


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/recompute",
    summary="Recompute metastable states across all descriptor-ready trajectories",
)
async def recompute_metastable(
    project_id: str,
    system_id: str,
    n_microstates: int = Query(20, ge=2, le=500),
    k_meta_min: int = Query(1, ge=1, le=10),
    k_meta_max: int = Query(4, ge=1, le=10),
    tica_lag_frames: int = Query(5, ge=1),
    tica_dim: int = Query(5, ge=1),
    random_state: int = Query(0),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro-states before running metastable analysis.")
    if getattr(system_meta, "analysis_mode", None) == "macro":
        raise HTTPException(status_code=400, detail="System is locked to macro-only analysis.")
    ensure_not_metastable_locked(system_meta)

    try:
        result = await run_in_threadpool(
            recompute_metastable_states,
            project_id,
            system_id,
            n_microstates=n_microstates,
            k_meta_min=k_meta_min,
            k_meta_max=max(k_meta_min, k_meta_max),
            tica_lag_frames=tica_lag_frames,
            tica_dim=tica_dim,
            random_state=random_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metastable recompute failed: {exc}") from exc

    return result


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable",
    summary="List metastable states for a system",
)
async def list_metastable_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    return {
        "metastable_states": system_meta.metastable_states or [],
        "model_dir": system_meta.metastable_model_dir,
    }


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clear",
    summary="Clear metastable states, labels, and clusters",
)
async def clear_metastable_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if getattr(system_meta, "metastable_locked", False):
        raise HTTPException(status_code=400, detail="Unlock metastable states before clearing.")

    for cluster in system_meta.metastable_clusters or []:
        rel_path = cluster.get("path")
        if not rel_path:
            continue
        abs_path = project_store.resolve_path(project_id, system_id, rel_path)
        try:
            abs_path.unlink(missing_ok=True)
        except Exception:
            pass

    if system_meta.metastable_model_dir:
        model_dir = project_store.resolve_path(project_id, system_id, system_meta.metastable_model_dir)
        try:
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass

    for state in system_meta.states.values():
        if state.metastable_labels_file:
            label_path = project_store.resolve_path(project_id, system_id, state.metastable_labels_file)
            try:
                label_path.unlink(missing_ok=True)
            except Exception:
                pass
            state.metastable_labels_file = None

        if state.descriptor_file:
            descriptor_path = project_store.resolve_path(project_id, system_id, state.descriptor_file)
            if descriptor_path.exists():
                try:
                    npz = np.load(descriptor_path, allow_pickle=True)
                    if "metastable_labels" in npz.files:
                        data = {k: npz[k] for k in npz.files if k != "metastable_labels"}
                        tmp_path = descriptor_path.with_suffix(".tmp.npz")
                        np.savez_compressed(tmp_path, **data)
                        tmp_path.replace(descriptor_path)
                except Exception:
                    pass

    system_meta.metastable_states = []
    system_meta.metastable_clusters = []
    system_meta.metastable_model_dir = None
    system_meta.metastable_locked = False
    system_meta.analysis_mode = "macro"
    project_store.save_system(system_meta)
    return serialize_system(system_meta)


@router.post(
    "/projects/{project_id}/systems/{system_id}/states/confirm",
    summary="Lock macro-states to proceed to metastable analysis",
)
async def confirm_macro_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if getattr(system_meta, "macro_locked", False):
        return serialize_system(system_meta)

    if not system_meta.states:
        raise HTTPException(status_code=400, detail="Add at least one state before locking.")

    system_meta.macro_locked = True
    project_store.save_system(system_meta)
    return serialize_system(system_meta)


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/confirm",
    summary="Lock metastable states to proceed to clustering and analysis",
)
async def confirm_metastable_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro-states first.")

    if getattr(system_meta, "metastable_locked", False):
        return serialize_system(system_meta)

    if not (system_meta.metastable_states or []):
        raise HTTPException(status_code=400, detail="Run metastable recompute before locking.")

    system_meta.metastable_locked = True
    system_meta.analysis_mode = "metastable"
    project_store.save_system(system_meta)
    return serialize_system(system_meta)


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/{metastable_id}/pdb",
    summary="Download representative PDB for a metastable state",
)
async def fetch_metastable_pdb(project_id: str, system_id: str, metastable_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    metas = system_meta.metastable_states or []
    target = next((m for m in metas if m.get("metastable_id") == metastable_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Metastable state '{metastable_id}' not found.")
    pdb_rel = target.get("representative_pdb")
    if not pdb_rel:
        raise HTTPException(status_code=404, detail="No representative PDB stored for this metastable state.")
    pdb_path = project_store.resolve_path(project_id, system_id, pdb_rel)
    if not pdb_path.exists():
        raise HTTPException(status_code=404, detail="Representative PDB file is missing on disk.")
    return FileResponse(pdb_path, filename=pdb_path.name, media_type="chemical/x-pdb")


@router.patch(
    "/projects/{project_id}/systems/{system_id}/metastable/{metastable_id}",
    summary="Rename a metastable state",
)
async def rename_metastable_state(
    project_id: str,
    system_id: str,
    metastable_id: str,
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

    updated = False
    metas = system_meta.metastable_states or []
    for meta in metas:
        if meta.get("metastable_id") == metastable_id:
            meta["name"] = new_name
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail=f"Metastable state '{metastable_id}' not found.")

    system_meta.metastable_states = metas
    project_store.save_system(system_meta)
    return {"metastable_states": metas}
