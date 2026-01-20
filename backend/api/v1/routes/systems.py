import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.api.v1.common import (
    parse_residue_selections,
    project_store,
    refresh_system_metadata,
    serialize_system,
    stream_upload,
)
from backend.services.project_store import DescriptorState


router = APIRouter()


@router.get("/projects/{project_id}/systems/{system_id}", summary="Get system metadata")
async def get_system_detail(project_id: str, system_id: str):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{system_id}' not found in project '{project_id}'.",
        )
    return serialize_system(system)


@router.post(
    "/projects/{project_id}/systems",
    summary="Create a system with one or more PDB states",
)
async def create_system_with_descriptors(
    project_id: str,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    state_names: Optional[str] = Form(None),
    residue_selections_text: Optional[str] = Form(None),
    residue_selections_json: Optional[str] = Form(None),
    pdb_files: List[UploadFile] = File(...),
):
    try:
        project_store.get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    if not pdb_files:
        raise HTTPException(status_code=400, detail="At least one PDB file is required.")

    try:
        parsed_names = json.loads(state_names) if state_names else []
        if not isinstance(parsed_names, list):
            parsed_names = []
    except Exception:
        parsed_names = []

    raw_selection_input = residue_selections_text or residue_selections_json
    residue_selections = parse_residue_selections(
        raw_selection_input,
        expect_json=bool(residue_selections_json and not residue_selections_text),
    )
    try:
        system_meta = project_store.create_system(
            project_id, name=name, description=description, residue_selections=residue_selections
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to create system: {exc}") from exc

    dirs = project_store.ensure_directories(project_id, system_meta.system_id)
    system_dir = dirs["system_dir"]
    structures_dir = dirs["structures_dir"]
    print(f"[system-create] Uploading {len(pdb_files)} states for project={project_id} system={system_meta.system_id}")
    for idx, upload in enumerate(pdb_files):
        state_id = str(uuid.uuid4())
        state_name = None
        if idx < len(parsed_names) and isinstance(parsed_names[idx], str):
            state_name = parsed_names[idx].strip() or None
        state_name = (state_name or (upload.filename or f"State {idx + 1}")).strip()
        pdb_path = structures_dir / f"{state_id}.pdb"
        await stream_upload(upload, pdb_path)

        system_meta.states[state_id] = DescriptorState(
            state_id=state_id,
            name=state_name,
            pdb_file=str(pdb_path.relative_to(system_dir)),
            stride=1,
        )

    refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)
    return serialize_system(system_meta)


@router.delete("/projects/{project_id}/systems/{system_id}", summary="Delete a system")
async def delete_system(project_id: str, system_id: str):
    try:
        project_store.delete_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{system_id}' not found in project '{project_id}'.",
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to delete system: {exc}") from exc
    return {"status": "deleted", "system_id": system_id}
