from datetime import datetime
from pathlib import Path
import re
import shutil

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.api.v1.schemas import ProjectCreateRequest
from backend.api.v1.common import DATA_ROOT, project_store, serialize_project, serialize_system


router = APIRouter()

def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_")


def _unique_id(base: str, existing: set[str]) -> str:
    candidate = base
    if candidate not in existing:
        return candidate
    idx = 2
    while f"{candidate}-{idx}" in existing:
        idx += 1
    return f"{candidate}-{idx}"


@router.post("/projects", summary="Create a new project")
async def create_project(payload: ProjectCreateRequest):
    try:
        project_id = None
        if payload.use_slug_ids:
            slug = _slugify(payload.name)
            if slug:
                existing = {p.project_id for p in project_store.list_projects()}
                project_id = _unique_id(slug, existing)
        project = project_store.create_project(payload.name, payload.description, project_id=project_id)
    except Exception as exc:  # pragma: no cover - filesystem failure paths
        raise HTTPException(status_code=500, detail=f"Failed to create project: {exc}") from exc
    return serialize_project(project)


@router.get("/projects", summary="List all projects")
async def list_projects():
    projects = [serialize_project(p) for p in project_store.list_projects()]
    return projects


@router.get("/projects/dump", summary="Download a zip with all projects")
async def dump_projects():
    projects_dir = project_store.base_dir
    if not projects_dir.exists():
        raise HTTPException(status_code=404, detail="Projects directory not found.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tmp_dir = DATA_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_name = tmp_dir / f"projects_dump_{timestamp}"
    archive_path = await run_in_threadpool(
        shutil.make_archive,
        str(base_name),
        "zip",
        root_dir=str(projects_dir.parent),
        base_dir=projects_dir.name,
    )
    return FileResponse(archive_path, filename=Path(archive_path).name, media_type="application/zip")


@router.get("/projects/{project_id}", summary="Project detail including systems")
async def get_project_detail(project_id: str):
    try:
        project = project_store.get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    systems = [serialize_system(s) for s in project_store.list_systems(project_id)]
    payload = serialize_project(project)
    payload["systems"] = systems
    return payload


@router.get("/projects/{project_id}/systems", summary="List systems for a project")
async def list_systems(project_id: str):
    try:
        project_store.get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    systems = [serialize_system(s) for s in project_store.list_systems(project_id)]
    return systems


@router.delete("/projects/{project_id}", summary="Delete a project and all its systems")
async def delete_project(project_id: str):
    try:
        project_store.delete_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {exc}") from exc
    return {"status": "deleted", "project_id": project_id}
