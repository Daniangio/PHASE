from fastapi import APIRouter, HTTPException

from backend.api.v1.schemas import ProjectCreateRequest
from backend.api.v1.common import project_store, serialize_project, serialize_system


router = APIRouter()


@router.post("/projects", summary="Create a new project")
async def create_project(payload: ProjectCreateRequest):
    try:
        project = project_store.create_project(payload.name, payload.description)
    except Exception as exc:  # pragma: no cover - filesystem failure paths
        raise HTTPException(status_code=500, detail=f"Failed to create project: {exc}") from exc
    return serialize_project(project)


@router.get("/projects", summary="List all projects")
async def list_projects():
    projects = [serialize_project(p) for p in project_store.list_projects()]
    return projects


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
