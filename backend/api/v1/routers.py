"""
API Routers for V1
Defines endpoints for job submission, status polling, and result retrieval.
"""

import shutil
import json, os
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
import uuid
from pathlib import Path
from rq.job import Job
from redis import RedisError

# Import the master task function
from backend.tasks import run_analysis_job
from backend.api.v1.schemas import (
    ProjectCreateRequest,
    StaticJobRequest,
    DynamicJobRequest,
    QUBOJobRequest,
)
from backend.services.project_store import (
    ProjectStore,
    DescriptorState,
    ProjectMetadata,
    SystemMetadata,
)
from backend.services.preprocessing import DescriptorPreprocessor
from backend.services.descriptors import save_descriptor_npz

api_router = APIRouter()

# --- Directory Definitions ---
RESULTS_DIR = Path("/app/data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
project_store = ProjectStore()


def _serialize_project(meta: ProjectMetadata) -> Dict[str, Any]:
    return asdict(meta)


def _serialize_system(meta: SystemMetadata) -> Dict[str, Any]:
    return asdict(meta)


async def _stream_upload(upload: UploadFile, destination: Path) -> None:
    """Writes an UploadFile to disk in streaming fashion."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as buffer:
        while chunk := await upload.read(1024 * 1024):
            buffer.write(chunk)


def _normalize_stride(label: str, raw_value: int) -> int:
    try:
        stride = int(raw_value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid stride value '{raw_value}' for {label}.")
    if stride <= 0:
        raise HTTPException(status_code=400, detail=f"Stride for {label} must be >= 1.")
    return stride


def _stride_to_slice(stride: int) -> Optional[str]:
    return f"::{stride}" if stride > 1 else None


def _parse_residue_selections(raw_json: Optional[str]) -> Optional[Dict[str, str]]:
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for residue selections.")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Residue selections JSON must be an object.")
    return data


def _ensure_system_ready(project_id: str, system_id: str) -> SystemMetadata:
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{system_id}' not found in project '{project_id}'.",
        )
    if system.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"System '{system_id}' is not ready (status={system.status}).",
        )
    if not system.descriptor_keys:
        raise HTTPException(
            status_code=400,
            detail=f"System '{system_id}' has no descriptor keys. Re-run preprocessing.",
        )
    return system


# --- Project & System Management ---

@api_router.post("/projects", summary="Create a new project")
async def create_project(payload: ProjectCreateRequest):
    try:
        project = project_store.create_project(payload.name, payload.description)
    except Exception as exc:  # pragma: no cover - filesystem failure paths
        raise HTTPException(status_code=500, detail=f"Failed to create project: {exc}") from exc
    return _serialize_project(project)


@api_router.get("/projects", summary="List all projects")
async def list_projects():
    projects = [_serialize_project(p) for p in project_store.list_projects()]
    return projects


@api_router.get("/projects/{project_id}", summary="Project detail including systems")
async def get_project_detail(project_id: str):
    try:
        project = project_store.get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    systems = [_serialize_system(s) for s in project_store.list_systems(project_id)]
    payload = _serialize_project(project)
    payload["systems"] = systems
    return payload


@api_router.get("/projects/{project_id}/systems", summary="List systems for a project")
async def list_systems(project_id: str):
    try:
        project_store.get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    systems = [_serialize_system(s) for s in project_store.list_systems(project_id)]
    return systems


@api_router.get("/projects/{project_id}/systems/{system_id}", summary="Get system metadata")
async def get_system_detail(project_id: str, system_id: str):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{system_id}' not found in project '{project_id}'.",
        )
    return _serialize_system(system)


@api_router.get(
    "/projects/{project_id}/systems/{system_id}/structures/{state}",
    summary="Download the stored PDB file for a system state",
)
async def download_structure(project_id: str, system_id: str, state: str):
    state_key = state.lower()
    if state_key not in ("active", "inactive"):
        raise HTTPException(status_code=400, detail="State must be 'active' or 'inactive'.")
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = system.states.get(state_key)
    if not state_meta or not state_meta.pdb_file:
        raise HTTPException(status_code=404, detail=f"No PDB stored for state '{state_key}'.")

    file_path = project_store.resolve_path(project_id, system_id, state_meta.pdb_file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored PDB file is missing on disk.")

    return FileResponse(
        file_path,
        filename=os.path.basename(file_path),
        media_type="chemical/x-pdb",
    )


@api_router.delete("/projects/{project_id}", summary="Delete a project and all its systems")
async def delete_project(project_id: str):
    try:
        project_store.delete_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {exc}") from exc
    return {"status": "deleted", "project_id": project_id}


@api_router.delete("/projects/{project_id}/systems/{system_id}", summary="Delete a system")
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


@api_router.post(
    "/projects/{project_id}/systems",
    summary="Upload structures and trajectories to build descriptor files",
)
async def create_system_with_descriptors(
    project_id: str,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    active_pdb: UploadFile = File(...),
    inactive_pdb: Optional[UploadFile] = File(None),
    active_traj: UploadFile = File(...),
    inactive_traj: UploadFile = File(...),
    active_stride: int = Form(1),
    inactive_stride: int = Form(1),
    residue_selections_json: Optional[str] = Form(None),
):
    try:
        project_store.get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    residue_selections = _parse_residue_selections(residue_selections_json)
    try:
        system_meta = project_store.create_system(
            project_id, name=name, description=description, residue_selections=residue_selections
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to create system: {exc}") from exc

    dirs = project_store.ensure_directories(project_id, system_meta.system_id)
    system_dir = dirs["system_dir"]
    structures_dir = dirs["structures_dir"]
    descriptors_dir = dirs["descriptors_dir"]
    tmp_dir = dirs["tmp_dir"]

    active_stride_val = _normalize_stride("active", active_stride)
    inactive_stride_val = _normalize_stride("inactive", inactive_stride)
    active_slice = _stride_to_slice(active_stride_val)
    inactive_slice = _stride_to_slice(inactive_stride_val)

    active_pdb_path = structures_dir / "active.pdb"
    await _stream_upload(active_pdb, active_pdb_path)

    inactive_pdb_path = structures_dir / "inactive.pdb"
    if inactive_pdb:
        await _stream_upload(inactive_pdb, inactive_pdb_path)
    else:
        shutil.copy(active_pdb_path, inactive_pdb_path)

    active_traj_path = tmp_dir / f"active_{active_traj.filename or 'traj'}"
    inactive_traj_path = tmp_dir / f"inactive_{inactive_traj.filename or 'traj'}"
    await _stream_upload(active_traj, active_traj_path)
    await _stream_upload(inactive_traj, inactive_traj_path)

    preprocessor = DescriptorPreprocessor(residue_selections=residue_selections)
    try:
        build_result = preprocessor.build(
            str(active_traj_path),
            str(active_pdb_path),
            str(inactive_traj_path),
            str(inactive_pdb_path),
            active_slice=active_slice,
            inactive_slice=inactive_slice,
        )
    except Exception as exc:
        system_meta.status = "failed"
        project_store.save_system(system_meta)
        raise HTTPException(status_code=500, detail=f"Descriptor pre-processing failed: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    active_npz_path = descriptors_dir / "active_descriptors.npz"
    inactive_npz_path = descriptors_dir / "inactive_descriptors.npz"
    save_descriptor_npz(active_npz_path, build_result.active_features)
    save_descriptor_npz(inactive_npz_path, build_result.inactive_features)

    relative_active_pdb = str(active_pdb_path.relative_to(system_dir))
    relative_inactive_pdb = str(inactive_pdb_path.relative_to(system_dir))
    relative_active_npz = str(active_npz_path.relative_to(system_dir))
    relative_inactive_npz = str(inactive_npz_path.relative_to(system_dir))

    system_meta.states["active"] = DescriptorState(
        role="active",
        pdb_file=relative_active_pdb,
        descriptor_file=relative_active_npz,
        n_frames=build_result.n_frames_active,
        stride=active_stride_val,
        source_traj=active_traj.filename,
        slice_spec=active_slice,
    )
    system_meta.states["inactive"] = DescriptorState(
        role="inactive",
        pdb_file=relative_inactive_pdb,
        descriptor_file=relative_inactive_npz,
        n_frames=build_result.n_frames_inactive,
        stride=inactive_stride_val,
        source_traj=inactive_traj.filename,
        slice_spec=inactive_slice,
    )
    system_meta.descriptor_keys = build_result.residue_keys
    system_meta.residue_selections_mapping = build_result.residue_mapping
    system_meta.status = "ready"

    project_store.save_system(system_meta)
    return _serialize_system(system_meta)


# --- Dependencies ---
def get_queue(request: Request):
    """Provides the RQ queue object."""
    if request.app.state.task_queue is None:
        raise HTTPException(status_code=503, detail="Worker queue not initialized. Check Redis connection.")
    return request.app.state.task_queue

# --- Health Check Endpoint ---

@api_router.get("/health/check", summary="End-to-end system health check")
async def health_check(request: Request):
    """
    Performs an end-to-end health check of the API, Redis connection,
    and RQ worker availability.
    """
    report = {
        "api_status": "ok",
        "redis_status": {"status": "unknown"},
        "worker_status": {"status": "unknown"},
    }
    
    # 1. Check Redis connection
    redis_conn = request.app.state.redis_conn
    if redis_conn:
        try:
            redis_conn.ping()
            report["redis_status"] = {"status": "ok", "info": "Connected and ping successful."}
        except RedisError as e:
            report["redis_status"] = {"status": "error", "error": f"Redis connection failed: {str(e)}"}
    else:
         report["redis_status"] = {"status": "error", "error": "Redis client failed to initialize."}
        
    # 2. Check Worker Queue status
    if report["redis_status"]["status"] == "ok":
        task_queue = request.app.state.task_queue
        try:
            report["worker_status"] = {
                "status": "ok", 
                "queue_name": task_queue.name,
                "queue_length": task_queue.count
            }
        except Exception as e:
            report["worker_status"] = {"status": "error", "error": f"Error interacting with RQ queue: {str(e)}"}

    if report["redis_status"]["status"] != "ok" or report["worker_status"]["status"] != "ok":
        raise HTTPException(status_code=503, detail=report)
    return report

# --- Job Status Endpoint ---

@api_router.get("/job/status/{job_id}", summary="Get the live status of a running job")
async def get_job_status(job_id: str, request: Request):
    """
    Polls RQ for the live status of an enqueued job.
    The job_id here is the RQ job ID, not our UUID.
    """
    task_queue = get_queue(request)
    try:
        job = Job.fetch(job_id, connection=task_queue.connection)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found in RQ.")
        
    status = job.get_status()
    response = {
        "job_id": job_id,
        "status": status,
        "meta": job.meta,
    }
    
    if status == 'finished' or status == 'failed':
        # The return value of the task (`result_payload`)
        response["result"] = job.result
    
    return response

def submit_job(
    analysis_type: str,
    project_id: str,
    system_id: str,
    params: Dict[str, Any],
    task_queue: Any,
):
    """Helper to enqueue a job backed by a preprocessed system."""
    _ensure_system_ready(project_id, system_id)
    job_uuid = str(uuid.uuid4())
    dataset_ref = {"project_id": project_id, "system_id": system_id}

    try:
        job = task_queue.enqueue(
            run_analysis_job,
            args=(
                job_uuid,
                analysis_type,
                dataset_ref,
                params,
            ),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"analysis-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc

@api_router.post("/submit/static", summary="Submit a Static Reporters analysis")
async def submit_static_job(
    payload: StaticJobRequest,
    task_queue: get_queue = Depends(),
):
    params = {
        "state_metric": payload.state_metric,
    }
    return submit_job("static", payload.project_id, payload.system_id, params, task_queue)


@api_router.post("/submit/dynamic", summary="Submit a Dynamic (Transfer Entropy) analysis")
async def submit_dynamic_job(
    payload: DynamicJobRequest,
    task_queue: get_queue = Depends(),
):
    params = {"te_lag": payload.te_lag}
    return submit_job("dynamic", payload.project_id, payload.system_id, params, task_queue)


@api_router.post("/submit/qubo", summary="Submit a QUBO analysis")
async def submit_qubo_job(
    payload: QUBOJobRequest,
    task_queue: get_queue = Depends(),
):
    params = {
        "alpha_size": payload.alpha_size,
        "beta_hub": payload.beta_hub,
        "beta_switch": payload.beta_switch,
        "gamma_redundancy": payload.gamma_redundancy,
        "ii_threshold": payload.ii_threshold,
        "filter_top_total": payload.filter_top_total,
        "filter_top_jsd": payload.filter_top_jsd,
        "filter_min_id": payload.filter_min_id,
    }
    if payload.static_job_uuid:
        params["static_job_uuid"] = payload.static_job_uuid

    return submit_job("qubo", payload.project_id, payload.system_id, params, task_queue)

# --- Results Endpoints ---

@api_router.get("/results", summary="List all available analysis results")
async def get_results_list():
    """
    Fetches the metadata for all jobs (finished, running, or failed)
    by reading the JSON files from the persistent results directory.
    """
    results_list = []
    try:
        # Sort by mtime (newest first)
        sorted_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        
        for result_file in sorted_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                # Return just the metadata, not the full (large) result payload
                system_ref = data.get("system_reference") or {}
                results_list.append({
                    "job_id": data.get("job_id"),
                    "rq_job_id": data.get("rq_job_id"), # <-- Pass this to frontend
                    "analysis_type": data.get("analysis_type"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                    "error": data.get("error"),
                    "project_id": system_ref.get("project_id"),
                    "system_id": system_ref.get("system_id"),
                    "structures": system_ref.get("structures"),
                })
            except Exception as e:
                print(f"Failed to read result file: {result_file}. Error: {e}")
        
        return results_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {e}")

@api_router.get("/results/{job_uuid}", summary="Get the full JSON data for a specific result")
async def get_result_detail(job_uuid: str):
    """
    Fetches the complete, persisted JSON data for a single analysis job
    using its unique job_uuid.
    """
    try:
        result_file = RESULTS_DIR / f"{job_uuid}.json"
        if not result_file.exists() or not result_file.is_file():
            # It's possible the user is requesting a job that just started
            # and the file hasn't been written yet.
            # But 'started' jobs should lead to the status page.
            # If they have a direct link, 404 is correct.
            raise HTTPException(status_code=404, detail=f"Result file for job '{job_uuid}' not found.")
        
        # Use FileResponse to correctly stream the file for download
        return Response(
            content=result_file.read_text(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(result_file)}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to read result: {e}")

@api_router.delete("/results/{job_uuid}", summary="Delete a job and its associated data")
async def delete_result(job_uuid: str):
    """
    Deletes a job's persisted JSON file.
    """
    result_file = RESULTS_DIR / f"{job_uuid}.json"

    try:
        if not result_file.exists():
            raise HTTPException(status_code=404, detail=f"No data found for job UUID '{job_uuid}'.")
        result_file.unlink()
        return {"status": "deleted", "job_id": job_uuid}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to delete job data: {str(e)}")
