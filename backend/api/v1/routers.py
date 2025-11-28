"""
API Routers for V1
Defines endpoints for job submission, status polling, and result retrieval.
"""

import shutil
import functools
import json, os
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any, List, Optional, Tuple
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
DATA_ROOT = Path(os.getenv("ALLOSKIN_DATA_ROOT", "/app/data"))
RESULTS_DIR = DATA_ROOT / "results"
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


def _parse_residue_selections(raw_value: Optional[str], expect_json: bool = False):
    """
    Accepts either legacy JSON objects/arrays or newline-delimited text inputs.
    Returns None, a dict, or a list of selection strings.
    """
    if not raw_value:
        return None
    stripped = raw_value.strip()
    if not stripped:
        return None

    should_parse_json = expect_json or stripped[0] in ("{", "[")

    if should_parse_json:
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if expect_json:
                raise HTTPException(status_code=400, detail="Invalid JSON for residue selections.") from exc
        else:
            if isinstance(data, dict):
                cleaned = {k: str(v).strip() for k, v in data.items() if isinstance(v, str) and v.strip()}
                return cleaned or None
            if isinstance(data, list):
                lines = [str(item).strip() for item in data if str(item).strip()]
                return lines or None
            raise HTTPException(status_code=400, detail="Residue selections JSON must be an object or array.")

    lines = [line.strip() for line in raw_value.splitlines() if line.strip()]
    return lines or None


def _get_state_or_404(system_meta: SystemMetadata, state_id: str) -> DescriptorState:
    state = system_meta.states.get(state_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"State '{state_id}' not found for system '{system_meta.system_id}'.")
    return state


def _update_system_status(system_meta: SystemMetadata) -> None:
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


def _refresh_system_metadata(system_meta: SystemMetadata) -> None:
    all_keys = set()
    for state in system_meta.states.values():
        all_keys.update(state.residue_keys or [])
    system_meta.descriptor_keys = sorted(all_keys)
    _update_system_status(system_meta)


def _build_state_artifacts(
    preprocessor: DescriptorPreprocessor,
    *,
    traj_path: Path,
    pdb_path: Path,
    descriptors_dir: Path,
    slice_spec: Optional[str],
    state_id: str,
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
    }
    artifact_paths["metadata"].write_text(json.dumps(metadata_payload, indent=2))
    return build_result, artifact_paths


async def _build_state_descriptors(
    project_id: str, system_meta: SystemMetadata, state_meta: DescriptorState
) -> SystemMetadata:
    if not state_meta.trajectory_file:
        raise HTTPException(status_code=400, detail="No trajectory uploaded for this state.")
    if not state_meta.pdb_file:
        raise HTTPException(status_code=400, detail="No PDB stored for this state.")

    dirs = project_store.ensure_directories(project_id, system_meta.system_id)
    system_dir = dirs["system_dir"]
    descriptors_dir = dirs["descriptors_dir"]

    traj_path = project_store.resolve_path(project_id, system_meta.system_id, state_meta.trajectory_file)
    pdb_path = project_store.resolve_path(project_id, system_meta.system_id, state_meta.pdb_file)

    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Stored trajectory file missing on disk.")
    if not pdb_path.exists():
        raise HTTPException(status_code=404, detail="Stored PDB file missing on disk.")

    preprocessor = DescriptorPreprocessor(residue_selections=system_meta.residue_selections)
    print(f"[state-update] Building descriptors for state={state_meta.state_id} system={system_meta.system_id}")
    build_result, artifact_paths = await run_in_threadpool(
        functools.partial(
            _build_state_artifacts,
            preprocessor,
            traj_path=traj_path,
            pdb_path=pdb_path,
            descriptors_dir=descriptors_dir,
            slice_spec=state_meta.slice_spec,
            state_id=state_meta.state_id,
        )
    )

    rel_npz = str(artifact_paths["npz"].relative_to(system_dir))
    rel_meta = str(artifact_paths["metadata"].relative_to(system_dir))

    state_meta.descriptor_file = rel_npz
    state_meta.descriptor_metadata_file = rel_meta
    state_meta.n_frames = build_result.n_frames
    state_meta.residue_keys = build_result.residue_keys
    state_meta.residue_mapping = build_result.residue_mapping

    _refresh_system_metadata(system_meta)

    project_store.save_system(system_meta)
    return system_meta


def _pick_state_pair(system_meta: SystemMetadata, state_a_id: Optional[str], state_b_id: Optional[str]):
    if state_a_id and state_b_id:
        if state_a_id == state_b_id:
            raise HTTPException(status_code=400, detail="Select two different states.")
        return _get_state_or_404(system_meta, state_a_id), _get_state_or_404(system_meta, state_b_id)

    descriptor_states = [s for s in system_meta.states.values() if s.descriptor_file]
    if len(descriptor_states) < 2:
        raise HTTPException(status_code=400, detail="At least two states with descriptors are required.")
    return descriptor_states[0], descriptor_states[1]


def _ensure_system_ready(project_id: str, system_id: str, state_a_id: Optional[str], state_b_id: Optional[str]):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{system_id}' not found in project '{project_id}'.",
        )
    state_a, state_b = _pick_state_pair(system, state_a_id, state_b_id)
    if not state_a.descriptor_file or not state_b.descriptor_file:
        raise HTTPException(status_code=400, detail="Selected states do not have built descriptors.")
    if not state_a.residue_keys or not state_b.residue_keys:
        raise HTTPException(status_code=400, detail="Selected states are missing descriptor metadata.")
    return system, state_a, state_b


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
    "/projects/{project_id}/systems/{system_id}/structures/{state_id}",
    summary="Download the stored PDB file for a system state",
)
async def download_structure(project_id: str, system_id: str, state_id: str):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = _get_state_or_404(system, state_id)
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
    residue_selections = _parse_residue_selections(
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
        await _stream_upload(upload, pdb_path)

        system_meta.states[state_id] = DescriptorState(
            state_id=state_id,
            name=state_name,
            pdb_file=str(pdb_path.relative_to(system_dir)),
            stride=1,
        )

    _refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)
    return _serialize_system(system_meta)


@api_router.post(
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
        await _stream_upload(pdb, pdb_path)
    else:
        source_state = _get_state_or_404(system_meta, source_state_id)
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
    _refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)
    return _serialize_system(system_meta)


@api_router.post(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory",
    summary="Upload/replace a trajectory for a state and rebuild descriptors",
)
async def upload_state_trajectory(
    project_id: str,
    system_id: str,
    state_id: str,
    trajectory: UploadFile = File(...),
    stride: int = Form(1),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = _get_state_or_404(system_meta, state_id)
    stride_val = _normalize_stride(state_meta.name, stride)
    slice_spec = _stride_to_slice(stride_val)

    dirs = project_store.ensure_directories(project_id, system_id)
    traj_dir = dirs["trajectories_dir"]
    system_dir = dirs["system_dir"]

    traj_path = traj_dir / f"{state_id}_{trajectory.filename or 'traj'}"
    await _stream_upload(trajectory, traj_path)

    state_meta.trajectory_file = str(traj_path.relative_to(system_dir))
    state_meta.source_traj = trajectory.filename
    state_meta.stride = stride_val
    state_meta.slice_spec = slice_spec
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

    # Ensure PDB exists for the state
    if not state_meta.pdb_file:
        raise HTTPException(status_code=400, detail="No stored PDB for this state. Upload PDB first.")

    project_store.save_system(system_meta)

    try:
        await _build_state_descriptors(project_id, system_meta, state_meta)
    except Exception as exc:
        system_meta.status = "failed"
        project_store.save_system(system_meta)
        raise HTTPException(status_code=500, detail=f"Descriptor build failed after upload: {exc}") from exc

    return _serialize_system(system_meta)


@api_router.delete(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/trajectory",
    summary="Delete the trajectory and descriptors for a state",
)
async def delete_state_trajectory(project_id: str, system_id: str, state_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = _get_state_or_404(system_meta, state_id)

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

    _refresh_system_metadata(system_meta)
    project_store.save_system(system_meta)

    return _serialize_system(system_meta)


@api_router.delete(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}",
    summary="Delete a state and its stored files",
)
async def delete_state(project_id: str, system_id: str, state_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = _get_state_or_404(system_meta, state_id)

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
    _refresh_system_metadata(system_meta)
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
    state_a_id: str,
    state_b_id: str,
    params: Dict[str, Any],
    task_queue: Any,
):
    """Helper to enqueue a job backed by a preprocessed system."""
    system_meta, state_a, state_b = _ensure_system_ready(project_id, system_id, state_a_id, state_b_id)
    job_uuid = str(uuid.uuid4())
    try:
        project_meta = project_store.get_project(project_id)
        project_name = project_meta.name
    except Exception:
        project_name = None

    dataset_ref = {
        "project_id": project_id,
        "project_name": project_name,
        "system_id": system_id,
        "system_name": system_meta.name,
        "state_a_id": state_a.state_id,
        "state_b_id": state_b.state_id,
        "state_a_name": state_a.name,
        "state_b_name": state_b.name,
    }

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
    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "state_a_id", "state_b_id"})
    return submit_job(
        "static",
        payload.project_id,
        payload.system_id,
        payload.state_a_id,
        payload.state_b_id,
        params,
        task_queue,
    )


@api_router.post("/submit/dynamic", summary="Submit a Dynamic (Transfer Entropy) analysis")
async def submit_dynamic_job(
    payload: DynamicJobRequest,
    task_queue: get_queue = Depends(),
):
    params = {"te_lag": payload.te_lag}
    return submit_job(
        "dynamic",
        payload.project_id,
        payload.system_id,
        payload.state_a_id,
        payload.state_b_id,
        params,
        task_queue,
    )


@api_router.post("/submit/qubo", summary="Submit a QUBO analysis")
async def submit_qubo_job(
    payload: QUBOJobRequest,
    task_queue: get_queue = Depends(),
):
    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "state_a_id", "state_b_id"})
    # Clean up optional static reference to avoid passing empty strings
    static_uuid = params.get("static_job_uuid")
    if not static_uuid:
        params.pop("static_job_uuid", None)
    else:
        params["static_job_uuid"] = static_uuid
    return submit_job(
        "qubo",
        payload.project_id,
        payload.system_id,
        payload.state_a_id,
        payload.state_b_id,
        params,
        task_queue,
    )

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
                state_ref = system_ref.get("states") or {}
                results_list.append({
                    "job_id": data.get("job_id"),
                    "rq_job_id": data.get("rq_job_id"), # <-- Pass this to frontend
                    "analysis_type": data.get("analysis_type"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                    "error": data.get("error"),
                    "project_id": system_ref.get("project_id"),
                    "project_name": system_ref.get("project_name"),
                    "system_id": system_ref.get("system_id"),
                    "system_name": system_ref.get("system_name"),
                    "state_a_id": state_ref.get("state_a", {}).get("id"),
                    "state_a_name": state_ref.get("state_a", {}).get("name"),
                    "state_b_id": state_ref.get("state_b", {}).get("id"),
                    "state_b_name": state_ref.get("state_b", {}).get("name"),
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
