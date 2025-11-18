"""
API Routers for V1
Defines endpoints for job submission, status polling, and result retrieval.
"""

import shutil
import json, os
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form, Response
from typing import Dict, Any, List, Optional
import uuid
from pathlib import Path
from rq.job import Job
from redis import RedisError

# Import the master task function
from backend.tasks import run_analysis_job

api_router = APIRouter()

# --- Directory Definitions ---
UPLOAD_DIR = Path("/app/data/uploads")
RESULTS_DIR = Path("/app/data/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Dependencies ---
def get_queue(request: Request):
    """Provides the RQ queue object."""
    if request.app.state.task_queue is None:
        raise HTTPException(status_code=503, detail="Worker queue not initialized. Check Redis connection.")
    return request.app.state.task_queue

# --- REFACTORED File Saving Helper ---
async def save_uploaded_files(
    files_dict: Dict[str, UploadFile], 
    job_folder: Path
) -> Dict[str, str]:
    """
    Saves uploaded files from a dictionary to a job-specific folder
    and returns a path dict for the worker.
    """
    saved_paths = {}
    
    # Map frontend form keys to worker path keys
    key_to_path_key = {
        "active_traj": "active_traj_file",
        "active_topo": "active_topo_file",
        "inactive_traj": "inactive_traj_file",
        "inactive_topo": "inactive_topo_file",
        "config": "config_path",
    }

    for key, file_obj in files_dict.items():
        if key not in key_to_path_key:
            print(f"Warning: Unknown file key '{key}' skipped.")
            continue 
        
        # This check is important: if file_obj is None (from File(None)), skip it
        if file_obj is None:
            continue

        # Use the original filename for saving
        file_path = job_folder / file_obj.filename
        try:
            with open(file_path, "wb") as buffer:
                # Read in 1MB chunks
                while content := await file_obj.read(1024 * 1024): 
                    buffer.write(content)
            
            # Use the dict 'key' to build the path map for the worker
            path_key = key_to_path_key[key]
            saved_paths[path_key] = str(file_path)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file {file_obj.filename}: {e}")

    # --- BUG FIX ---
    # The old check `if len(saved_paths) != 5:` was incorrect.
    # We must validate that the 4 *required* files are present.
    # The 'config' file is optional.
    required_paths = ["active_traj_file", "active_topo_file", "inactive_traj_file", "inactive_topo_file"]
    missing_files = [key for key in required_paths if key not in saved_paths]
    if missing_files:
            raise HTTPException(status_code=400, detail=f"Missing required files: {', '.join(missing_files)}")
    # --- END BUG FIX ---
    
    return saved_paths

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

# --- REFACTORED Job Submission Endpoints ---

async def submit_job(
    analysis_type: str,
    files_dict: Dict[str, UploadFile], # Now contains 4 or 5 files
    params: Dict[str, Any],            # Now contains optional 'residue_selections_dict'
    task_queue: Any                    # Dependency
):
    """Helper function to enqueue any analysis job."""
    
    # 1. Create a unique ID for this analysis run
    job_uuid = str(uuid.uuid4())
    job_folder = UPLOAD_DIR / job_uuid
    job_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # 2. Save all uploaded files (4 or 5)
        saved_paths = await save_uploaded_files(files_dict, job_folder)
        
        # 3. Prepare arguments for the worker task
        # Pop worker-specific params. `params` will be passed to the worker.
        residue_selections_dict = params.pop("residue_selections_dict", None)
        # Get the config_path *if it was saved*
        config_path = saved_paths.get('config_path')
        
        # 4. Enqueue the Master Job
        # The `run_analysis_job` function in tasks.py expects 6 arguments.
        # We must pass them all, respecting their order.
        job = task_queue.enqueue(
            run_analysis_job,
            args=(
                job_uuid,                 # 1. job_uuid
                analysis_type,            # 2. analysis_type
                saved_paths,              # 3. file_paths
                params,                   # 4. params (now contains analysis-specific args)
                config_path,              # 5. config_path (can be None)
                residue_selections_dict   # 6. residue_selections_dict (can be None)
            ),
            job_timeout='2h',
            result_ttl=86400, # Keep result in Redis for 1 day
            job_id=f"analysis-{job_uuid}" # Use a predictable RQ job ID
        )

        # 5. Return the RQ job ID for polling
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}

    except Exception as e:
        if job_folder.exists():
            shutil.rmtree(job_folder)
        if isinstance(e, HTTPException):
            raise e
        print(f"Error during job submission: {e}") # Log for debugging
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@api_router.post("/submit/static", summary="Submit a Static Reporters analysis")
async def submit_static_job(
    active_traj: UploadFile = File(...),
    active_topo: UploadFile = File(...),
    inactive_traj: UploadFile = File(...),
    inactive_topo: UploadFile = File(...),
    config: Optional[UploadFile] = File(None),
    residue_selections_json: Optional[str] = Form(None),
    active_slice: Optional[str] = Form(None),
    inactive_slice: Optional[str] = Form(None),
    state_metric: str = Form("auc"), # Add new parameter
    task_queue: get_queue = Depends(),
):
    # Create the files dict,
    # `config` will be None if not provided by the frontend.
    files_dict = {
        "active_traj": active_traj, "active_topo": active_topo,
        "inactive_traj": inactive_traj, "inactive_topo": inactive_topo,
        "config": config
    }
    
    # Prepare params for the worker
    params = {
        "active_slice": active_slice,
        "inactive_slice": inactive_slice,
        "state_metric": state_metric, # Pass to worker
    }
    
    if residue_selections_json:
        try:
            params["residue_selections_dict"] = json.loads(residue_selections_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for residue_selections_json.")
            
    return await submit_job("static", files_dict, params, task_queue)

@api_router.post("/submit/dynamic", summary="Submit a Dynamic (Transfer Entropy) analysis")
async def submit_dynamic_job(
    active_traj: UploadFile = File(...),
    active_topo: UploadFile = File(...),
    inactive_traj: UploadFile = File(...),
    inactive_topo: UploadFile = File(...),
    config: Optional[UploadFile] = File(None),
    residue_selections_json: Optional[str] = Form(None),
    te_lag: int = Form(10),
    active_slice: Optional[str] = Form(None),
    inactive_slice: Optional[str] = Form(None),
    task_queue: get_queue = Depends(),
):
    files_dict = {
        "active_traj": active_traj, "active_topo": active_topo,
        "inactive_traj": inactive_traj, "inactive_topo": inactive_topo,
        "config": config
    }
    params = {
        "te_lag": te_lag,
        "active_slice": active_slice,
        "inactive_slice": inactive_slice,
    }

    if residue_selections_json:
        try:
            params["residue_selections_dict"] = json.loads(residue_selections_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for residue_selections_json.")

    return await submit_job("dynamic", files_dict, params, task_queue)

@api_router.post("/submit/qubo", summary="Submit a QUBO analysis")
async def submit_qubo_job(
    active_traj: UploadFile = File(...),
    active_topo: UploadFile = File(...),
    inactive_traj: UploadFile = File(...),
    inactive_topo: UploadFile = File(...),
    config: Optional[UploadFile] = File(None),
    residue_selections_json: Optional[str] = Form(None),
    active_slice: Optional[str] = Form(None),
    inactive_slice: Optional[str] = Form(None),
    static_job_uuid: Optional[str] = Form(None),
    
    # QUBO Hamiltonian params
    alpha_size: float = Form(1.0),
    beta_hub: float = Form(2.0),
    beta_switch: float = Form(5.0),
    gamma_redundancy: float = Form(3.0),
    ii_threshold: float = Form(0.4),
    
    # New Filter Params
    filter_top_total: int = Form(100, description="Total candidates to send to QUBO"),
    filter_top_jsd: int = Form(20, description="Number of top JSD residues guaranteed to be included"),
    filter_min_id: float = Form(1.5, description="Minimum Intrinsic Dimension to be considered a Mover"),

    task_queue: get_queue = Depends(),
):
    files_dict = {
        "active_traj": active_traj, "active_topo": active_topo,
        "inactive_traj": inactive_traj, "inactive_topo": inactive_topo,
        "config": config
    }

    if not static_job_uuid and not residue_selections_json and not config:
         raise HTTPException(status_code=400, detail="Missing candidate source.")

    params = {
        "active_slice": active_slice,
        "inactive_slice": inactive_slice,
        "alpha_size": alpha_size,
        "beta_hub": beta_hub,
        "beta_switch": beta_switch,
        "gamma_redundancy": gamma_redundancy,
        "ii_threshold": ii_threshold,
        # New params passed to runner
        "filter_top_total": filter_top_total,
        "filter_top_jsd": filter_top_jsd,
        "filter_min_id": filter_min_id, 
    }

    if static_job_uuid:
        params["static_job_uuid"] = static_job_uuid
    
    if residue_selections_json:
        try:
            params["residue_selections_dict"] = json.loads(residue_selections_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON.")

    return await submit_job("qubo", files_dict, params, task_queue)

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
                results_list.append({
                    "job_id": data.get("job_id"),
                    "rq_job_id": data.get("rq_job_id"), # <-- Pass this to frontend
                    "analysis_type": data.get("analysis_type"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                    "error": data.get("error"),
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
    Deletes a job's result JSON file and its uploaded data folder.
    """
    result_file = RESULTS_DIR / f"{job_uuid}.json"
    upload_folder = UPLOAD_DIR / job_uuid

    file_existed = False
    folder_existed = False

    try:
        if result_file.exists():
            result_file.unlink()
            file_existed = True

        if upload_folder.exists() and upload_folder.is_dir():
            shutil.rmtree(upload_folder)
            folder_existed = True

        if not file_existed and not folder_existed:
            raise HTTPException(status_code=404, detail=f"No data found for job UUID '{job_uuid}'.")

        return {"status": "deleted", "job_id": job_uuid}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to delete job data: {str(e)}")