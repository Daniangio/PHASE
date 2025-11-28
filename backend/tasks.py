import json
import os
import traceback
from pathlib import Path
from datetime import datetime
from rq import get_current_job
from typing import Dict, Any
from alloskin.pipeline.runner import run_analysis
from backend.services.project_store import ProjectStore

# Define the persistent results directory (aligned with ALLOSKIN_DATA_ROOT).
DATA_ROOT = Path(os.getenv("ALLOSKIN_DATA_ROOT", "/app/data"))
RESULTS_DIR = DATA_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
project_store = ProjectStore()

# Helper to convert NaN to None for JSON serialization
def _convert_nan_to_none(obj):
    """
    Recursively converts numpy.nan values in dicts and lists to None.
    """
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_nan_to_none(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return _convert_nan_to_none(obj.tolist())
    elif isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        return None if isinstance(val, float) and not np.isfinite(val) else val
    # --- FIX: Handle both nan and inf ---
    elif isinstance(obj, float) and not np.isfinite(obj):
        return None # Convert nan, inf, -inf to None (JSON null)
    return obj


# --- Master Analysis Job ---

def run_analysis_job(
    job_uuid: str,
    analysis_type: str, 
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    The main, long-running analysis function.
    This function is executed by the RQ Worker and handles all analysis types.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    
    # This ID is needed by the frontend to link from the results page
    # back to the live status page.
    rq_job_id = job.id if job else f"analysis-{job_uuid}" # Reconstruct as fallback
    
    result_payload = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id, # <-- NEW: Store the RQ ID
        "analysis_type": analysis_type,
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "residue_selections_mapping": None,
        "results": None,
        "system_reference": {
            "project_id": dataset_ref.get("project_id"),
            "system_id": dataset_ref.get("system_id"),
            "project_name": dataset_ref.get("project_name"),
            "system_name": dataset_ref.get("system_name"),
            "structures": {},
            "states": {},
        },
        "error": None,
        "completed_at": None, # Will be filled in 'finally'
    }
    
    result_filepath = RESULTS_DIR / f"{job_uuid}.json"

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta['status'] = status_msg
            job.meta['progress'] = progress
            job.save_meta()
        print(f"[Job {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        """Helper to write the result payload to the persistent JSON file."""
        try:
            with open(result_filepath, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"Saved result to {result_filepath}")
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            # If saving fails, update the in-memory payload for the final RQ return
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        # Step 1: Resolve the dataset (stored descriptors + PDBs)
        project_id = dataset_ref.get("project_id")
        system_id = dataset_ref.get("system_id")
        state_a_id = dataset_ref.get("state_a_id")
        state_b_id = dataset_ref.get("state_b_id")
        if not project_id or not system_id or not state_a_id or not state_b_id:
            raise ValueError("Dataset reference missing project_id/system_id/state pair.")

        system_meta = project_store.get_system(project_id, system_id)
        state_a = system_meta.states.get(state_a_id)
        state_b = system_meta.states.get(state_b_id)
        if not state_a or not state_b:
            raise ValueError("Selected states not found on the system.")
        if not state_a.descriptor_file or not state_b.descriptor_file:
            raise ValueError("Descriptor files are missing for the selected states.")

        # Determine common descriptor keys across the selected states
        common_keys = sorted(set(state_a.residue_keys or []) & set(state_b.residue_keys or []))
        if not common_keys:
            raise ValueError("Selected states do not share descriptor keys. Rebuild descriptors.")

        mapping_a = state_a.residue_mapping or {}
        mapping_b = state_b.residue_mapping or {}

        dataset_paths = {
            "project_id": project_id,
            "system_id": system_id,
            "active_descriptors": str(
                project_store.resolve_path(project_id, system_id, state_a.descriptor_file)
            ),
            "inactive_descriptors": str(
                project_store.resolve_path(project_id, system_id, state_b.descriptor_file)
            ),
            "descriptor_keys": common_keys,
            "residue_mapping": {k: mapping_a.get(k) or mapping_b.get(k) for k in common_keys},
            "n_frames_active": state_a.n_frames,
            "n_frames_inactive": state_b.n_frames,
        }

        result_payload["system_reference"] = {
            "project_id": project_id,
            "system_id": system_id,
            "project_name": dataset_ref.get("project_name"),
            "system_name": dataset_ref.get("system_name"),
            "structures": {
                state_a.state_id: state_a.pdb_file,
                state_b.state_id: state_b.pdb_file,
            },
            "states": {
                "state_a": {"id": state_a.state_id, "name": state_a.name},
                "state_b": {"id": state_b.state_id, "name": state_b.name},
            },
        }

        # --- NEW: Handle pre-computed static analysis for QUBO ---
        if analysis_type == 'qubo' and params.get('static_job_uuid'):
            static_uuid = params['static_job_uuid']
            static_results_path = RESULTS_DIR / f"{static_uuid}.json"
            if not static_results_path.exists():
                raise FileNotFoundError(f"Could not find specified static result file for job UUID: {static_uuid}")
            params['static_results_path'] = str(static_results_path)

        # Step 2: Delegate to the core runner
        job_results, mapping = run_analysis(
            analysis_type=analysis_type,
            file_paths=dataset_paths,
            params=params,
            progress_callback=save_progress
        )
        # --- NEW: Persist the final mapping and params ---
        result_payload["residue_selections_mapping"] = mapping or dataset_paths.get("residue_mapping")
        # The `params` dict might have been modified by the runner,
        # so we update it in the payload before saving.
        result_payload["params"] = params

        # Step 4: Finalize
        result_payload["status"] = "finished"
        result_payload["results"] = job_results

    except Exception as e:
        print(f"[Job {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        # --- Re-raise the exception AFTER saving the result. ---
        # This ensures that the RQ job itself is marked as 'failed',
        # which is what the frontend status page is polling for.
        raise e
    
    finally:
        # Step 5: Save final persistent JSON result
        save_progress("Saving final result", 95)
        # --- FIX: Sanitize the payload *before* writing and returning ---
        result_payload["completed_at"] = datetime.utcnow().isoformat() 
        sanitized_payload = _convert_nan_to_none(result_payload)
        
        write_result_to_disk(sanitized_payload)

    # This is the value returned to RQ and shown on the status page
    # --- FIX: Return the sanitized payload ---
    save_progress("Analysis completed", 100)
    return sanitized_payload
