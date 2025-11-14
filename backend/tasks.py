import os
import shutil
import yaml
import json
import traceback
from pathlib import Path
from datetime import datetime
from rq import get_current_job
from typing import Dict, Any, Optional

# Import core analysis components
from alloskin.io.readers import MDAnalysisReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.pipeline.builder import DatasetBuilder
from alloskin.analysis.static import StaticReportersRF
from alloskin.analysis.qubo import QUBOSet
from alloskin.analysis.dynamic import TransferEntropy

# Define the persistent results directory
RESULTS_DIR = Path("/app/data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Function (Unchanged) ---
def get_builder(
    config_path: Optional[str] = None, 
    residue_selections_dict: Optional[Dict[str, str]] = None
) -> DatasetBuilder:
    """
    Initializes the core components.
    The builder will now be responsible for generating the mapping.
    """
    residue_selections = None

    if residue_selections_dict is not None:
        residue_selections = residue_selections_dict
        print(f"[Worker] Using provided residue selections: {residue_selections}")
    elif config_path:
        print(f"[Worker] Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        residue_selections = config.get('residue_selections')
        if residue_selections is None:
            print("[Worker] Warning: 'residue_selections' not found in config file. Will analyze all protein residues.")
    else:
        print("[Worker] No config file or residue selections provided. Will analyze all protein residues.")
        
    reader = MDAnalysisReader()
    # This extractor is the 'prototype' holding the user's config
    extractor = FeatureExtractor(residue_selections)
    builder = DatasetBuilder(reader, extractor)
    # Only return the builder
    return builder

# --- Master Analysis Job (MODIFIED) ---

def run_analysis_job(
    job_uuid: str,
    analysis_type: str, 
    file_paths: Dict[str, str],
    params: Dict[str, Any],
    config_path: Optional[str] = None,
    residue_selections_dict: Optional[Dict[str, str]] = None
):
    """
    The main, long-running analysis function.
    This function is executed by the RQ Worker and handles all analysis types.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    
    # --- MODIFICATION: Get RQ Job ID ---
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
        # --- MODIFICATION: Write "started" file immediately ---
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)
        # --- END MODIFICATION ---

        # Step 1: Initialize components
        save_progress("Initializing Analysis Pipeline", 10)
        builder = get_builder(
            config_path=config_path, 
            residue_selections_dict=residue_selections_dict
        )
        
        # Step 2: Prepare data based on analysis type
        analysis_data = None
        mapping = None
        
        active_slice = params.get("active_slice")
        inactive_slice = params.get("inactive_slice")

        if analysis_type in ['static', 'qubo']:
            save_progress("Preparing static dataset", 30)
            all_features_static, labels_Y, mapping = builder.prepare_static_analysis_data(
                file_paths['active_traj_path'], file_paths['active_topo_path'],
                file_paths['inactive_traj_path'], file_paths['inactive_topo_path'],
                active_slice=active_slice,
                inactive_slice=inactive_slice
            )
            analysis_data = (all_features_static, labels_Y)
            
        elif analysis_type == 'dynamic':
            save_progress("Preparing dynamic dataset", 30)
            features_active, features_inactive, mapping = builder.prepare_dynamic_analysis_data(
                file_paths['active_traj_path'], file_paths['active_topo_path'],
                file_paths['inactive_traj_path'], file_paths['inactive_topo_path'],
                active_slice=active_slice,
                inactive_slice=inactive_slice
            )
            analysis_data = (features_active, features_inactive)
            
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Save the mapping as soon as we have it
        result_payload["residue_selections_mapping"] = mapping

        # Step 3: Run the correct analysis
        save_progress(f"Running {analysis_type} analysis", 60)
        
        if analysis_type == 'static':
            analyzer = StaticReportersRF()
            job_results = analyzer.run(analysis_data)
            
        elif analysis_type == 'qubo':
            analyzer = QUBOSet()
            target_switch = params.get('target_switch')
            if not target_switch:
                raise ValueError("QUBO analysis requires 'target_switch' parameter.")
            job_results = analyzer.run(analysis_data, target_switch=target_switch)
            
        elif analysis_type == 'dynamic':
            analyzer = TransferEntropy()
            te_lag = params.get('te_lag', 10)
            job_results = analyzer.run(analysis_data, lag=te_lag)

        # Step 4: Finalize
        save_progress("Analysis complete", 90)
        result_payload["status"] = "finished"
        result_payload["results"] = job_results
        
    except Exception as e:
        print(f"[Job {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
    
    finally:
        # Step 5: Save final persistent JSON result
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        
        # --- MODIFICATION: Overwrite file with final status ---
        write_result_to_disk(result_payload)
        # --- END MODIFICATION ---

        # Clean up the temporary upload folder
        try:
            upload_dir = Path("/app/data/uploads") / job_uuid
            if upload_dir.exists() and upload_dir.name == job_uuid:
                shutil.rmtree(upload_dir)
                print(f"Cleaned up upload directory: {upload_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up upload dir: {e}")

    # This is the value returned to RQ and shown on the status page
    return result_payload