import json
import os
import traceback
from pathlib import Path
from datetime import datetime
from rq import get_current_job
from typing import Dict, Any
from alloskin.pipeline.runner import run_analysis
from alloskin.simulation.main import parse_args as parse_simulation_args
from alloskin.simulation.main import run_pipeline as run_simulation_pipeline
from backend.services.project_store import ProjectStore

# Define the persistent results directory (aligned with ALLOSKIN_DATA_ROOT).
DATA_ROOT = Path(os.getenv("ALLOSKIN_DATA_ROOT", "/app/data"))
RESULTS_DIR = DATA_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SIMULATION_RESULTS_DIR = RESULTS_DIR / "simulation"
SIMULATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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


def _relativize_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_ROOT))
    except Exception:
        return str(path)


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


def run_simulation_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Run the Potts simulation pipeline using a saved cluster NPZ as input.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"analysis-{job_uuid}"

    result_payload = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "simulation",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": dataset_ref.get("project_id"),
            "system_id": dataset_ref.get("system_id"),
            "project_name": dataset_ref.get("project_name"),
            "system_name": dataset_ref.get("system_name"),
            "structures": {},
            "states": {},
            "cluster_id": dataset_ref.get("cluster_id"),
        },
        "error": None,
        "completed_at": None,
    }

    result_filepath = RESULTS_DIR / f"{job_uuid}.json"

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[Simulation {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved result to {result_filepath}")
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        project_id = dataset_ref.get("project_id")
        system_id = dataset_ref.get("system_id")
        cluster_id = dataset_ref.get("cluster_id")
        if not project_id or not system_id or not cluster_id:
            raise ValueError("Simulation dataset reference missing project_id/system_id/cluster_id.")

        system_meta = project_store.get_system(project_id, system_id)
        clusters = system_meta.metastable_clusters or []
        entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
        if not entry:
            raise FileNotFoundError(f"Cluster NPZ not found for cluster_id='{cluster_id}'.")
        rel_path = entry.get("path")
        if not rel_path:
            raise FileNotFoundError("Cluster NPZ path missing in system metadata.")
        cluster_path = Path(rel_path)
        if not cluster_path.is_absolute():
            cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster NPZ file is missing on disk: {cluster_path}")

        results_dir = SIMULATION_RESULTS_DIR / job_uuid
        results_dir.mkdir(parents=True, exist_ok=True)

        sim_params = dict(params or {})
        rex_betas = sim_params.get("rex_betas")
        if isinstance(rex_betas, (list, tuple)):
            rex_betas = ",".join(str(float(b)) for b in rex_betas)
        if isinstance(rex_betas, str) and not rex_betas.strip():
            rex_betas = None

        args_list = [
            "--npz",
            str(cluster_path),
            "--results-dir",
            str(results_dir),
            "--gibbs-method",
            "rex",
            "--estimate-beta-eff",
        ]

        if isinstance(rex_betas, str) and rex_betas.strip():
            args_list += ["--rex-betas", rex_betas.strip()]
        else:
            rex_beta_min = sim_params.get("rex_beta_min")
            rex_beta_max = sim_params.get("rex_beta_max")
            rex_spacing = sim_params.get("rex_spacing")
            if rex_beta_min is not None:
                args_list += ["--rex-beta-min", str(float(rex_beta_min))]
            if rex_beta_max is not None:
                args_list += ["--rex-beta-max", str(float(rex_beta_max))]
            if rex_spacing is not None:
                args_list += ["--rex-spacing", str(rex_spacing)]

        rex_samples = sim_params.get("rex_samples")
        if rex_samples is not None:
            args_list += ["--rex-rounds", str(int(rex_samples))]
        rex_burnin = sim_params.get("rex_burnin")
        if rex_burnin is not None:
            args_list += ["--rex-burnin-rounds", str(int(rex_burnin))]
        rex_thin = sim_params.get("rex_thin")
        if rex_thin is not None:
            args_list += ["--rex-thin-rounds", str(int(rex_thin))]
        rex_max_workers = sim_params.get("rex_max_workers")
        if rex_max_workers is None:
            rex_max_workers = 1
        if rex_max_workers is not None:
            args_list += ["--rex-max-workers", str(int(rex_max_workers))]

        sa_reads = sim_params.get("sa_reads")
        if sa_reads is not None:
            args_list += ["--sa-reads", str(int(sa_reads))]
        sa_sweeps = sim_params.get("sa_sweeps")
        if sa_sweeps is not None:
            args_list += ["--sa-sweeps", str(int(sa_sweeps))]
        sa_beta_schedules = []
        raw_schedules = sim_params.get("sa_beta_schedules") or []
        for schedule in raw_schedules:
            if schedule is None:
                continue
            if isinstance(schedule, dict):
                hot = schedule.get("beta_hot")
                cold = schedule.get("beta_cold")
            else:
                try:
                    hot, cold = schedule
                except Exception:
                    continue
            if hot is None or cold is None:
                continue
            sa_beta_schedules.append((float(hot), float(cold)))

        sa_beta_hot = sim_params.get("sa_beta_hot")
        sa_beta_cold = sim_params.get("sa_beta_cold")
        if sa_beta_hot is not None and sa_beta_cold is not None:
            sa_beta_schedules.append((float(sa_beta_hot), float(sa_beta_cold)))

        for hot, cold in sa_beta_schedules:
            args_list += ["--sa-beta-schedule", f"{float(hot)},{float(cold)}"]

        plm_epochs = sim_params.get("plm_epochs")
        if plm_epochs is not None:
            args_list += ["--plm-epochs", str(int(plm_epochs))]
        plm_lr = sim_params.get("plm_lr")
        if plm_lr is not None:
            args_list += ["--plm-lr", str(float(plm_lr))]
        plm_lr_min = sim_params.get("plm_lr_min")
        if plm_lr_min is not None:
            args_list += ["--plm-lr-min", str(float(plm_lr_min))]
        plm_lr_schedule = sim_params.get("plm_lr_schedule")
        if plm_lr_schedule is not None:
            args_list += ["--plm-lr-schedule", str(plm_lr_schedule)]
        plm_l2 = sim_params.get("plm_l2")
        if plm_l2 is not None:
            args_list += ["--plm-l2", str(float(plm_l2))]
        plm_batch_size = sim_params.get("plm_batch_size")
        if plm_batch_size is not None:
            args_list += ["--plm-batch-size", str(int(plm_batch_size))]
        plm_progress_every = sim_params.get("plm_progress_every")
        if plm_progress_every is not None:
            args_list += ["--plm-progress-every", str(int(plm_progress_every))]

        save_progress("Running Potts simulation", 20)
        try:
            sim_args = parse_simulation_args(args_list)
        except SystemExit as exc:
            raise ValueError("Invalid simulation arguments.") from exc

        run_result = run_simulation_pipeline(sim_args, progress_callback=save_progress)

        def _coerce_path(value: object) -> Path | None:
            if value is None:
                return None
            if isinstance(value, Path):
                return value
            return Path(str(value))

        summary_path = _coerce_path(run_result.get("summary_path"))
        plot_path = _coerce_path(run_result.get("plot_path"))
        meta_path = _coerce_path(run_result.get("metadata_path"))
        beta_scan_path = _coerce_path(run_result.get("beta_scan_path"))

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "results_dir": _relativize_path(results_dir),
            "summary_npz": _relativize_path(summary_path) if summary_path else None,
            "metadata_json": _relativize_path(meta_path) if meta_path else None,
            "marginals_plot": _relativize_path(plot_path) if plot_path else None,
            "beta_scan_plot": _relativize_path(beta_scan_path) if beta_scan_path else None,
            "cluster_npz": _relativize_path(cluster_path),
            "beta_eff": run_result.get("beta_eff"),
        }

    except Exception as e:
        print(f"[Simulation {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Simulation completed", 100)
    return sanitized_payload
