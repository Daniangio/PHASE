import json
import os
import time
import traceback
import shutil
import re
from pathlib import Path
from datetime import datetime
from rq import Queue, Worker, get_current_job
from rq.job import Job
from typing import Dict, Any, List
from phase.pipeline.runner import run_analysis
from phase.simulation.main import parse_args as parse_simulation_args
from phase.simulation.main import run_pipeline as run_simulation_pipeline
from backend.services.metastable_clusters import (
    generate_metastable_cluster_npz,
    prepare_cluster_workspace,
    reduce_cluster_workspace,
    run_cluster_chunk,
    assign_cluster_labels_to_states,
    update_cluster_metadata_with_assignments,
)
from backend.services.project_store import ProjectStore

# Define the persistent results directory (aligned with PHASE_DATA_ROOT).
DATA_ROOT = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
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


def _sanitize_model_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    safe = safe.strip("._-")
    return safe or "potts_model"


def _update_cluster_entry(
    project_id: str,
    system_id: str,
    cluster_id: str,
    updates: Dict[str, Any],
) -> None:
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except Exception:
        return
    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        return
    entry.update(updates)
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)


def _persist_potts_model(
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_path: Path,
    params: Dict[str, Any],
    *,
    source: str,
) -> str:
    entry = None
    try:
        system_meta = project_store.get_system(project_id, system_id)
        clusters = system_meta.metastable_clusters or []
        entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    except Exception:
        entry = None

    dirs = project_store.ensure_directories(project_id, system_id)
    system_dir = dirs["system_dir"]
    model_dir = dirs["potts_models_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    existing_name = entry.get("potts_model_name") if isinstance(entry, dict) else None
    existing_path = entry.get("potts_model_path") if isinstance(entry, dict) else None
    cluster_name = entry.get("name") if isinstance(entry, dict) else None
    display_name = (
        existing_name
        or (f"{cluster_name} Potts Model" if isinstance(cluster_name, str) and cluster_name.strip() else None)
        or f"{cluster_id} Potts Model"
    )

    dest_path = None
    if isinstance(existing_path, str) and existing_path:
        try:
            dest_path = project_store.resolve_path(project_id, system_id, existing_path)
        except Exception:
            dest_path = None
    if dest_path is None:
        base_name = _sanitize_model_filename(display_name)
        filename = f"{base_name}.npz"
        dest_path = model_dir / filename
        if dest_path.exists():
            suffix = cluster_id[:8]
            dest_path = model_dir / f"{base_name}-{suffix}.npz"
            counter = 2
            while dest_path.exists():
                dest_path = model_dir / f"{base_name}-{suffix}-{counter}.npz"
                counter += 1
    if model_path.resolve() != dest_path.resolve():
        shutil.copy2(model_path, dest_path)
    try:
        rel_path = str(dest_path.relative_to(system_dir))
    except Exception:
        rel_path = str(dest_path)
    _update_cluster_entry(
        project_id,
        system_id,
        cluster_id,
        {
            "potts_model_path": rel_path,
            "potts_model_name": display_name,
            "potts_model_updated_at": datetime.utcnow().isoformat(),
            "potts_model_source": source,
            "potts_model_params": params,
        },
    )
    return rel_path


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

    def force_single_thread():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        try:
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

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
        force_single_thread()
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
        model_rel = None
        if sim_params.get("use_potts_model", True):
            model_rel = sim_params.get("potts_model_path") or entry.get("potts_model_path")
        rex_betas_raw = sim_params.get("rex_betas")
        rex_betas_list = []
        if isinstance(rex_betas_raw, (list, tuple)):
            for b in rex_betas_raw:
                if b is None:
                    continue
                try:
                    rex_betas_list.append(float(b))
                except (TypeError, ValueError):
                    continue
        elif isinstance(rex_betas_raw, str):
            for part in rex_betas_raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    rex_betas_list.append(float(part))
                except (TypeError, ValueError):
                    continue

        rex_betas = ",".join(str(float(b)) for b in rex_betas_list) if rex_betas_list else None

        gibbs_method = "rex"
        beta_override = None
        if len(rex_betas_list) == 1:
            gibbs_method = "single"
            beta_override = rex_betas_list[0]
            rex_betas = None
        elif not rex_betas_list:
            rex_beta_min = sim_params.get("rex_beta_min")
            rex_beta_max = sim_params.get("rex_beta_max")
            rex_n_replicas = sim_params.get("rex_n_replicas")
            try:
                if rex_beta_min is not None and rex_beta_max is not None:
                    min_b = float(rex_beta_min)
                    max_b = float(rex_beta_max)
                    if abs(min_b - max_b) < 1e-12:
                        gibbs_method = "single"
                        beta_override = min_b
            except (TypeError, ValueError):
                pass
            try:
                if rex_n_replicas is not None and int(rex_n_replicas) <= 1:
                    gibbs_method = "single"
                    if beta_override is None:
                        if rex_beta_max is not None:
                            beta_override = float(rex_beta_max)
                        elif rex_beta_min is not None:
                            beta_override = float(rex_beta_min)
            except (TypeError, ValueError):
                pass

        args_list = [
            "--npz",
            str(cluster_path),
            "--results-dir",
            str(results_dir),
            "--gibbs-method",
            gibbs_method,
            "--estimate-beta-eff",
        ]
        if beta_override is not None:
            args_list += ["--beta", str(float(beta_override))]

        model_path = None
        if model_rel:
            model_path = Path(model_rel)
            if not model_path.is_absolute():
                model_path = project_store.resolve_path(project_id, system_id, model_rel)
            if model_path.exists():
                args_list += ["--model-npz", str(model_path)]
            else:
                model_path = None

        if gibbs_method == "rex":
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
        report_path = _coerce_path(run_result.get("report_path"))
        meta_path = _coerce_path(run_result.get("metadata_path"))
        beta_scan_path = _coerce_path(run_result.get("beta_scan_path"))
        model_artifact = _coerce_path(run_result.get("model_path"))

        potts_model_rel = None
        if model_artifact and model_artifact.exists():
            if model_path and model_path.exists() and model_artifact.resolve() == model_path.resolve():
                potts_model_rel = str(model_rel)
            else:
                potts_model_rel = _persist_potts_model(
                    project_id,
                    system_id,
                    cluster_id,
                    model_artifact,
                    params,
                    source="simulation",
                )

        cluster_name = entry.get("name") if isinstance(entry, dict) else None
        if cluster_name:
            result_payload["system_reference"]["cluster_name"] = cluster_name

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "results_dir": _relativize_path(results_dir),
            "summary_npz": _relativize_path(summary_path) if summary_path else None,
            "metadata_json": _relativize_path(meta_path) if meta_path else None,
            "marginals_plot": _relativize_path(plot_path) if plot_path else None,
            "sampling_report": _relativize_path(report_path) if report_path else None,
            "beta_scan_plot": _relativize_path(beta_scan_path) if beta_scan_path else None,
            "cluster_npz": _relativize_path(cluster_path),
            "potts_model": potts_model_rel or (_relativize_path(model_path) if model_path else None),
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


def run_potts_fit_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Fit a Potts model for a cluster NPZ and store the model for reuse.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"potts-fit-{job_uuid}"

    result_payload = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "potts_fit",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": dataset_ref.get("project_id"),
            "system_id": dataset_ref.get("system_id"),
            "project_name": dataset_ref.get("project_name"),
            "system_name": dataset_ref.get("system_name"),
            "cluster_id": dataset_ref.get("cluster_id"),
        },
        "error": None,
        "completed_at": None,
    }

    result_filepath = RESULTS_DIR / f"{job_uuid}.json"

    def force_single_thread():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        try:
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[PottsFit {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        force_single_thread()
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        project_id = dataset_ref.get("project_id")
        system_id = dataset_ref.get("system_id")
        cluster_id = dataset_ref.get("cluster_id")
        if not project_id or not system_id or not cluster_id:
            raise ValueError("Potts fit requires project_id/system_id/cluster_id.")

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

        dirs = project_store.ensure_directories(project_id, system_id)
        model_path = dirs["potts_models_dir"] / f"{cluster_id}_potts_model.npz"

        fit_params = dict(params or {})
        args_list = [
            "--npz",
            str(cluster_path),
            "--results-dir",
            str(SIMULATION_RESULTS_DIR / job_uuid),
            "--fit-only",
            "--model-out",
            str(model_path),
        ]

        fit_method = fit_params.get("fit_method")
        if fit_method is not None:
            args_list += ["--fit", str(fit_method)]

        for key, flag in (
            ("plm_epochs", "--plm-epochs"),
            ("plm_lr", "--plm-lr"),
            ("plm_lr_min", "--plm-lr-min"),
            ("plm_lr_schedule", "--plm-lr-schedule"),
            ("plm_l2", "--plm-l2"),
            ("plm_batch_size", "--plm-batch-size"),
            ("plm_progress_every", "--plm-progress-every"),
            ("plm_device", "--plm-device"),
        ):
            val = fit_params.get(key)
            if val is not None:
                args_list += [flag, str(val)]

        save_progress("Fitting Potts model", 20)
        try:
            sim_args = parse_simulation_args(args_list)
        except SystemExit as exc:
            raise ValueError("Invalid potts fit arguments.") from exc
        run_result = run_simulation_pipeline(sim_args, progress_callback=save_progress)

        potts_model_rel = _persist_potts_model(
            project_id,
            system_id,
            cluster_id,
            Path(model_path),
            fit_params,
            source="potts_fit",
        )

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "results_dir": _relativize_path(SIMULATION_RESULTS_DIR / job_uuid),
            "potts_model": potts_model_rel,
            "cluster_npz": _relativize_path(cluster_path),
            "metadata_json": _relativize_path(run_result.get("metadata_path")) if run_result.get("metadata_path") else None,
        }

    except Exception as e:
        print(f"[PottsFit {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Potts fit completed", 100)
    return sanitized_payload


def run_cluster_chunk_job(work_dir: str, residue_index: int) -> Dict[str, Any]:
    """Cluster a single residue from a prepared workspace."""
    return run_cluster_chunk(Path(work_dir), residue_index)


def run_cluster_job(
    job_uuid: str,
    project_id: str,
    system_id: str,
    cluster_id: str,
    params: Dict[str, Any],
):
    """
    Run per-residue clustering in the background and update cluster metadata.
    """
    job = get_current_job()
    start_time = datetime.utcnow()

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        _update_cluster_entry(
            project_id,
            system_id,
            cluster_id,
            {
                "status": "running" if progress < 100 else "finished",
                "progress": progress,
                "status_message": status_msg,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )
        print(f"[Cluster {job_uuid}] {status_msg}")

    def progress_callback(status_msg: str, current: int, total: int):
        if not total:
            return
        ratio = max(0.0, min(1.0, current / float(total)))
        progress = 10 + int(ratio * 70)
        save_progress(status_msg, min(progress, 90))

    try:
        save_progress("Initializing...", 0)
        _update_cluster_entry(
            project_id,
            system_id,
            cluster_id,
            {"status": "running", "started_at": start_time.isoformat()},
        )

        meta_ids = params.get("metastable_ids") or []
        if not meta_ids:
            raise ValueError("Provide at least one metastable_id.")

        parallel_ok = False
        queue = None
        if job and job.connection:
            workers = Worker.all(connection=job.connection)
            parallel_ok = len(workers) > 1
            queue = Queue(name=job.origin, connection=job.connection)

        if not parallel_ok or queue is None:
            save_progress("Clustering residues...", 10)
            npz_path, meta = generate_metastable_cluster_npz(
                project_id,
                system_id,
                meta_ids,
                max_clusters_per_residue=params.get("max_clusters_per_residue", 6),
                max_cluster_frames=params.get("max_cluster_frames"),
                random_state=params.get("random_state", 0),
                contact_cutoff=params.get("contact_cutoff", 10.0),
                contact_atom_mode=params.get("contact_atom_mode", "CA"),
                cluster_algorithm=params.get("cluster_algorithm", "density_peaks"),
                dbscan_eps=params.get("dbscan_eps", 0.5),
                dbscan_min_samples=params.get("dbscan_min_samples", 5),
                hierarchical_n_clusters=params.get("hierarchical_n_clusters"),
                hierarchical_linkage=params.get("hierarchical_linkage", "ward"),
                density_maxk=params.get("density_maxk", 100),
                density_z=params.get("density_z"),
                tomato_k=params.get("tomato_k", 15),
                tomato_tau=params.get("tomato_tau", "auto"),
                tomato_k_max=params.get("tomato_k_max"),
                progress_callback=progress_callback,
            )
        else:
            save_progress("Preparing clustering workspace...", 5)
            dirs = project_store.ensure_directories(project_id, system_id)
            cluster_dir = dirs["system_dir"] / "metastable" / "clusters"
            work_dir = cluster_dir / f"{cluster_id}_work"
            manifest = prepare_cluster_workspace(
                project_id,
                system_id,
                meta_ids,
                max_clusters_per_residue=params.get("max_clusters_per_residue", 6),
                max_cluster_frames=params.get("max_cluster_frames"),
                random_state=params.get("random_state", 0),
                contact_cutoff=params.get("contact_cutoff", 10.0),
                contact_atom_mode=params.get("contact_atom_mode", "CA"),
                cluster_algorithm=params.get("cluster_algorithm", "density_peaks"),
                density_maxk=params.get("density_maxk", 100),
                density_z=params.get("density_z"),
                dbscan_eps=params.get("dbscan_eps", 0.5),
                dbscan_min_samples=params.get("dbscan_min_samples", 5),
                hierarchical_n_clusters=params.get("hierarchical_n_clusters"),
                hierarchical_linkage=params.get("hierarchical_linkage", "ward"),
                tomato_k=params.get("tomato_k", 15),
                tomato_tau=params.get("tomato_tau", "auto"),
                tomato_k_max=params.get("tomato_k_max"),
                work_dir=work_dir,
            )

            residue_total = int(manifest.get("n_residues", 0))
            if residue_total <= 0:
                raise ValueError("No residues found to cluster.")

            chunk_job_ids: List[str] = []
            for idx in range(residue_total):
                chunk_job = queue.enqueue(
                    run_cluster_chunk_job,
                    args=(str(work_dir), idx),
                    job_timeout="2h",
                    result_ttl=86400,
                    job_id=f"cluster-chunk-{cluster_id}-{idx}",
                )
                chunk_job_ids.append(chunk_job.id)

            completed = 0
            failed = 0
            while completed < residue_total:
                completed = 0
                failed = 0
                for job_id in chunk_job_ids:
                    try:
                        chunk_job = Job.fetch(job_id, connection=queue.connection)
                    except Exception:
                        failed += 1
                        continue
                    status = chunk_job.get_status()
                    if status == "finished":
                        completed += 1
                    elif status == "failed":
                        failed += 1
                if failed:
                    raise RuntimeError(f"{failed} clustering chunks failed.")
                save_progress(
                    f"Clustering residues: {completed}/{residue_total}",
                    10 + int((completed / float(residue_total)) * 70),
                )
                if completed >= residue_total:
                    break
                time.sleep(2)

            save_progress("Reducing cluster outputs...", 90)
            npz_path, meta = reduce_cluster_workspace(work_dir)

        save_progress("Saving cluster metadata...", 90)
        dirs = project_store.ensure_directories(project_id, system_id)
        try:
            rel_path = str(npz_path.relative_to(dirs["system_dir"]))
        except Exception:
            rel_path = str(npz_path)

        save_progress("Assigning clusters to MD states...", 92)
        assignments = assign_cluster_labels_to_states(npz_path, project_id, system_id)
        update_cluster_metadata_with_assignments(npz_path, assignments)

        _update_cluster_entry(
            project_id,
            system_id,
            cluster_id,
            {
                "status": "finished",
                "progress": 100,
                "path": rel_path,
                "assigned_state_paths": assignments.get("assigned_state_paths", {}),
                "assigned_metastable_paths": assignments.get("assigned_metastable_paths", {}),
                "generated_at": meta.get("generated_at") if isinstance(meta, dict) else None,
                "contact_edge_count": meta.get("contact_edge_count") if isinstance(meta, dict) else None,
                "error": None,
                "completed_at": datetime.utcnow().isoformat(),
            },
        )
        save_progress("Cluster completed", 100)
    except Exception as exc:
        _update_cluster_entry(
            project_id,
            system_id,
            cluster_id,
            {
                "status": "failed",
                "error": str(exc),
                "progress": 100,
                "status_message": "Failed",
                "completed_at": datetime.utcnow().isoformat(),
            },
        )
        print(f"[Cluster {job_uuid}] FAILED: {exc}")
        raise
