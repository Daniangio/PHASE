import json
import os
import time
import traceback
import shutil
import re
import sys
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from rq import Queue, Worker, get_current_job
from rq.job import Job
from typing import Dict, Any, List
# Ensure local repo modules shadow any installed package copies.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase.pipeline.runner import run_analysis
from phase.potts.pipeline import parse_args as parse_simulation_args
from phase.potts.pipeline import run_pipeline as run_simulation_pipeline
from phase.potts.sampling_run import run_sampling
from phase.potts.analysis_run import (
    append_state_pose_energies,
    compute_delta_transition_analysis,
    upsert_delta_energy_analysis,
    upsert_endpoint_frustration_analysis,
    upsert_delta_commitment_analysis,
    upsert_delta_js_analysis,
    compute_lambda_sweep_analysis,
    compute_md_delta_preference,
    _gibbs_relax_worker,
    _single_frame_ligand_completion_worker,
)
from phase.potts.transient_analysis import run_transient_state_analysis
from phase.potts.spectral_analysis import (
    upsert_hamiltonian_spectral_batch,
    upsert_piston_ligand_projection_analysis,
    upsert_spectral_intersection_analysis,
)
from phase.potts.orchestration import (
    aggregate_gibbs_relaxation_batch,
    aggregate_ligand_completion_batch,
    aggregate_lambda_sweep_batch,
    aggregate_potts_analysis_batch,
    aggregate_potts_nn_mapping_batch,
    aggregate_sampling_batch,
    load_partial_errors,
    load_partial_results,
    orchestration_paths,
    pickle_load,
    prepare_gibbs_relaxation_batch,
    prepare_ligand_completion_batch,
    prepare_lambda_sweep_batch,
    prepare_potts_analysis_batch,
    prepare_potts_nn_mapping_batch,
    prepare_sampling_batch,
    run_local_payload_batch,
    run_lambda_sweep_payload,
    run_sampling_chain_payload,
    _potts_analysis_payload_worker,
    _potts_nn_row_worker,
    write_partial_error,
    write_partial_result,
)
from phase.potts.potts_model import interpolate_potts_models, load_potts_model, zero_sum_gauge_model
from phase.potts.sample_io import save_sample_npz
from phase.potts.sampling import gibbs_sample_potts, make_beta_ladder, replica_exchange_gibbs_potts
from phase.common.runtime import RuntimePolicy
from phase.potts import pipeline as sim_main
from phase.potts import delta_fit as delta_fit_main
from phase.workflows.clustering import (
    generate_metastable_cluster_npz,
    prepare_cluster_workspace,
    reduce_cluster_workspace,
    run_cluster_chunk,
    build_md_eval_samples_for_cluster,
    evaluate_state_with_models,
    build_cluster_output_path,
)
from phase.workflows.backmapping import build_backmapping_npz, build_sample_backmapping_dataset
from backend.services.project_store import ProjectStore

# Define the persistent data root (aligned with PHASE_DATA_ROOT).
DATA_ROOT = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
project_store = ProjectStore()


def _default_state_assignment_workers() -> int:
    raw = os.getenv("PHASE_STATE_ASSIGN_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            pass
    return max(1, min(4, int(os.cpu_count() or 1)))


# Helper to convert NaN to None for JSON serialization
def _convert_nan_to_none(obj):
    """
    Recursively converts numpy.nan values in dicts and lists to None.
    """
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


def _supports_sim_arg(option: str) -> bool:
    try:
        parser = sim_main._build_arg_parser()
    except Exception:
        return False
    return option in parser._option_string_actions


def _collect_contact_pdbs(system_meta, selected_ids: List[str], analysis_mode: str | None) -> List[str]:
    pdbs: List[str] = []
    # Always allow macro-state IDs in the selection.
    for state_id in selected_ids:
        state = (system_meta.states or {}).get(state_id)
        if state and state.pdb_file:
            pdbs.append(state.pdb_file)
    meta_by_id = {}
    for meta in system_meta.metastable_states or []:
        meta_id = meta.get("metastable_id") or meta.get("id")
        if meta_id:
            meta_by_id[str(meta_id)] = meta
    for meta_id in selected_ids:
        meta = meta_by_id.get(str(meta_id))
        if not meta:
            continue
        rep = meta.get("representative_pdb") or meta.get("pdb_file")
        if rep:
            pdbs.append(rep)
    return pdbs


def _resolve_contact_pdbs(project_id: str, system_id: str, pdb_paths: List[str]) -> List[Path]:
    resolved: List[Path] = []
    for value in pdb_paths:
        path = Path(value)
        if not path.is_absolute():
            path = project_store.resolve_path(project_id, system_id, value)
        if path.exists():
            resolved.append(path)
    return resolved


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                parts = [part.strip() for part in item.split(",") if part.strip()]
                items.extend(parts)
            else:
                items.append(str(item))
        return items
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _build_standard_fit_dataset_from_md_samples(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    cluster_path: Path,
    sample_entries: List[Dict[str, Any]],
    sample_ids: List[str],
    output_path: Path,
) -> Dict[str, Any]:
    requested_ids = _dedupe_preserve_order(_coerce_str_list(sample_ids))
    if not requested_ids:
        raise ValueError("Standard Potts fit requires at least one MD sample.")

    cluster_npz = np.load(cluster_path, allow_pickle=True)
    residue_keys = np.asarray(cluster_npz["residue_keys"])
    if "cluster_counts" in cluster_npz:
        cluster_counts = np.asarray(cluster_npz["cluster_counts"], dtype=np.int32)
    elif "merged__cluster_counts" in cluster_npz:
        cluster_counts = np.asarray(cluster_npz["merged__cluster_counts"], dtype=np.int32)
    else:
        raise KeyError("Cluster NPZ is missing cluster_counts.")

    cluster_meta: Dict[str, Any] = {}
    if "metadata_json" in cluster_npz:
        try:
            raw_meta = cluster_npz["metadata_json"]
            if isinstance(raw_meta, np.ndarray):
                raw_meta = raw_meta.item()
            cluster_meta = json.loads(str(raw_meta))
        except Exception:
            cluster_meta = {}

    sample_by_id = {str(entry.get("sample_id") or ""): entry for entry in sample_entries if isinstance(entry, dict)}
    labels_parts: List[np.ndarray] = []
    frame_state_parts: List[np.ndarray] = []
    frame_index_parts: List[np.ndarray] = []
    selected_sample_names: List[str] = []
    selected_state_ids: List[str] = []

    for sample_id in requested_ids:
        meta = sample_by_id.get(sample_id)
        if not meta:
            raise FileNotFoundError(f"Sample '{sample_id}' not found in cluster '{cluster_id}'.")
        if str(meta.get("type") or "") != "md_eval":
            raise ValueError(f"Sample '{sample_id}' is not an MD evaluation sample.")

        raw_npz = meta.get("path") or (meta.get("paths") or {}).get("summary_npz")
        if not raw_npz:
            raise FileNotFoundError(f"Sample '{sample_id}' is missing its NPZ path.")
        sample_path = Path(str(raw_npz))
        if not sample_path.is_absolute():
            sample_path = project_store.resolve_path(project_id, system_id, str(raw_npz))
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample NPZ is missing on disk: {sample_path}")

        sample_npz = np.load(sample_path, allow_pickle=True)
        if "labels" not in sample_npz:
            raise KeyError(f"Sample '{sample_id}' does not contain assigned labels.")
        labels = np.asarray(sample_npz["labels"], dtype=np.int32)
        if labels.ndim != 2:
            raise ValueError(f"Sample '{sample_id}' labels must have shape (frames, residues).")
        if labels.shape[1] != residue_keys.shape[0]:
            raise ValueError(
                f"Sample '{sample_id}' residue count mismatch: expected {residue_keys.shape[0]}, got {labels.shape[1]}."
            )

        labels_parts.append(labels)
        selected_sample_names.append(str(meta.get("name") or sample_id))

        state_id = str(meta.get("state_id") or "").strip()
        if state_id:
            selected_state_ids.append(state_id)

        frame_state_ids = None
        if "frame_state_ids" in sample_npz:
            frame_state_ids = np.asarray(sample_npz["frame_state_ids"], dtype=str)
            if frame_state_ids.shape[0] != labels.shape[0]:
                frame_state_ids = None
        if frame_state_ids is None:
            frame_state_ids = np.asarray([state_id] * labels.shape[0], dtype=str)
        frame_state_parts.append(frame_state_ids)

        if "frame_indices" in sample_npz:
            frame_indices = np.asarray(sample_npz["frame_indices"], dtype=np.int64)
            if frame_indices.shape[0] == labels.shape[0]:
                frame_index_parts.append(frame_indices)

    if not labels_parts:
        raise ValueError("Selected MD samples did not provide any labels for fitting.")

    fit_metadata = dict(cluster_meta) if isinstance(cluster_meta, dict) else {}
    fit_metadata.update(
        {
            "fit_dataset_type": "selected_md_samples",
            "fit_sample_ids": requested_ids,
            "fit_sample_names": selected_sample_names,
            "fit_state_ids": _dedupe_preserve_order([sid for sid in selected_state_ids if sid]),
            "source_cluster_npz": _relativize_path(cluster_path),
        }
    )

    arrays: Dict[str, Any] = {
        "residue_keys": residue_keys,
        "merged__labels": np.concatenate(labels_parts, axis=0),
        "merged__labels_assigned": np.concatenate(labels_parts, axis=0),
        "cluster_counts": cluster_counts,
        "merged__cluster_counts": cluster_counts,
        "metadata_json": np.asarray(json.dumps(fit_metadata), dtype=str),
    }
    if "contact_edge_index" in cluster_npz:
        arrays["contact_edge_index"] = np.asarray(cluster_npz["contact_edge_index"], dtype=np.int32)
    if frame_state_parts:
        arrays["merged__frame_state_ids"] = np.concatenate(frame_state_parts, axis=0)
    if frame_index_parts and sum(arr.shape[0] for arr in frame_index_parts) == arrays["merged__labels"].shape[0]:
        arrays["merged__frame_indices"] = np.concatenate(frame_index_parts, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "path": output_path,
        "sample_ids": requested_ids,
        "sample_names": selected_sample_names,
        "state_ids": _dedupe_preserve_order([sid for sid in selected_state_ids if sid]),
        "n_frames": int(arrays["merged__labels"].shape[0]),
    }


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
    model_id: str,
    model_name: str | None = None,
) -> str:
    entry = None
    try:
        system_meta = project_store.get_system(project_id, system_id)
        clusters = system_meta.metastable_clusters or []
        entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    except Exception:
        entry = None

    dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = dirs["system_dir"]
    model_dir = dirs["potts_models_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    cluster_name = entry.get("name") if isinstance(entry, dict) else None
    display_name = (
        (model_name.strip() if isinstance(model_name, str) and model_name.strip() else None)
        or (f"{cluster_name} Potts Model" if isinstance(cluster_name, str) and cluster_name.strip() else None)
        or f"{cluster_id} Potts Model"
    )

    dest_path = None
    base_name = _sanitize_model_filename(display_name)
    filename = f"{base_name}.npz"
    model_bucket = model_dir / model_id
    model_bucket.mkdir(parents=True, exist_ok=True)
    dest_path = model_bucket / filename
    if dest_path.exists():
        suffix = cluster_id[:8]
        dest_path = model_bucket / f"{base_name}-{suffix}.npz"
        counter = 2
        while dest_path.exists():
            dest_path = model_bucket / f"{base_name}-{suffix}-{counter}.npz"
            counter += 1
    if model_path.resolve() != dest_path.resolve():
        shutil.copy2(model_path, dest_path)
    try:
        rel_path = str(dest_path.relative_to(system_dir))
    except Exception:
        rel_path = str(dest_path)
    def _relativize_path_value(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and "," in value:
            parts = [p.strip() for p in value.split(",") if p.strip()]
            updated = []
            for part in parts:
                updated.append(str(_relativize_path_value(part)))
            return ",".join(updated)
        try:
            path = Path(str(value))
        except Exception:
            return value
        if not path.is_absolute():
            return value
        try:
            return str(path.relative_to(system_dir))
        except Exception:
            return value

    def _normalize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        path_keys = {
            "npz",
            "data_npz",
            "plm_init_model",
            "plm_resume_model",
            "pdbs",
            "base_model",
            "active_npz",
            "inactive_npz",
        }
        cleaned = dict(raw)
        for key in path_keys:
            if key in cleaned:
                cleaned[key] = _relativize_path_value(cleaned[key])
        return cleaned

    params = _normalize_params(params)

    model_entry = {
        "model_id": model_id,
        "name": display_name,
        "path": rel_path,
        "created_at": datetime.utcnow().isoformat(),
        "source": source,
        "params": params,
    }
    if isinstance(entry, dict):
        models = entry.get("potts_models")
        if not isinstance(models, list):
            models = []
        existing = next((m for m in models if m.get("model_id") == model_id), None)
        if existing:
            existing.update(model_entry)
        else:
            models.append(model_entry)
        _update_cluster_entry(project_id, system_id, cluster_id, {"potts_models": models})
    meta_path = model_bucket / "model_metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    for key, value in model_entry.items():
        if value is None:
            continue
        if key == "created_at" and meta.get("created_at"):
            continue
        meta[key] = value
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
    
    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    if not project_id or not system_id:
        raise ValueError("Dataset reference missing project_id/system_id.")
    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

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
        state_a_id = dataset_ref.get("state_a_id")
        state_b_id = dataset_ref.get("state_b_id")
        if not state_a_id or not state_b_id:
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

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    if not project_id or not system_id:
        raise ValueError("Simulation dataset reference missing project_id/system_id.")
    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

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

        cluster_id = dataset_ref.get("cluster_id")
        if not cluster_id:
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

        sample_id = str(uuid.uuid4())
        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        sample_dir = cluster_dirs["samples_dir"] / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        results_dir = sample_dir

        sim_params = dict(params or {})
        sampling_method = sim_params.get("sampling_method") or "gibbs"
        model_rels: List[str] = []
        model_ids: List[str] = []
        model_names: List[str] = []
        model_id = None
        model_name = None
        if sim_params.get("use_potts_model", True):
            requested_ids = _coerce_str_list(sim_params.get("potts_model_ids"))
            if not requested_ids:
                requested_ids = _coerce_str_list(sim_params.get("potts_model_id"))
            requested_paths = _coerce_str_list(sim_params.get("potts_model_paths") or sim_params.get("potts_model_path"))
            models = entry.get("potts_models") if isinstance(entry, dict) else None
            model_entries: List[Dict[str, Any]] = []
            missing_ids: List[str] = []
            if requested_ids:
                if isinstance(models, list):
                    for mid in requested_ids:
                        model_entry = next((m for m in models if m.get("model_id") == mid), None)
                        if model_entry:
                            model_entries.append(model_entry)
                        else:
                            missing_ids.append(mid)
                else:
                    missing_ids = requested_ids
            if missing_ids:
                raise FileNotFoundError(f"Potts model(s) not found for ids: {', '.join(missing_ids)}")
            if not model_entries and isinstance(models, list) and models and not requested_paths and not requested_ids:
                model_entries.append(models[-1])
            for model_entry in model_entries:
                rel = model_entry.get("path")
                if rel:
                    model_rels.append(rel)
                mid = model_entry.get("model_id")
                if mid:
                    model_ids.append(mid)
                name = model_entry.get("name")
                if not name and rel:
                    name = Path(rel).stem
                if name:
                    model_names.append(name)
            for path in requested_paths:
                if path:
                    model_rels.append(path)
                    model_names.append(Path(path).stem)
            model_rels = _dedupe_preserve_order([str(p) for p in model_rels if p])
            model_ids = _dedupe_preserve_order([str(p) for p in model_ids if p])
            model_names = _dedupe_preserve_order([str(p) for p in model_names if p])
            if model_ids:
                model_id = model_ids[0]
            if model_names:
                model_name = " + ".join(model_names)
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

        model_paths: List[Path] = []
        if model_rels:
            missing_paths: List[str] = []
            for rel in model_rels:
                model_path = Path(rel)
                if not model_path.is_absolute():
                    model_path = project_store.resolve_path(project_id, system_id, rel)
                if model_path.exists():
                    model_paths.append(model_path)
                else:
                    missing_paths.append(rel)
            if missing_paths:
                raise FileNotFoundError(f"Potts model file(s) missing on disk: {', '.join(missing_paths)}")
        if not model_paths:
            raise FileNotFoundError("No Potts model selected for sampling.")
        model_path = model_paths[0]

        effective_beta = float(beta_override) if beta_override is not None else float(sim_params.get("beta", 1.0))
        effective_seed = int(sim_params.get("seed", 0) or 0)
        sa_md_sample_id = str(sim_params.get("md_sample_id") or "").strip()
        sa_md_sample_path = None
        if sampling_method == "sa":
            if not sa_md_sample_id:
                raise ValueError("SA sampling requires md_sample_id.")
            samples = entry.get("samples") if isinstance(entry, dict) else None
            sample_entry = None
            if isinstance(samples, list):
                sample_entry = next(
                    (
                        s for s in samples
                        if isinstance(s, dict)
                        and str(s.get("sample_id") or "").strip() == sa_md_sample_id
                    ),
                    None,
                )
            if not sample_entry:
                raise FileNotFoundError(f"Start MD sample not found: {sa_md_sample_id}")
            if str(sample_entry.get("type") or "").strip() != "md_eval":
                raise ValueError(f"Start sample must be an md_eval sample: {sa_md_sample_id}")
            sample_rel = (
                (sample_entry.get("paths") or {}).get("summary_npz")
                or sample_entry.get("path")
            )
            if not sample_rel:
                raise FileNotFoundError(f"MD sample NPZ path missing: {sa_md_sample_id}")
            sample_path = Path(str(sample_rel))
            if not sample_path.is_absolute():
                sample_path = project_store.resolve_path(project_id, system_id, str(sample_rel))
            if not sample_path.exists():
                raise FileNotFoundError(f"MD sample NPZ missing on disk: {sample_path}")
            sa_md_sample_path = str(sample_path)

        # Backwards-compatible support for older UIs that sent one SA range in sa_beta_schedules.
        if sampling_method == "sa":
            schedules = sim_params.get("sa_beta_schedules")
            if isinstance(schedules, list) and schedules:
                first = schedules[0]
                try:
                    hot, cold = first
                except Exception:
                    hot, cold = None, None
                if hot is not None and cold is not None:
                    sim_params.setdefault("sa_beta_hot", hot)
                    sim_params.setdefault("sa_beta_cold", cold)

        save_progress("Running Potts sampling", 20)

        # Normalize legacy SA restart modes ("prev-topk"/"prev-uniform") emitted by older UIs.
        raw_sa_restart = str(sim_params.get("sa_restart") or "independent").strip().lower()
        if raw_sa_restart in {"prev-topk", "prev-uniform", "prev", "chain"}:
            raw_sa_restart = "previous"
        elif raw_sa_restart in {"md-frame", "md_random", "md-random"}:
            raw_sa_restart = "md"
        elif raw_sa_restart in {"indep", "iid", "rand", "random"}:
            raw_sa_restart = "independent"

        prepared = prepare_sampling_batch(
            cluster_npz=str(cluster_path),
            sa_md_sample_npz=sa_md_sample_path,
            results_dir=results_dir,
            model_npz=[str(p) for p in model_paths],
            sampling_method=str(sampling_method),
            beta=effective_beta,
            seed=effective_seed,
            progress=False,
            gibbs_method=str(sim_params.get("gibbs_method") or gibbs_method),
            gibbs_samples=int(sim_params.get("gibbs_samples") or 500),
            gibbs_burnin=int(sim_params.get("gibbs_burnin") or 50),
            gibbs_thin=int(sim_params.get("gibbs_thin") or 2),
            gibbs_chains=int(sim_params.get("gibbs_chains") or 1),
            rex_betas=str(rex_betas or ""),
            rex_n_replicas=int(sim_params.get("rex_n_replicas") or 8),
            rex_beta_min=float(sim_params.get("rex_beta_min") or 0.2),
            rex_beta_max=float(sim_params.get("rex_beta_max") or 1.0),
            rex_spacing=str(sim_params.get("rex_spacing") or "geom"),
            rex_rounds=int(sim_params.get("rex_samples") or sim_params.get("rex_rounds") or 2000),
            rex_burnin_rounds=int(sim_params.get("rex_burnin") or sim_params.get("rex_burnin_rounds") or 50),
            rex_sweeps_per_round=int(sim_params.get("rex_sweeps_per_round") or 2),
            rex_thin_rounds=int(sim_params.get("rex_thin") or sim_params.get("rex_thin_rounds") or 1),
            rex_chains=int(sim_params.get("rex_chains") or sim_params.get("rex_chain_count") or 1),
            sa_reads=int(sim_params.get("sa_reads") or 2000),
            sa_chains=int(sim_params.get("sa_chains") or 1),
            sa_sweeps=int(sim_params.get("sa_sweeps") or 2000),
            sa_beta_hot=float(sim_params.get("sa_beta_hot") or 0.0),
            sa_beta_cold=float(sim_params.get("sa_beta_cold") or 0.0),
            sa_schedule_type=str(sim_params.get("sa_schedule_type") or "geometric"),
            sa_custom_beta_schedule=sim_params.get("sa_custom_beta_schedule"),
            sa_num_sweeps_per_beta=int(sim_params.get("sa_num_sweeps_per_beta") or 1),
            sa_randomize_order=bool(sim_params.get("sa_randomize_order") or False),
            sa_acceptance_criteria=str(sim_params.get("sa_acceptance_criteria") or "Metropolis"),
            sa_init=str(sim_params.get("sa_init") or "md"),
            sa_init_md_frame=int(sim_params.get("sa_init_md_frame") or -1),
            sa_restart=raw_sa_restart,
            sa_restart_topk=int(sim_params.get("sa_restart_topk") or 200),
            sa_md_sample_id=sa_md_sample_id,
            sa_md_state_ids=str(sim_params.get("sa_md_state_ids") or ""),
            penalty_safety=float(sim_params.get("penalty_safety") or 8.0),
            repair=str(sim_params.get("repair") or "none"),
        )
        payloads = prepared.get("payloads") or []
        redis_conn = getattr(job, "connection", None)
        queue_name = getattr(job, "origin", "phase-jobs")
        queue = Queue(queue_name, connection=redis_conn) if redis_conn is not None else None
        available_queue_workers = _count_queue_workers(queue) if queue is not None else 0
        distributed_parallelism = _distributed_batch_parallelism(
            requested_workers=int(prepared.get("requested_workers", 1)),
            available_queue_workers=available_queue_workers,
            n_payloads=len(payloads),
        )

        if distributed_parallelism > 1 and queue is not None:
            keys = _simulation_batch_keys(job_uuid)
            redis_conn.delete(keys["done"], keys["error"], keys["child_error"])
            redis_conn.set(keys["remaining"], int(len(payloads)), ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "running", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            save_progress(
                f"Dispatching {len(payloads)} sampling tasks across {distributed_parallelism} queue workers...",
                25,
            )
            for row in range(len(payloads)):
                queue.enqueue(
                    run_simulation_chain_job,
                    args=(job_uuid, str(results_dir), int(row), int(distributed_parallelism)),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"simulation-chain-{job_uuid}-{row:06d}",
                )
            while True:
                done = _decode_redis_value(redis_conn.get(keys["done"]))
                remaining_raw = _decode_redis_value(redis_conn.get(keys["remaining"]))
                remaining = int(remaining_raw) if remaining_raw is not None else len(payloads)
                state = _decode_redis_value(redis_conn.get(keys["state"])) or "running"
                child_error = _decode_redis_value(redis_conn.get(keys["child_error"]))
                if done:
                    break
                if remaining <= 0:
                    save_progress("Aggregating distributed sampling results...", 90)
                else:
                    completed = max(0, min(len(payloads), len(payloads) - remaining))
                    pct = 25 + int(60 * (float(completed) / float(max(1, len(payloads)))))
                    status = f"Running distributed sampling ({completed}/{len(payloads)} chains)"
                    if child_error:
                        status = f"{status}; worker error recorded, waiting for aggregation"
                    elif state and state != "running":
                        status = f"{status}; state={state}"
                    save_progress(status, pct)
                time.sleep(1.0)
            error = _decode_redis_value(redis_conn.get(keys["error"]))
            if error:
                raise RuntimeError(error)
            sampling_out = pickle_load(orchestration_paths(results_dir)["aggregate"])
        else:
            def progress_cb(message: str, current: int, total: int):
                if total <= 0:
                    return
                ratio = max(0.0, min(1.0, float(current) / float(total)))
                save_progress(message, 25 + int(60 * ratio))

            out_rows = run_local_payload_batch(
                payloads,
                worker_fn=run_sampling_chain_payload,
                max_workers=1,
                progress_callback=progress_cb,
                progress_label=str(prepared.get("progress_label") or "Running sampling"),
            )
            sampling_out = aggregate_sampling_batch(prepared, out_rows, workers_used=1)

        summary_path = Path(str(sampling_out["sample_path"]))

        potts_model_rels = [str(rel) for rel in model_rels] if model_rels else []
        cluster_name = entry.get("name") if isinstance(entry, dict) else None
        if cluster_name:
            result_payload["system_reference"]["cluster_name"] = cluster_name

        if not model_name:
            if potts_model_rels:
                model_name = " + ".join(Path(rel).stem for rel in potts_model_rels)

        sample_paths: Dict[str, str] = {}
        summary_path = Path(summary_path)
        if summary_path.exists():
            try:
                sample_paths["summary_npz"] = str(summary_path.relative_to(cluster_dirs["system_dir"]))
            except Exception:
                sample_paths["summary_npz"] = str(summary_path)
        primary_path = sample_paths.get("summary_npz")
        sample_label = sim_params.get("sample_name")
        if isinstance(sample_label, str):
            sample_label = sample_label.strip() or None

        def _filter_sampling_params(raw: Dict[str, Any]) -> Dict[str, Any]:
            # Store a normalized, minimal sampler configuration consistent with offline runs.
            method = (raw.get("sampling_method") or sampling_method or "gibbs").lower()
            out: Dict[str, Any] = {"sampling_method": method}

            defaults = {
                "rex_spacing": "geom",
                "rex_rounds": 2000,
                "rex_burnin_rounds": 50,
                "rex_thin_rounds": 1,
                "rex_n_replicas": 8,
                "rex_beta_min": 0.2,
                "rex_beta_max": 1.0,
                "sa_reads": 2000,
                "sa_sweeps": 2000,
                "sa_schedule_type": "geometric",
                "sa_num_sweeps_per_beta": 1,
                "sa_randomize_order": False,
                "sa_acceptance_criteria": "Metropolis",
                "sa_init": "md",
                "sa_restart": "independent",
                "sa_chains": 1,
            }

            def _maybe(key: str, value: Any) -> None:
                if value in (None, "", [], {}):
                    return
                if key in defaults and value == defaults[key]:
                    return
                out[key] = value

            if method == "gibbs":
                gm = str(raw.get("gibbs_method") or gibbs_method or "rex").lower()
                _maybe("gibbs_method", gm)
                _maybe("beta", float(effective_beta))

                # Prefer storing the explicit ladder (if any), otherwise the generation params.
                if rex_betas:
                    _maybe("rex_betas", str(rex_betas))
                else:
                    _maybe("rex_beta_min", float(raw.get("rex_beta_min")) if raw.get("rex_beta_min") is not None else None)
                    _maybe("rex_beta_max", float(raw.get("rex_beta_max")) if raw.get("rex_beta_max") is not None else None)
                    _maybe("rex_n_replicas", int(raw.get("rex_n_replicas")) if raw.get("rex_n_replicas") is not None else None)
                    _maybe("rex_spacing", str(raw.get("rex_spacing") or "geom"))

                # UI uses rex_samples/burnin/thin naming; normalize to *_rounds.
                rounds = raw.get("rex_samples") if raw.get("rex_samples") is not None else raw.get("rex_rounds")
                burnin = raw.get("rex_burnin") if raw.get("rex_burnin") is not None else raw.get("rex_burnin_rounds")
                thin = raw.get("rex_thin") if raw.get("rex_thin") is not None else raw.get("rex_thin_rounds")
                _maybe("rex_rounds", int(rounds) if rounds is not None else None)
                _maybe("rex_burnin_rounds", int(burnin) if burnin is not None else None)
                _maybe("rex_thin_rounds", int(thin) if thin is not None else None)
            else:
                # Keep a few key SA parameters even when defaults are used (mirrors offline sampling metadata).
                out["beta"] = float(effective_beta)
                sr = str(raw.get("sa_restart") or "independent").strip().lower()
                if sr in {"prev-topk", "prev-uniform", "prev", "chain"}:
                    sr = "previous"
                elif sr in {"md-frame", "md_random", "md-random"}:
                    sr = "md"
                elif sr in {"indep", "iid", "rand", "random"}:
                    sr = "independent"
                out["sa_restart"] = sr
                out["sa_sweeps"] = int(raw.get("sa_sweeps") or 2000)

                _maybe("sa_reads", int(raw.get("sa_reads")) if raw.get("sa_reads") is not None else None)
                _maybe("sa_beta_hot", float(raw.get("sa_beta_hot")) if raw.get("sa_beta_hot") is not None else None)
                _maybe("sa_beta_cold", float(raw.get("sa_beta_cold")) if raw.get("sa_beta_cold") is not None else None)
                _maybe("sa_schedule_type", str(raw.get("sa_schedule_type") or "geometric"))
                custom_schedule = raw.get("sa_custom_beta_schedule")
                if custom_schedule not in (None, "", [], {}):
                    if isinstance(custom_schedule, str):
                        _maybe("sa_custom_beta_schedule", custom_schedule)
                    else:
                        _maybe("sa_custom_beta_schedule", list(custom_schedule))
                _maybe("sa_num_sweeps_per_beta", int(raw.get("sa_num_sweeps_per_beta")) if raw.get("sa_num_sweeps_per_beta") is not None else None)
                _maybe("sa_randomize_order", bool(raw.get("sa_randomize_order")) if raw.get("sa_randomize_order") is not None else None)
                _maybe("sa_acceptance_criteria", str(raw.get("sa_acceptance_criteria") or "Metropolis"))
                _maybe("sa_init", str(raw.get("sa_init") or "md"))
                _maybe("sa_init_md_frame", int(raw.get("sa_init_md_frame")) if raw.get("sa_init_md_frame") is not None else None)
                _maybe("sa_chains", int(raw.get("sa_chains")) if raw.get("sa_chains") is not None else None)
                _maybe("sa_md_state_ids", str(raw.get("sa_md_state_ids") or ""))
                _maybe("sa_restart_topk", int(raw.get("sa_restart_topk")) if raw.get("sa_restart_topk") is not None else None)

            return out

        sample_entry = {
            "sample_id": sample_id,
            "name": sample_label or f"Sampling {datetime.utcnow().strftime('%Y%m%d %H:%M')}",
            "type": "potts_sampling",
            "method": "sa" if sampling_method == "sa" else "gibbs",
            "source": "simulation",
            "model_id": model_id,
            "model_ids": model_ids or None,
            "model_names": model_names or None,
            "created_at": datetime.utcnow().isoformat(),
            "path": primary_path,
            "paths": sample_paths,
            "params": _filter_sampling_params(sim_params),
        }
        if isinstance(entry, dict):
            samples = entry.get("samples")
            if not isinstance(samples, list):
                samples = []
            samples.append(sample_entry)
            _update_cluster_entry(project_id, system_id, cluster_id, {"samples": samples})

        result_payload["system_reference"]["sample_id"] = sample_id
        result_payload["system_reference"]["sample_name"] = sample_entry.get("name")
        result_payload["system_reference"]["potts_model_id"] = model_id
        result_payload["system_reference"]["potts_model_name"] = model_name
        if model_ids:
            result_payload["system_reference"]["potts_model_ids"] = model_ids

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "results_dir": _relativize_path(results_dir),
            "summary_npz": _relativize_path(summary_path) if summary_path else None,
            "cluster_npz": _relativize_path(cluster_path),
            "potts_model": potts_model_rels[0] if potts_model_rels else (_relativize_path(model_path) if model_path else None),
            "potts_models": potts_model_rels or None,
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




def run_lambda_sweep_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Run the validation_ladder4.MD lambda-interpolation experiment:
      - sample N ensembles from interpolated endpoint models with λ in [0,1]
      - persist each ensemble as an independent sample folder (correlated via series metadata)
      - compute a dedicated lambda_sweep analysis vs flexible reference/comparison samples

    Results:
      - samples written under clusters/<cluster_id>/samples/<sample_id>/sample.npz
      - analysis written under clusters/<cluster_id>/analyses/lambda_sweep/<analysis_id>/
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"lambda-sweep-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("lambda_sweep requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "lambda_sweep",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def force_single_thread():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[LambdaSweep {job_uuid}] {status_msg}")

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

        workers = params.get("workers")
        workers_val = int(workers) if workers is not None else None

        reference_a = str(params.get("reference_sample_id_a") or params.get("md_sample_id_1") or "").strip()
        reference_b = str(params.get("reference_sample_id_b") or params.get("md_sample_id_2") or "").strip()
        comparison_ids = params.get("comparison_sample_ids") or []
        if isinstance(comparison_ids, str):
            comparison_ids = [v.strip() for v in comparison_ids.split(",") if v.strip()]
        elif isinstance(comparison_ids, (tuple, list, set)):
            comparison_ids = [str(v).strip() for v in comparison_ids if str(v).strip()]
        else:
            comparison_ids = []
        if not comparison_ids:
            legacy_c = str(params.get("md_sample_id_3") or "").strip()
            if legacy_c:
                comparison_ids = [legacy_c]

        save_progress("Preparing lambda sweep...", 10)
        prepared = prepare_lambda_sweep_batch(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_a_id=str(params.get("model_a_id") or "").strip(),
            model_b_id=str(params.get("model_b_id") or "").strip(),
            reference_sample_id_a=reference_a,
            reference_sample_id_b=reference_b,
            comparison_sample_ids=comparison_ids,
            series_id=str(params.get("series_id") or "").strip() or None,
            series_label=str(params.get("series_label") or "").strip() or None,
            lambda_count=int(params.get("lambda_count") or 11),
            alpha=float(params.get("alpha") or 0.5),
            md_label_mode=str(params.get("md_label_mode") or "assigned"),
            keep_invalid=bool(params.get("keep_invalid", False)),
            gibbs_method=str(params.get("gibbs_method") or "rex"),
            beta=float(params.get("beta") or 1.0),
            seed=int(params.get("seed") or 0),
            gibbs_samples=int(params.get("gibbs_samples") or 500),
            gibbs_burnin=int(params.get("gibbs_burnin") or 50),
            gibbs_thin=int(params.get("gibbs_thin") or 2),
            gibbs_chains=int(params.get("gibbs_chains") or 1),
            rex_betas=params.get("rex_betas"),
            rex_beta_min=float(params.get("rex_beta_min") or 0.2),
            rex_beta_max=float(params.get("rex_beta_max") or 1.0),
            rex_spacing=str(params.get("rex_spacing") or "geom"),
            rex_n_replicas=int(params.get("rex_n_replicas") or 8),
            rex_rounds=int(params.get("rex_rounds") or params.get("rex_samples") or 2000),
            rex_burnin_rounds=int(params.get("rex_burnin_rounds") or params.get("rex_burnin") or 50),
            rex_sweeps_per_round=int(params.get("rex_sweeps_per_round") or 2),
            rex_thin_rounds=int(params.get("rex_thin_rounds") or params.get("rex_thin") or 1),
            rex_chains=int(params.get("rex_chains") or 1),
            n_workers=workers_val,
        )
        payloads = prepared.get("payloads") or []
        analysis_id = str(prepared.get("analysis_id") or "")
        analysis_dir = Path(str(prepared.get("analysis_dir") or "")).resolve()
        redis_conn = getattr(job, "connection", None)
        queue_name = getattr(job, "origin", "phase-jobs")
        queue = Queue(queue_name, connection=redis_conn) if redis_conn is not None else None
        available_queue_workers = _count_queue_workers(queue) if queue is not None else 0
        distributed_parallelism = _distributed_batch_parallelism(
            requested_workers=int(prepared.get("requested_workers", workers_val or 1)),
            available_queue_workers=available_queue_workers,
            n_payloads=len(payloads),
        )

        if distributed_parallelism > 1 and queue is not None:
            keys = _lambda_sweep_batch_keys(analysis_id)
            redis_conn.delete(keys["done"], keys["error"], keys["child_error"])
            redis_conn.set(keys["remaining"], int(len(payloads)), ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "running", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            save_progress(
                f"Dispatching {len(payloads)} lambda sweep tasks across {distributed_parallelism} queue workers...",
                15,
            )
            for row in range(len(payloads)):
                queue.enqueue(
                    run_lambda_sweep_payload_job,
                    args=(analysis_id, str(analysis_dir), int(row), int(distributed_parallelism)),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"lambda-sweep-task-{analysis_id}-{row:06d}",
                )
            while True:
                done = _decode_redis_value(redis_conn.get(keys["done"]))
                remaining_raw = _decode_redis_value(redis_conn.get(keys["remaining"]))
                remaining = int(remaining_raw) if remaining_raw is not None else len(payloads)
                state = _decode_redis_value(redis_conn.get(keys["state"])) or "running"
                child_error = _decode_redis_value(redis_conn.get(keys["child_error"]))
                if done:
                    break
                if remaining <= 0:
                    save_progress("Aggregating distributed lambda sweep results...", 92)
                else:
                    completed = max(0, min(len(payloads), len(payloads) - remaining))
                    pct = 15 + int(70 * (float(completed) / float(max(1, len(payloads)))))
                    status = f"Running distributed lambda sweep ({completed}/{len(payloads)} tasks)"
                    if child_error:
                        status = f"{status}; worker error recorded, waiting for aggregation"
                    elif state and state != "running":
                        status = f"{status}; state={state}"
                    save_progress(status, pct)
                time.sleep(1.0)
            error = _decode_redis_value(redis_conn.get(keys["error"]))
            if error:
                raise RuntimeError(error)
            out = pickle_load(orchestration_paths(analysis_dir)["aggregate"])
        else:
            save_progress("Running lambda sweep locally...", 15)

            def progress_cb(message: str, current: int, total: int):
                if total <= 0:
                    return
                ratio = max(0.0, min(1.0, float(current) / float(total)))
                save_progress(message, 15 + int(70 * ratio))

            out_rows = run_local_payload_batch(
                payloads,
                worker_fn=run_lambda_sweep_payload,
                max_workers=1,
                progress_callback=progress_cb,
                progress_label="Running lambda sweep",
            )
            out = aggregate_lambda_sweep_batch(prepared, out_rows, workers_used=1)

        meta = out.get("metadata") or {}
        analysis_id = str(out.get("analysis_id") or meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("lambda_sweep did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "lambda_sweep",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
            "series_id": out.get("series_id"),
            "sample_ids": out.get("sample_ids") or [],
            "sample_names": out.get("sample_names") or [],
        }

    except Exception as e:
        print(f"[LambdaSweep {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Lambda sweep completed", 100)
    return sanitized_payload

def run_potts_analysis_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Compute Potts sample analyses for one cluster:
      - MD-vs-sample distribution metrics (node/edge JS)
      - optional per-sample energies under a chosen model

    Results are written under clusters/<cluster_id>/analyses/.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"potts-analysis-{job_uuid}"

    result_payload = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "potts_analysis",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": dataset_ref.get("project_id"),
            "system_id": dataset_ref.get("system_id"),
            "cluster_id": dataset_ref.get("cluster_id"),
        },
        "error": None,
        "completed_at": None,
    }

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("Potts analysis dataset reference missing project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[PottsAnalysis {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        # Validate cluster existence
        system_meta = project_store.get_system(project_id, system_id)
        clusters = system_meta.metastable_clusters or []
        entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
        if not entry:
            raise FileNotFoundError(f"Cluster '{cluster_id}' not found.")

        model_ref = None
        model_id = params.get("model_id")
        model_path = params.get("model_path")
        if model_path:
            model_ref = str(model_path)
        elif model_id:
            model_ref = str(model_id)

        md_label_mode = (params.get("md_label_mode") or "assigned").lower()
        keep_invalid = bool(params.get("keep_invalid", False))
        pose_only = bool(params.get("pose_only", False))
        state_pose_ids = [str(v).strip() for v in (params.get("state_pose_ids") or []) if str(v).strip()]

        save_progress("Running analyses...", 20)
        if pose_only:
            if not model_ref:
                raise ValueError("state pose energy analysis requires model_id or model_path.")
            if not state_pose_ids:
                raise ValueError("state pose energy analysis requires at least one state_pose_id.")
            state_eval_workers = _default_state_assignment_workers()
            summary = append_state_pose_energies(
                project_id=project_id,
                system_id=system_id,
                cluster_id=cluster_id,
                model_ref=model_ref,
                state_ids=state_pose_ids,
                state_eval_workers=state_eval_workers,
                progress_callback=save_progress,
            )
        else:
            workers = params.get("workers")
            workers_val = int(workers) if workers not in (None, "") else None
            save_progress("Preparing Sampling Explorer analysis...", 10)
            prepared = prepare_potts_analysis_batch(
                project_id=project_id,
                system_id=system_id,
                cluster_id=cluster_id,
                model_ref=model_ref,
                comparison_sample_ids=params.get("sample_ids"),
                md_label_mode=md_label_mode,
                drop_invalid=not keep_invalid,
                n_workers=workers_val,
                analysis_edge_mode=params.get("analysis_edge_mode"),
                analysis_contact_cutoff=params.get("analysis_contact_cutoff"),
                analysis_contact_atom_mode=params.get("analysis_contact_atom_mode"),
                analysis_contact_state_ids=params.get("analysis_contact_state_ids"),
                analysis_contact_pdbs=params.get("analysis_contact_pdbs"),
            )
            payloads = prepared.get("payloads") or []
            analysis_id = str(prepared.get("analysis_id") or "")
            analysis_dir = Path(str(prepared.get("analysis_dir") or "")).resolve()
            redis_conn = getattr(job, "connection", None)
            queue_name = getattr(job, "origin", "phase-jobs")
            queue = Queue(queue_name, connection=redis_conn) if redis_conn is not None else None
            available_queue_workers = _count_queue_workers(queue) if queue is not None else 0
            distributed_parallelism = _distributed_batch_parallelism(
                requested_workers=int(prepared.get("requested_workers", workers_val or 0)),
                available_queue_workers=available_queue_workers,
                n_payloads=len(payloads),
            )

            if distributed_parallelism > 1 and queue is not None:
                keys = _potts_analysis_batch_keys(analysis_id)
                redis_conn.delete(keys["done"], keys["error"], keys["child_error"])
                redis_conn.set(keys["remaining"], int(len(payloads)), ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
                redis_conn.set(keys["state"], "running", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
                save_progress(
                    f"Dispatching {len(payloads)} Sampling Explorer tasks across {distributed_parallelism} queue workers...",
                    15,
                )
                for row in range(len(payloads)):
                    queue.enqueue(
                        run_potts_analysis_payload_job,
                        args=(analysis_id, str(analysis_dir), int(row), int(distributed_parallelism)),
                        job_timeout="8h",
                        result_ttl=86400,
                        job_id=f"potts-analysis-task-{analysis_id}-{row:06d}",
                    )
                while True:
                    done = _decode_redis_value(redis_conn.get(keys["done"]))
                    remaining_raw = _decode_redis_value(redis_conn.get(keys["remaining"]))
                    remaining = int(remaining_raw) if remaining_raw is not None else len(payloads)
                    state = _decode_redis_value(redis_conn.get(keys["state"])) or "running"
                    child_error = _decode_redis_value(redis_conn.get(keys["child_error"]))
                    if done:
                        break
                    if remaining <= 0:
                        save_progress("Aggregating distributed Sampling Explorer results...", 92)
                    else:
                        completed = max(0, min(len(payloads), len(payloads) - remaining))
                        pct = 15 + int(70 * (float(completed) / float(max(1, len(payloads)))))
                        status = f"Running distributed Sampling Explorer analysis ({completed}/{len(payloads)} tasks)"
                        if child_error:
                            status = f"{status}; worker error recorded, waiting for aggregation"
                        elif state and state != "running":
                            status = f"{status}; state={state}"
                        save_progress(status, pct)
                    time.sleep(1.0)
                error = _decode_redis_value(redis_conn.get(keys["error"]))
                if error:
                    raise RuntimeError(error)
                out = pickle_load(orchestration_paths(analysis_dir)["aggregate"])
            else:
                save_progress("Running Sampling Explorer analysis locally...", 15)

                def progress_cb(message: str, current: int, total: int):
                    if total <= 0:
                        return
                    ratio = max(0.0, min(1.0, float(current) / float(total)))
                    save_progress(message, 15 + int(70 * ratio))

                requested_workers = int(prepared.get("requested_workers") or 0)
                if requested_workers > 0:
                    max_workers = requested_workers
                else:
                    max_workers = max(1, min(int(os.cpu_count() or 1), max(1, len(payloads))))
                if job is not None:
                    max_workers = 1
                out_rows = run_local_payload_batch(
                    payloads,
                    worker_fn=_potts_analysis_payload_worker,
                    max_workers=max_workers,
                    progress_callback=progress_cb,
                    progress_label="Running Sampling Explorer analysis",
                )
                out = aggregate_potts_analysis_batch(
                    prepared,
                    out_rows,
                    workers_used=(1 if not payloads else max(1, min(max_workers, len(payloads)))),
                )
            summary = out.get("metadata") or {}

        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        analyses_dir = cluster_dirs["cluster_dir"] / "analyses"

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analyses_dir": _relativize_path(analyses_dir),
            "summary": summary,
        }

    except Exception as e:
        print(f"[PottsAnalysis {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e
    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Potts analysis completed", 100)
    return sanitized_payload


def run_potts_nearest_neighbor_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"potts-nn-mapping-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("Potts NN mapping requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"
    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "potts_nn_mapping",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[PottsNNMapping {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {exc}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        workers = params.get("workers")
        workers_val = int(workers) if workers is not None else None
        save_progress("Preparing nearest-neighbor mapping...", 10)
        prepared = prepare_potts_nn_mapping_batch(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_id=str(params.get("model_id") or "").strip() or None,
            model_path=str(params.get("model_path") or "").strip() or None,
            sample_id=str(params.get("sample_id") or "").strip(),
            md_sample_id=str(params.get("md_sample_id") or "").strip(),
            md_label_mode=str(params.get("md_label_mode") or "assigned"),
            keep_invalid=bool(params.get("keep_invalid", False)),
            use_unique=bool(params.get("use_unique", True)),
            normalize=bool(params.get("normalize", True)),
            compute_per_residue=bool(params.get("compute_per_residue", True)),
            alpha=float(params.get("alpha") or 0.75),
            beta_node=float(params.get("beta_node") or 1.0),
            beta_edge=float(params.get("beta_edge") or 1.0),
            top_k_candidates=(int(params.get("top_k_candidates")) if params.get("top_k_candidates") not in (None, "", 0) else None),
            chunk_size=int(params.get("chunk_size") or 256),
            distance_thresholds=params.get("distance_thresholds") or [0.05, 0.1, 0.2],
            n_workers=workers_val,
        )
        payloads = prepared.get("payloads") or []
        analysis_id = str(prepared.get("analysis_id") or "")
        analysis_dir = Path(str(prepared.get("analysis_dir") or "")).resolve()
        redis_conn = getattr(job, "connection", None)
        queue_name = getattr(job, "origin", "phase-jobs")
        queue = Queue(queue_name, connection=redis_conn) if redis_conn is not None else None
        available_queue_workers = _count_queue_workers(queue) if queue is not None else 0
        distributed_parallelism = _distributed_batch_parallelism(
            requested_workers=int(prepared.get("requested_workers", workers_val or 1)),
            available_queue_workers=available_queue_workers,
            n_payloads=len(payloads),
        )

        if distributed_parallelism > 1 and queue is not None:
            keys = _potts_nn_mapping_batch_keys(analysis_id)
            redis_conn.delete(keys["done"], keys["error"], keys["child_error"])
            redis_conn.set(keys["remaining"], int(len(payloads)), ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "running", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            save_progress(
                f"Dispatching {len(payloads)} nearest-neighbor tasks across {distributed_parallelism} queue workers...",
                15,
            )
            for row in range(len(payloads)):
                queue.enqueue(
                    run_potts_nn_mapping_payload_job,
                    args=(analysis_id, str(analysis_dir), int(row), int(distributed_parallelism)),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"potts-nn-mapping-task-{analysis_id}-{row:06d}",
                )
            while True:
                done = _decode_redis_value(redis_conn.get(keys["done"]))
                remaining_raw = _decode_redis_value(redis_conn.get(keys["remaining"]))
                remaining = int(remaining_raw) if remaining_raw is not None else len(payloads)
                state = _decode_redis_value(redis_conn.get(keys["state"])) or "running"
                child_error = _decode_redis_value(redis_conn.get(keys["child_error"]))
                if done:
                    break
                if remaining <= 0:
                    save_progress("Aggregating distributed nearest-neighbor results...", 92)
                else:
                    completed = max(0, min(len(payloads), len(payloads) - remaining))
                    pct = 15 + int(70 * (float(completed) / float(max(1, len(payloads)))))
                    status = f"Running distributed nearest-neighbor mapping ({completed}/{len(payloads)} uniques)"
                    if child_error:
                        status = f"{status}; worker error recorded, waiting for aggregation"
                    elif state and state != "running":
                        status = f"{status}; state={state}"
                    save_progress(status, pct)
                time.sleep(1.0)
            error = _decode_redis_value(redis_conn.get(keys["error"]))
            if error:
                raise RuntimeError(error)
            out = pickle_load(orchestration_paths(analysis_dir)["aggregate"])
        else:
            save_progress("Running nearest-neighbor mapping locally...", 15)

            def progress_cb(message: str, current: int, total: int):
                if total <= 0:
                    return
                ratio = max(0.0, min(1.0, float(current) / float(total)))
                save_progress(message, 15 + int(70 * ratio))

            max_workers = max(1, int(prepared.get("requested_workers") or 1))
            out_rows = run_local_payload_batch(
                payloads,
                worker_fn=_potts_nn_row_worker,
                max_workers=max_workers,
                progress_callback=progress_cb,
                progress_label="Running Potts nearest-neighbor mapping",
            )
            out = aggregate_potts_nn_mapping_batch(
                prepared,
                out_rows,
                workers_used=(1 if not payloads else max(1, min(max_workers, len(payloads)))),
            )

        meta = out.get("metadata") or {}
        analysis_id = str(out.get("analysis_id") or meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("potts_nn_mapping did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "potts_nn_mapping",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
            "summary": meta.get("summary") or {},
        }
    except Exception as exc:
        print(f"[PottsNNMapping {job_uuid}] FAILED: {exc}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(exc)
        raise
    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Potts NN mapping completed", 100)
    return sanitized_payload


def run_md_samples_refresh_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Recompute md_eval samples for all descriptor-ready states of a system, for a given cluster.

    This overwrites existing md_eval sample folders by default (stable sample_id per state)
    and updates sample metadata so the UI/offline console immediately see refreshed MD samples.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"md-samples-refresh-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("md_samples_refresh requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "md_samples_refresh",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[MdSamplesRefresh {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    overwrite = bool(params.get("overwrite", True))
    cleanup = bool(params.get("cleanup", True))
    state_eval_workers = _default_state_assignment_workers()
    requested_state_ids: List[str] = []
    seen_state_ids: set[str] = set()
    for raw in params.get("state_ids") or []:
        sid = str(raw or "").strip()
        if not sid or sid in seen_state_ids:
            continue
        seen_state_ids.add(sid)
        requested_state_ids.append(sid)

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        system_meta = project_store.get_system(project_id, system_id)
        clusters = system_meta.metastable_clusters or []
        entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
        if not isinstance(entry, dict):
            raise FileNotFoundError(f"Cluster '{cluster_id}' not found.")

        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        samples_dir = cluster_dirs["samples_dir"]

        samples = entry.get("samples")
        if not isinstance(samples, list):
            samples = []

        descriptor_states = [s for s in system_meta.states.values() if getattr(s, "descriptor_file", None)]
        if requested_state_ids:
            requested = set(requested_state_ids)
            descriptor_states = [
                state
                for state in descriptor_states
                if str(getattr(state, "state_id", "") or "") in requested
            ]
        total = len(descriptor_states)
        if total == 0:
            result_payload["status"] = "finished"
            result_payload["results"] = {
                "refreshed": 0,
                "states": 0,
                "sample_ids": [],
                "requested_state_ids": requested_state_ids,
            }
            return result_payload

        refreshed_ids: List[str] = []
        for idx, state in enumerate(descriptor_states, start=1):
            state_id = getattr(state, "state_id", None) or ""
            if not state_id:
                continue

            pct = 5 + int(85 * (idx - 1) / max(1, total))
            save_progress(f"Evaluating {state_id} ({idx}/{total})", pct)

            existing = [
                s
                for s in samples
                if isinstance(s, dict) and (s.get("type") or "") == "md_eval" and (s.get("state_id") or "") == state_id and s.get("sample_id")
            ]
            keep_id = None
            dup_ids: List[str] = []
            if existing:
                existing.sort(key=lambda s: str(s.get("created_at") or ""))
                keep = existing[-1]
                keep_id = str(keep.get("sample_id")) if keep.get("sample_id") else None
                dup_ids = [str(s.get("sample_id")) for s in existing[:-1] if s.get("sample_id")]

            reuse_id = keep_id if overwrite else None
            if cleanup and dup_ids:
                for sid in dup_ids:
                    try:
                        shutil.rmtree(samples_dir / sid, ignore_errors=True)
                    except Exception:
                        pass
                dup_set = set(dup_ids)
                samples = [
                    s
                    for s in samples
                    if not (
                        isinstance(s, dict)
                        and (s.get("type") or "") == "md_eval"
                        and (s.get("state_id") or "") == state_id
                        and s.get("sample_id") in dup_set
                    )
                ]

            sample_entry = evaluate_state_with_models(
                project_id,
                system_id,
                cluster_id,
                state_id,
                store=project_store,
                sample_id=reuse_id,
                workers=state_eval_workers,
                progress_callback=save_progress,
            )

            out_id = sample_entry.get("sample_id")
            replaced = False
            for j, existing_entry in enumerate(samples):
                if not isinstance(existing_entry, dict):
                    continue
                if existing_entry.get("sample_id") == out_id:
                    samples[j] = sample_entry
                    replaced = True
                    break
            if not replaced:
                if overwrite:
                    # Drop any md_eval entries for this state (malformed or missing ids).
                    samples = [
                        s
                        for s in samples
                        if not (isinstance(s, dict) and (s.get("type") or "") == "md_eval" and (s.get("state_id") or "") == state_id)
                    ]
                samples.append(sample_entry)

            entry["samples"] = samples
            system_meta.metastable_clusters = clusters
            project_store.save_system(system_meta)

            if out_id:
                refreshed_ids.append(str(out_id))

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "refreshed": len(refreshed_ids),
            "states": total,
            "sample_ids": refreshed_ids,
            "requested_state_ids": requested_state_ids,
        }

    except Exception as e:
        print(f"[MdSamplesRefresh {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("MD sample refresh completed", 100)
    return sanitized_payload


def run_delta_eval_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Compute per-residue/per-edge delta preferences on an MD sample for two selected Potts models.
    This implements point (4) in validation_ladder2.MD.

    Results are written under clusters/<cluster_id>/analyses/delta_eval/.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"delta-eval-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("delta_eval requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "delta_eval",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[DeltaEval {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        md_sample_id = str(params.get("md_sample_id") or "").strip()
        model_a_id = str(params.get("model_a_id") or "").strip()
        model_b_id = str(params.get("model_b_id") or "").strip()
        if not md_sample_id or not model_a_id or not model_b_id:
            raise ValueError("delta_eval requires md_sample_id, model_a_id, model_b_id.")
        if model_a_id == model_b_id:
            raise ValueError("Select two different models.")

        md_label_mode = (params.get("md_label_mode") or "assigned").lower()
        keep_invalid = bool(params.get("keep_invalid", False))

        save_progress("Computing delta preferences...", 20)
        payload = compute_md_delta_preference(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            md_sample_id=md_sample_id,
            model_a_ref=model_a_id,
            model_b_ref=model_b_id,
            md_label_mode=md_label_mode,
            drop_invalid=not keep_invalid,
            include_potts_overlay=True,
        )

        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        system_dir = cluster_dirs["system_dir"]
        analyses_dir = cluster_dirs["cluster_dir"] / "analyses" / "delta_eval"
        analyses_dir.mkdir(parents=True, exist_ok=True)

        analysis_id = str(uuid.uuid4())
        out_dir = analyses_dir / analysis_id
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "analysis.npz"
        meta_path = out_dir / "analysis_metadata.json"

        potts_a = payload.get("delta_energy_potts_a")
        if potts_a is None:
            potts_a = np.zeros((0,), dtype=float)
        potts_b = payload.get("delta_energy_potts_b")
        if potts_b is None:
            potts_b = np.zeros((0,), dtype=float)

        # Persist NPZ
        np.savez_compressed(
            npz_path,
            delta_energy=np.asarray(payload["delta_energy"], dtype=float),
            delta_residue_mean=np.asarray(payload["delta_residue_mean"], dtype=float),
            delta_residue_std=np.asarray(payload["delta_residue_std"], dtype=float),
            edges=np.asarray(payload["edges"], dtype=int),
            delta_edge_mean=np.asarray(payload["delta_edge_mean"], dtype=float),
            delta_energy_potts_a=np.asarray(potts_a, dtype=float),
            delta_energy_potts_b=np.asarray(potts_b, dtype=float),
        )

        # Persist metadata (cluster-relative paths)
        meta = {
            "analysis_id": analysis_id,
            "analysis_type": "delta_eval",
            "created_at": datetime.utcnow().isoformat(),
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
            "md_sample_id": md_sample_id,
            "md_sample_name": payload.get("md_sample_name"),
            "model_a_id": model_a_id,
            "model_a_name": payload.get("model_a_name"),
            "model_b_id": model_b_id,
            "model_b_name": payload.get("model_b_name"),
            "drop_invalid": bool(not keep_invalid),
            "md_label_mode": md_label_mode,
            "paths": {"analysis_npz": str(npz_path.relative_to(system_dir))},
            "potts_overlay": {
                "sample_ids_a": payload.get("potts_sample_ids_a") or [],
                "sample_ids_b": payload.get("potts_sample_ids_b") or [],
                "frames_a": int(np.asarray(potts_a).shape[0]),
                "frames_b": int(np.asarray(potts_b).shape[0]),
            },
            "summary": {
                "frames": int(np.asarray(payload["delta_energy"]).shape[0]),
                "residues": int(np.asarray(payload["delta_residue_mean"]).shape[0]),
                "edges": int(np.asarray(payload["delta_edge_mean"]).shape[0]),
            },
        }
        meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "delta_eval",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(out_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[DeltaEval {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Delta eval completed", 100)
    return sanitized_payload


def run_delta_transition_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Implements the "TS-like" operational analysis described in validation_ladder3.MD (Step 1-4).

    Results are written under clusters/<cluster_id>/analyses/delta_transition/.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"delta-transition-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("delta_transition requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "delta_transition",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[DeltaTransition {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        active_id = str(params.get("active_md_sample_id") or "").strip()
        inactive_id = str(params.get("inactive_md_sample_id") or "").strip()
        pas_id = str(params.get("pas_md_sample_id") or "").strip()
        model_a_id = str(params.get("model_a_id") or "").strip()
        model_b_id = str(params.get("model_b_id") or "").strip()
        if not active_id or not inactive_id or not pas_id or not model_a_id or not model_b_id:
            raise ValueError(
                "delta_transition requires active_md_sample_id, inactive_md_sample_id, pas_md_sample_id, model_a_id, model_b_id."
            )
        if model_a_id == model_b_id:
            raise ValueError("Select two different models.")

        md_label_mode = (params.get("md_label_mode") or "assigned").lower()
        keep_invalid = bool(params.get("keep_invalid", False))
        band_fraction = float(params.get("band_fraction", 0.1))
        top_k_residues = int(params.get("top_k_residues", 20))
        # Store commitment for the top-K edges by |ΔJ| (default high enough for rich 3D link visualization).
        top_k_edges = int(params.get("top_k_edges", 2000))
        seed = int(params.get("seed", 0))

        save_progress("Computing TS-band analysis...", 20)
        payload = compute_delta_transition_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            active_md_sample_id=active_id,
            inactive_md_sample_id=inactive_id,
            pas_md_sample_id=pas_id,
            model_a_ref=model_a_id,
            model_b_ref=model_b_id,
            md_label_mode=md_label_mode,
            drop_invalid=not keep_invalid,
            band_fraction=band_fraction,
            top_k_residues=top_k_residues,
            top_k_edges=top_k_edges,
            seed=seed,
        )

        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        system_dir = cluster_dirs["system_dir"]
        analyses_dir = cluster_dirs["cluster_dir"] / "analyses" / "delta_transition"
        analyses_dir.mkdir(parents=True, exist_ok=True)

        analysis_id = str(uuid.uuid4())
        out_dir = analyses_dir / analysis_id
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "analysis.npz"
        meta_path = out_dir / "analysis_metadata.json"

        edges_arr = payload.get("edges")
        if edges_arr is None:
            edges_arr = np.zeros((0, 2), dtype=int)

        np.savez_compressed(
            npz_path,
            delta_energy_active=np.asarray(payload["delta_energy_active"], dtype=float),
            delta_energy_inactive=np.asarray(payload["delta_energy_inactive"], dtype=float),
            delta_energy_pas=np.asarray(payload["delta_energy_pas"], dtype=float),
            edges=np.asarray(edges_arr, dtype=int),
            z_active=np.asarray(payload["z_active"], dtype=float),
            z_inactive=np.asarray(payload["z_inactive"], dtype=float),
            z_pas=np.asarray(payload["z_pas"], dtype=float),
            median_train=np.asarray([payload["median_train"]], dtype=float),
            mad_train=np.asarray([payload["mad_train"]], dtype=float),
            tau=np.asarray([payload["tau"]], dtype=float),
            p_train=np.asarray([payload["p_train"]], dtype=float),
            p_pas=np.asarray([payload["p_pas"]], dtype=float),
            enrichment=np.asarray([payload["enrichment"]], dtype=float),
            D_residue=np.asarray(payload["D_residue"], dtype=float),
            top_residue_indices=np.asarray(payload["top_residue_indices"], dtype=int),
            q_residue=np.asarray(payload["q_residue"], dtype=float),
            D_edge=np.asarray(payload.get("D_edge") if payload.get("D_edge") is not None else np.zeros((0,), dtype=float), dtype=float),
            top_edge_indices=np.asarray(payload.get("top_edge_indices") if payload.get("top_edge_indices") is not None else np.zeros((0,), dtype=int), dtype=int),
            q_edge=np.asarray(payload.get("q_edge") if payload.get("q_edge") is not None else np.zeros((0, 0), dtype=float), dtype=float),
            ensemble_labels=np.asarray(payload.get("ensemble_labels") or [], dtype=str),
        )

        meta = {
            "analysis_id": analysis_id,
            "analysis_type": "delta_transition",
            "created_at": datetime.utcnow().isoformat(),
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
            "active_md_sample_id": active_id,
            "active_md_sample_name": payload.get("active_md_sample_name"),
            "inactive_md_sample_id": inactive_id,
            "inactive_md_sample_name": payload.get("inactive_md_sample_name"),
            "pas_md_sample_id": pas_id,
            "pas_md_sample_name": payload.get("pas_md_sample_name"),
            "model_a_id": model_a_id,
            "model_a_name": payload.get("model_a_name"),
            "model_b_id": model_b_id,
            "model_b_name": payload.get("model_b_name"),
            "drop_invalid": bool(not keep_invalid),
            "md_label_mode": md_label_mode,
            "band_fraction": float(band_fraction),
            "top_k_residues": int(np.asarray(payload["top_residue_indices"]).shape[0]),
            "top_k_edges": int(np.asarray(payload.get("top_edge_indices") if payload.get("top_edge_indices") is not None else []).shape[0]),
            "paths": {"analysis_npz": str(npz_path.relative_to(system_dir))},
            "summary": {
                "frames_active": int(np.asarray(payload["delta_energy_active"]).shape[0]),
                "frames_inactive": int(np.asarray(payload["delta_energy_inactive"]).shape[0]),
                "frames_pas": int(np.asarray(payload["delta_energy_pas"]).shape[0]),
                "residues": int(np.asarray(payload["D_residue"]).shape[0]),
                "edges": int(np.asarray(payload.get("D_edge") if payload.get("D_edge") is not None else []).shape[0]),
                "tau": float(payload["tau"]),
                "p_train": float(payload["p_train"]),
                "p_pas": float(payload["p_pas"]),
                "enrichment": float(payload["enrichment"]),
            },
        }
        meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "delta_transition",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(out_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[DeltaTransition {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Delta transition analysis completed", 100)
    return sanitized_payload


def run_delta_commitment_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Incremental delta-commitment analysis for a fixed (model A, model B) pair.

    This stores discriminative power once per (A,B,params) key and per-sample commitment
    for any selected samples (append/overwrite semantics).

    Results are written under clusters/<cluster_id>/analyses/delta_commitment/<analysis_id>/.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"delta-commitment-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("delta_commitment requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "delta_commitment",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[DeltaCommitment {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        model_a_id = str(params.get("model_a_id") or "").strip()
        model_b_id = str(params.get("model_b_id") or "").strip()
        if not model_a_id or not model_b_id:
            raise ValueError("delta_commitment requires model_a_id and model_b_id.")
        if model_a_id == model_b_id:
            raise ValueError("Select two different models.")

        sample_ids = params.get("sample_ids")
        if isinstance(sample_ids, str):
            sample_ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError("delta_commitment requires non-empty sample_ids.")

        md_label_mode = (params.get("md_label_mode") or "assigned").lower()
        keep_invalid = bool(params.get("keep_invalid", False))
        top_k_residues = int(params.get("top_k_residues", 20))
        top_k_edges = int(params.get("top_k_edges", 30))
        ranking_method = str(params.get("ranking_method") or "param_l2").strip()
        energy_bins = int(params.get("energy_bins", 80))

        save_progress("Computing commitment store...", 20)
        out = upsert_delta_commitment_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_a_ref=model_a_id,
            model_b_ref=model_b_id,
            sample_ids=sample_ids,
            md_label_mode=md_label_mode,
            drop_invalid=not keep_invalid,
            top_k_residues=top_k_residues,
            top_k_edges=top_k_edges,
            ranking_method=ranking_method,
            energy_bins=energy_bins,
        )

        meta = out.get("metadata") or {}
        analysis_id = str(meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("delta_commitment did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "delta_commitment",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[DeltaCommitment {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Delta commitment analysis completed", 100)
    return sanitized_payload


def run_endpoint_frustration_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Endpoint-local frustration analysis for a fixed (A,B) model pair.

    Results are written under clusters/<cluster_id>/analyses/endpoint_frustration/<analysis_id>/.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"endpoint-frustration-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("endpoint_frustration requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "endpoint_frustration",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[EndpointFrustration {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        model_a_id = str(params.get("model_a_id") or "").strip()
        model_b_id = str(params.get("model_b_id") or "").strip()
        if not model_a_id or not model_b_id:
            raise ValueError("endpoint_frustration requires model_a_id and model_b_id.")
        if model_a_id == model_b_id:
            raise ValueError("Select two different models.")

        sample_ids = params.get("sample_ids")
        if isinstance(sample_ids, str):
            sample_ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError("endpoint_frustration requires non-empty sample_ids.")

        md_label_mode = (params.get("md_label_mode") or "assigned").lower()
        keep_invalid = bool(params.get("keep_invalid", False))
        top_k_edges = int(params.get("top_k_edges", 2000))
        requested_workers = int(params.get("workers") or 0)

        def progress_cb(message: str, current: int, total: int):
            if total <= 0:
                return
            ratio = max(0.0, min(1.0, float(current) / float(total)))
            save_progress(message, 20 + int(70 * ratio))

        save_progress("Computing commitment and frustration summaries...", 20)
        out = upsert_endpoint_frustration_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_a_ref=model_a_id,
            model_b_ref=model_b_id,
            sample_ids=sample_ids,
            md_label_mode=md_label_mode,
            drop_invalid=not keep_invalid,
            top_k_edges=top_k_edges,
            n_workers=(requested_workers if requested_workers > 0 else None),
            progress_callback=progress_cb,
        )

        meta = out.get("metadata") or {}
        analysis_id = str(meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("endpoint_frustration did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "endpoint_frustration",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[EndpointFrustration {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Endpoint frustration analysis completed", 100)
    return sanitized_payload


def run_delta_energy_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Delta-energy distribution analysis for a fixed (A,B) model pair.

    Results are written under clusters/<cluster_id>/analyses/delta_energy/<analysis_id>/.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"delta-energy-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("delta_energy requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "delta_energy",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[DeltaEnergy {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        model_a_id = str(params.get("model_a_id") or "").strip()
        model_b_id = str(params.get("model_b_id") or "").strip()
        if not model_a_id or not model_b_id:
            raise ValueError("delta_energy requires model_a_id and model_b_id.")
        if model_a_id == model_b_id:
            raise ValueError("Select two different models.")

        sample_ids = params.get("sample_ids")
        if isinstance(sample_ids, str):
            sample_ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError("delta_energy requires non-empty sample_ids.")

        md_label_mode = (params.get("md_label_mode") or "assigned").lower()
        keep_invalid = bool(params.get("keep_invalid", False))
        frame_limits = params.get("frame_limits") if isinstance(params.get("frame_limits"), dict) else {}
        seed = int(params.get("seed") or 0)
        energy_bins = int(params.get("energy_bins") or 80)
        requested_workers = int(params.get("workers") or 0)

        def progress_cb(message: str, current: int, total: int):
            if total <= 0:
                return
            ratio = max(0.0, min(1.0, float(current) / float(total)))
            save_progress(message, 20 + int(70 * ratio))

        save_progress("Computing delta-energy distributions...", 20)
        out = upsert_delta_energy_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_a_ref=model_a_id,
            model_b_ref=model_b_id,
            sample_ids=sample_ids,
            md_label_mode=md_label_mode,
            drop_invalid=not keep_invalid,
            frame_limits={str(k): int(v) for k, v in (frame_limits or {}).items()},
            seed=seed,
            energy_bins=energy_bins,
            n_workers=(requested_workers if requested_workers > 0 else None),
            progress_callback=progress_cb,
        )

        meta = out.get("metadata") or {}
        analysis_id = str(meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("delta_energy did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "delta_energy",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[DeltaEnergy {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Delta-energy analysis completed", 100)
    return sanitized_payload



def run_hamiltonian_spectral_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """Incremental single-state and pair Hamiltonian spectral analysis."""
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"hamiltonian-spectral-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("hamiltonian_spectral requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"
    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "hamiltonian_spectral",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[HamiltonianSpectral {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        state_ids = params.get("state_ids")
        if isinstance(state_ids, str):
            state_ids = [s.strip() for s in state_ids.split(",") if s.strip()]
        if not isinstance(state_ids, list) or not state_ids:
            raise ValueError("hamiltonian_spectral requires non-empty state_ids.")
        top_k = int(params.get("top_k") or 20)
        overwrite = bool(params.get("overwrite", False))

        def progress_cb(message: str, current: int, total: int):
            if total <= 0:
                return
            ratio = max(0.0, min(1.0, float(current) / float(total)))
            base = 10 if "single" in message.lower() else 55
            span = 40 if "single" in message.lower() else 35
            save_progress(message, base + int(span * ratio))

        out = upsert_hamiltonian_spectral_batch(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            state_ids=state_ids,
            top_k=top_k,
            overwrite=overwrite,
            progress_callback=progress_cb,
        )

        result_payload["status"] = "finished"
        result_payload["results"] = out

    except Exception as e:
        print(f"[HamiltonianSpectral {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Hamiltonian spectral analysis completed", 100)
    return sanitized_payload


def run_spectral_intersection_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """Set-intersection between structural and functional spectral communities."""
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"spectral-intersection-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("spectral_intersection requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"
    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "spectral_intersection",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[SpectralIntersection {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)
        single_analysis_id = str(params.get("single_analysis_id") or "").strip()
        pair_analysis_id = str(params.get("pair_analysis_id") or "").strip()
        if not single_analysis_id or not pair_analysis_id:
            raise ValueError("spectral_intersection requires single_analysis_id and pair_analysis_id.")
        save_progress("Computing O(N) community intersection...", 40)
        out = upsert_spectral_intersection_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            single_analysis_id=single_analysis_id,
            pair_analysis_id=pair_analysis_id,
            min_group_size=int(params.get("min_group_size") or 3),
            overwrite=bool(params.get("overwrite", False)),
        )
        result_payload["status"] = "finished"
        result_payload["results"] = out

    except Exception as e:
        print(f"[SpectralIntersection {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Spectral intersection analysis completed", 100)
    return sanitized_payload


def run_piston_ligand_projection_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """Masked ligand/short-MD projection onto allosteric piston vectors."""
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"piston-ligand-projection-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("piston_ligand_projection requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"
    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "piston_ligand_projection",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[PistonLigandProjection {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)
        intersection_analysis_id = str(params.get("intersection_analysis_id") or "").strip()
        sample_ids = params.get("sample_ids") or []
        if isinstance(sample_ids, str):
            sample_ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        if not intersection_analysis_id or not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError("piston_ligand_projection requires intersection_analysis_id and non-empty sample_ids.")
        save_progress("Computing masked piston projections...", 35)
        out = upsert_piston_ligand_projection_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            intersection_analysis_id=intersection_analysis_id,
            sample_ids=sample_ids,
            label_mode=str(params.get("label_mode") or "assigned"),
            drop_invalid=bool(params.get("drop_invalid", True)),
            overwrite=bool(params.get("overwrite", False)),
        )
        result_payload["status"] = "finished"
        result_payload["results"] = out

    except Exception as e:
        print(f"[PistonLigandProjection {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Piston ligand projection completed", 100)
    return sanitized_payload

def run_delta_js_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Incremental delta-JS analysis for a fixed (model A, model B) pair.

    Stores A-vs-B-vs-Other JS distances (node/edge) and weighted trajectory-level distances.
    Potts models are optional (cluster topology mode).
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"delta-js-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("delta_js requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "delta_js",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[DeltaJS {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        model_a_id = str(params.get("model_a_id") or "").strip()
        model_b_id = str(params.get("model_b_id") or "").strip()
        using_models = bool(model_a_id or model_b_id)
        if using_models and (not model_a_id or not model_b_id):
            raise ValueError("Provide both model_a_id and model_b_id, or neither.")
        if using_models and model_a_id == model_b_id:
            raise ValueError("Select two different models.")

        sample_ids = params.get("sample_ids")
        if isinstance(sample_ids, str):
            sample_ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError("delta_js requires non-empty sample_ids.")

        ref_a = params.get("reference_sample_ids_a")
        if isinstance(ref_a, str):
            ref_a = [s.strip() for s in ref_a.split(",") if s.strip()]
        if ref_a is not None and not isinstance(ref_a, list):
            raise ValueError("reference_sample_ids_a must be a list when provided.")

        ref_b = params.get("reference_sample_ids_b")
        if isinstance(ref_b, str):
            ref_b = [s.strip() for s in ref_b.split(",") if s.strip()]
        if ref_b is not None and not isinstance(ref_b, list):
            raise ValueError("reference_sample_ids_b must be a list when provided.")
        if not using_models:
            if not ref_a or not ref_b:
                raise ValueError(
                    "reference_sample_ids_a and reference_sample_ids_b are required when no model pair is provided."
                )

        md_label_mode = (params.get("md_label_mode") or "assigned").lower()
        keep_invalid = bool(params.get("keep_invalid", False))
        top_k_residues = int(params.get("top_k_residues", 20))
        top_k_edges = int(params.get("top_k_edges", 30))
        ranking_method = str(params.get("ranking_method") or "js_ab").strip()
        edge_mode = str(params.get("edge_mode") or "").strip().lower()
        if edge_mode and edge_mode not in {"cluster", "all_vs_all", "contact"}:
            raise ValueError("edge_mode must be one of: cluster, all_vs_all, contact.")
        if not using_models and not edge_mode:
            raise ValueError("edge_mode is required when no model pair is provided.")

        contact_state_ids = params.get("contact_state_ids")
        if isinstance(contact_state_ids, str):
            contact_state_ids = [s.strip() for s in contact_state_ids.split(",") if s.strip()]
        if contact_state_ids is not None and not isinstance(contact_state_ids, list):
            raise ValueError("contact_state_ids must be a list when provided.")

        contact_pdbs = params.get("contact_pdbs")
        if isinstance(contact_pdbs, str):
            contact_pdbs = [s.strip() for s in contact_pdbs.split(",") if s.strip()]
        if contact_pdbs is not None and not isinstance(contact_pdbs, list):
            raise ValueError("contact_pdbs must be a list when provided.")

        contact_cutoff = float(params.get("contact_cutoff", 10.0))
        contact_atom_mode = str(params.get("contact_atom_mode") or "CA").strip().upper()
        if edge_mode == "contact":
            if not (contact_state_ids or contact_pdbs):
                raise ValueError("edge_mode=contact requires contact_state_ids and/or contact_pdbs.")
            if not np.isfinite(contact_cutoff) or contact_cutoff <= 0:
                raise ValueError("contact_cutoff must be > 0.")
            if contact_atom_mode not in {"CA", "CM"}:
                raise ValueError("contact_atom_mode must be 'CA' or 'CM'.")

        save_progress("Computing JS A/B/Other store...", 20)
        out = upsert_delta_js_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_a_ref=(model_a_id or None),
            model_b_ref=(model_b_id or None),
            sample_ids=sample_ids,
            reference_sample_ids_a=ref_a,
            reference_sample_ids_b=ref_b,
            md_label_mode=md_label_mode,
            drop_invalid=not keep_invalid,
            top_k_residues=top_k_residues,
            top_k_edges=top_k_edges,
            ranking_method=ranking_method,
            edge_mode=(edge_mode or None),
            contact_state_ids=contact_state_ids,
            contact_pdbs=contact_pdbs,
            contact_cutoff=contact_cutoff,
            contact_atom_mode=contact_atom_mode,
        )

        meta = out.get("metadata") or {}
        analysis_id = str(meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("delta_js did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "delta_js",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[DeltaJS {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Delta JS analysis completed", 100)
    return sanitized_payload


def run_transient_states_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Low-occupancy trajectory-enriched node/edge cluster-state analysis.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"transient-states-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("transient_states requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"
    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "transient_states",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[TransientStates {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        sample_ids = params.get("sample_ids")
        if isinstance(sample_ids, str):
            sample_ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        if not isinstance(sample_ids, list) or len(sample_ids) < 2:
            raise ValueError("transient_states requires at least two sample_ids.")

        def progress_cb(message: str, current: int, total: int):
            total_i = max(1, int(total))
            ratio = max(0.0, min(1.0, float(current) / float(total_i)))
            save_progress(message, 10 + int(80 * ratio))

        out = run_transient_state_analysis(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            sample_ids=sample_ids,
            md_label_mode=str(params.get("md_label_mode") or "assigned"),
            drop_invalid=not bool(params.get("keep_invalid", False)),
            p_min=float(params.get("p_min", 0.005)),
            p_max=float(params.get("p_max", 0.05)),
            enrichment_min=float(params.get("enrichment_min", 1.0)),
            epsilon=float(params.get("epsilon", 1e-9)),
            top_k_nodes=int(params.get("top_k_nodes", 500)),
            include_edges=bool(params.get("include_edges", True)),
            edge_mode=str(params.get("edge_mode") or "cluster"),
            edge_p_min=params.get("edge_p_min"),
            edge_p_max=params.get("edge_p_max"),
            edge_enrichment_min=params.get("edge_enrichment_min"),
            delta_pmi_min=params.get("delta_pmi_min"),
            top_k_edges=int(params.get("top_k_edges", 1000)),
            progress_callback=progress_cb,
        )

        meta = out.get("metadata") or {}
        analysis_id = str(meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("transient_states did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "transient_states",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
            "summary": meta.get("summary") or {},
        }

    except Exception as e:
        print(f"[TransientStates {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Transient-state analysis completed", 100)
    return sanitized_payload


def _gibbs_relaxation_batch_keys(analysis_id: str) -> Dict[str, str]:
    return _analysis_batch_keys("gibbs_relaxation", analysis_id)


def run_gibbs_relaxation_frame_job(
    analysis_id: str,
    analysis_dir: str,
    row: int,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    queue_name = getattr(job, "origin", "phase-jobs")
    keys = _gibbs_relaxation_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
    payloads = prepared.get("payloads") or []
    row = int(row)
    if row < 0 or row >= len(payloads):
        raise IndexError(f"Invalid Gibbs relaxation row index: {row}")
    try:
        out = _gibbs_relax_worker(payloads[row])
        write_partial_result(analysis_dir_path, row, out)
    except Exception as exc:
        write_partial_error(analysis_dir_path, row, f"{type(exc).__name__}: {exc}")
        if redis_conn is not None and redis_conn.get(keys["child_error"]) is None:
            redis_conn.set(
                keys["child_error"],
                f"row {row}: {type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
        raise
    finally:
        if redis_conn is not None:
            remaining = int(redis_conn.decr(keys["remaining"]))
            if remaining <= 0:
                redis_conn.set(keys["state"], "aggregating", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
                Queue(queue_name, connection=redis_conn).enqueue(
                    run_gibbs_relaxation_aggregate_job,
                    args=(str(analysis_id), str(analysis_dir_path), int(max(1, workers_used))),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"gibbs-relaxation-aggregate-{analysis_id}",
                )


def run_gibbs_relaxation_aggregate_job(
    analysis_id: str,
    analysis_dir: str,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    keys = _gibbs_relaxation_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    try:
        prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
        errors = load_partial_errors(analysis_dir_path)
        if errors:
            summary = "; ".join(
                f"row {err.get('row')}: {err.get('error')}" for err in errors[:3]
            )
            if len(errors) > 3:
                summary = f"{summary}; ... ({len(errors)} worker failures total)"
            raise RuntimeError(f"Distributed Gibbs relaxation worker failures: {summary}")
        out_rows = load_partial_results(
            analysis_dir_path,
            expected_rows=len(prepared.get("payloads") or []),
        )
        result = aggregate_gibbs_relaxation_batch(
            prepared,
            out_rows,
            workers_used=max(1, int(workers_used)),
        )
        if redis_conn is not None:
            redis_conn.delete(keys["error"])
            redis_conn.set(keys["state"], "finished", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        return result
    except Exception as exc:
        if redis_conn is not None:
            redis_conn.set(
                keys["error"],
                f"{type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
            redis_conn.set(keys["state"], "failed", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        raise


def run_gibbs_relaxation_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Gibbs relaxation analysis:
      - choose random starting frames from one sample
      - run Gibbs trajectories under a selected Potts model
      - store first-flip percentile statistics for visualization
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"gibbs-relaxation-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("gibbs_relaxation requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "gibbs_relaxation",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[GibbsRelaxation {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        start_sample_id = str(params.get("start_sample_id") or "").strip()
        if not start_sample_id:
            raise ValueError("gibbs_relaxation requires start_sample_id.")

        model_ref = str(params.get("model_id") or "").strip()
        if not model_ref:
            model_ref = str(params.get("model_path") or "").strip()
        if not model_ref:
            raise ValueError("gibbs_relaxation requires model_id or model_path.")

        beta = float(params.get("beta", 1.0))
        n_start_frames = int(params.get("n_start_frames", 100))
        gibbs_sweeps = int(params.get("gibbs_sweeps", 1000))
        seed = int(params.get("seed", 0))
        workers = params.get("workers")
        workers_val = int(workers) if workers is not None else None
        start_label_mode = str(params.get("start_label_mode") or "assigned").strip().lower()
        keep_invalid = bool(params.get("keep_invalid", False))

        save_progress("Preparing Gibbs relaxation analysis...", 10)
        prepared = prepare_gibbs_relaxation_batch(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            start_sample_id=start_sample_id,
            model_ref=model_ref,
            beta=beta,
            n_start_frames=n_start_frames,
            gibbs_sweeps=gibbs_sweeps,
            seed=seed,
            start_label_mode=start_label_mode,
            drop_invalid=not keep_invalid,
            n_workers=workers_val,
        )
        payloads = prepared.get("payloads") or []
        n_payloads = len(payloads)
        analysis_id = str(prepared.get("analysis_id") or "")
        analysis_dir = Path(str(prepared.get("analysis_dir") or "")).resolve()
        redis_conn = getattr(job, "connection", None)
        queue_name = getattr(job, "origin", "phase-jobs")
        queue = Queue(queue_name, connection=redis_conn) if redis_conn is not None else None
        available_queue_workers = _count_queue_workers(queue) if queue is not None else 0
        distributed_parallelism = _distributed_batch_parallelism(
            requested_workers=int(prepared.get("requested_workers", workers_val or 1)),
            available_queue_workers=available_queue_workers,
            n_payloads=n_payloads,
        )

        if distributed_parallelism > 1 and queue is not None:
            keys = _gibbs_relaxation_batch_keys(analysis_id)
            redis_conn.delete(keys["done"], keys["error"], keys["child_error"])
            redis_conn.set(keys["remaining"], int(n_payloads), ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "running", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            save_progress(
                f"Dispatching {n_payloads} relaxation tasks across {distributed_parallelism} queue workers...",
                15,
            )
            for row in range(n_payloads):
                queue.enqueue(
                    run_gibbs_relaxation_frame_job,
                    args=(analysis_id, str(analysis_dir), int(row), int(distributed_parallelism)),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"gibbs-relaxation-frame-{analysis_id}-{row:06d}",
                )
            while True:
                done = _decode_redis_value(redis_conn.get(keys["done"]))
                remaining_raw = _decode_redis_value(redis_conn.get(keys["remaining"]))
                remaining = int(remaining_raw) if remaining_raw is not None else n_payloads
                state = _decode_redis_value(redis_conn.get(keys["state"])) or "running"
                child_error = _decode_redis_value(redis_conn.get(keys["child_error"]))
                if done:
                    break
                if remaining <= 0:
                    save_progress("Aggregating distributed Gibbs relaxation results...", 92)
                else:
                    completed = max(0, min(n_payloads, n_payloads - remaining))
                    pct = 15 + int(70 * (float(completed) / float(max(1, n_payloads))))
                    status = f"Running distributed Gibbs relaxation ({completed}/{n_payloads} frames)"
                    if child_error:
                        status = f"{status}; worker error recorded, waiting for aggregation"
                    elif state and state != "running":
                        status = f"{status}; state={state}"
                    save_progress(status, pct)
                time.sleep(1.0)
            error = _decode_redis_value(redis_conn.get(keys["error"]))
            if error:
                raise RuntimeError(error)
            out = pickle_load(orchestration_paths(analysis_dir)["aggregate"])
        else:
            save_progress("Running Gibbs relaxation analysis locally...", 15)

            def progress_cb(message: str, current: int, total: int):
                if total <= 0:
                    return
                ratio = max(0.0, min(1.0, float(current) / float(total)))
                pct = 15 + int(70 * ratio)
                save_progress(message, pct)

            out_rows = run_local_payload_batch(
                payloads,
                worker_fn=_gibbs_relax_worker,
                max_workers=1,
                progress_callback=progress_cb,
                progress_label="Running Gibbs relaxations",
            )
            out = aggregate_gibbs_relaxation_batch(prepared, out_rows, workers_used=1)

        meta = out.get("metadata") or {}
        analysis_id = str(meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("gibbs_relaxation did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "gibbs_relaxation",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[GibbsRelaxation {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Gibbs relaxation analysis completed", 100)
    return sanitized_payload


_LIGAND_COMPLETION_BATCH_TTL_SECONDS = 86400


def _analysis_batch_keys(prefix: str, analysis_id: str) -> Dict[str, str]:
    base = f"{prefix}:{analysis_id}"
    return {
        "remaining": f"{base}:remaining",
        "done": f"{base}:done",
        "error": f"{base}:error",
        "state": f"{base}:state",
        "child_error": f"{base}:child_error",
    }


def _ligand_completion_batch_keys(analysis_id: str) -> Dict[str, str]:
    return _analysis_batch_keys("ligand_completion", analysis_id)


def _decode_redis_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _count_queue_workers(queue: Queue) -> int:
    try:
        workers = Worker.all(connection=queue.connection)
    except Exception:
        return 0
    total = 0
    for worker in workers:
        try:
            queue_names = set(worker.queue_names())
        except Exception:
            queue_names = set()
        if queue.name in queue_names:
            total += 1
    return total


def _distributed_batch_parallelism(
    *,
    requested_workers: int,
    available_queue_workers: int,
    n_payloads: int,
) -> int:
    if int(n_payloads) <= 1:
        return 0
    usable_workers = max(0, int(available_queue_workers) - 1)
    if usable_workers <= 0:
        return 0
    if int(requested_workers) > 0:
        usable_workers = min(usable_workers, int(requested_workers))
    return min(int(n_payloads), usable_workers)


def _simulation_batch_keys(job_uuid: str) -> Dict[str, str]:
    return _analysis_batch_keys("simulation", job_uuid)


def run_simulation_chain_job(
    job_uuid: str,
    sample_dir: str,
    row: int,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    queue_name = getattr(job, "origin", "phase-jobs")
    keys = _simulation_batch_keys(str(job_uuid))
    sample_dir_path = Path(str(sample_dir))
    prepared = pickle_load(orchestration_paths(sample_dir_path)["prepared"])
    payloads = prepared.get("payloads") or []
    row = int(row)
    if row < 0 or row >= len(payloads):
        raise IndexError(f"Invalid simulation chain row index: {row}")
    try:
        out = run_sampling_chain_payload(payloads[row])
        write_partial_result(sample_dir_path, row, out)
    except Exception as exc:
        write_partial_error(sample_dir_path, row, f"{type(exc).__name__}: {exc}")
        if redis_conn is not None and redis_conn.get(keys["child_error"]) is None:
            redis_conn.set(
                keys["child_error"],
                f"row {row}: {type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
        raise
    finally:
        if redis_conn is not None:
            remaining = int(redis_conn.decr(keys["remaining"]))
            if remaining <= 0:
                redis_conn.set(keys["state"], "aggregating", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
                Queue(queue_name, connection=redis_conn).enqueue(
                    run_simulation_aggregate_job,
                    args=(str(job_uuid), str(sample_dir_path), int(max(1, workers_used))),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"simulation-aggregate-{job_uuid}",
                )


def run_simulation_aggregate_job(
    job_uuid: str,
    sample_dir: str,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    keys = _simulation_batch_keys(str(job_uuid))
    sample_dir_path = Path(str(sample_dir))
    try:
        prepared = pickle_load(orchestration_paths(sample_dir_path)["prepared"])
        errors = load_partial_errors(sample_dir_path)
        if errors:
            summary = "; ".join(f"row {err.get('row')}: {err.get('error')}" for err in errors[:3])
            if len(errors) > 3:
                summary = f"{summary}; ... ({len(errors)} worker failures total)"
            raise RuntimeError(f"Distributed simulation worker failures: {summary}")
        out_rows = load_partial_results(sample_dir_path, expected_rows=len(prepared.get("payloads") or []))
        result = aggregate_sampling_batch(prepared, out_rows, workers_used=max(1, int(workers_used)))
        if redis_conn is not None:
            redis_conn.delete(keys["error"])
            redis_conn.set(keys["state"], "finished", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        return result
    except Exception as exc:
        if redis_conn is not None:
            redis_conn.set(keys["error"], f"{type(exc).__name__}: {exc}", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "failed", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        raise


def _lambda_sweep_batch_keys(analysis_id: str) -> Dict[str, str]:
    return _analysis_batch_keys("lambda_sweep", analysis_id)


def _potts_analysis_batch_keys(analysis_id: str) -> Dict[str, str]:
    return _analysis_batch_keys("potts_analysis", analysis_id)


def run_potts_analysis_payload_job(
    analysis_id: str,
    analysis_dir: str,
    row: int,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    queue_name = getattr(job, "origin", "phase-jobs")
    keys = _potts_analysis_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
    payloads = prepared.get("payloads") or []
    row = int(row)
    if row < 0 or row >= len(payloads):
        raise IndexError(f"Invalid Potts analysis row index: {row}")
    try:
        out = _potts_analysis_payload_worker(payloads[row])
        write_partial_result(analysis_dir_path, row, out)
    except Exception as exc:
        write_partial_error(analysis_dir_path, row, f"{type(exc).__name__}: {exc}")
        if redis_conn is not None and redis_conn.get(keys["child_error"]) is None:
            redis_conn.set(
                keys["child_error"],
                f"row {row}: {type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
        raise
    finally:
        if redis_conn is not None:
            remaining = int(redis_conn.decr(keys["remaining"]))
            if remaining <= 0:
                redis_conn.set(keys["state"], "aggregating", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
                Queue(queue_name, connection=redis_conn).enqueue(
                    run_potts_analysis_aggregate_job,
                    args=(str(analysis_id), str(analysis_dir_path), int(max(1, workers_used))),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"potts-analysis-aggregate-{analysis_id}",
                )


def run_potts_analysis_aggregate_job(
    analysis_id: str,
    analysis_dir: str,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    keys = _potts_analysis_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    try:
        prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
        errors = load_partial_errors(analysis_dir_path)
        if errors:
            summary = "; ".join(f"row {err.get('row')}: {err.get('error')}" for err in errors[:3])
            if len(errors) > 3:
                summary = f"{summary}; ... ({len(errors)} worker failures total)"
            raise RuntimeError(f"Distributed Potts analysis worker failures: {summary}")
        out_rows = load_partial_results(analysis_dir_path, expected_rows=len(prepared.get("payloads") or []))
        result = aggregate_potts_analysis_batch(prepared, out_rows, workers_used=max(1, int(workers_used)))
        if redis_conn is not None:
            redis_conn.delete(keys["error"])
            redis_conn.set(keys["state"], "finished", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        return result
    except Exception as exc:
        if redis_conn is not None:
            redis_conn.set(keys["error"], f"{type(exc).__name__}: {exc}", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "failed", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        raise


def run_lambda_sweep_payload_job(
    analysis_id: str,
    analysis_dir: str,
    row: int,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    queue_name = getattr(job, "origin", "phase-jobs")
    keys = _lambda_sweep_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
    payloads = prepared.get("payloads") or []
    row = int(row)
    if row < 0 or row >= len(payloads):
        raise IndexError(f"Invalid lambda sweep row index: {row}")
    try:
        out = run_lambda_sweep_payload(payloads[row])
        write_partial_result(analysis_dir_path, row, out)
    except Exception as exc:
        write_partial_error(analysis_dir_path, row, f"{type(exc).__name__}: {exc}")
        if redis_conn is not None and redis_conn.get(keys["child_error"]) is None:
            redis_conn.set(
                keys["child_error"],
                f"row {row}: {type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
        raise
    finally:
        if redis_conn is not None:
            remaining = int(redis_conn.decr(keys["remaining"]))
            if remaining <= 0:
                redis_conn.set(keys["state"], "aggregating", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
                Queue(queue_name, connection=redis_conn).enqueue(
                    run_lambda_sweep_aggregate_job,
                    args=(str(analysis_id), str(analysis_dir_path), int(max(1, workers_used))),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"lambda-sweep-aggregate-{analysis_id}",
                )


def run_lambda_sweep_aggregate_job(
    analysis_id: str,
    analysis_dir: str,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    keys = _lambda_sweep_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    try:
        prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
        errors = load_partial_errors(analysis_dir_path)
        if errors:
            summary = "; ".join(f"row {err.get('row')}: {err.get('error')}" for err in errors[:3])
            if len(errors) > 3:
                summary = f"{summary}; ... ({len(errors)} worker failures total)"
            raise RuntimeError(f"Distributed lambda sweep worker failures: {summary}")
        out_rows = load_partial_results(analysis_dir_path, expected_rows=len(prepared.get("payloads") or []))
        result = aggregate_lambda_sweep_batch(prepared, out_rows, workers_used=max(1, int(workers_used)))
        if redis_conn is not None:
            redis_conn.delete(keys["error"])
            redis_conn.set(keys["state"], "finished", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        return result
    except Exception as exc:
        if redis_conn is not None:
            redis_conn.set(keys["error"], f"{type(exc).__name__}: {exc}", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "failed", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        raise


def _potts_nn_mapping_batch_keys(analysis_id: str) -> Dict[str, str]:
    return _analysis_batch_keys("potts_nn_mapping", analysis_id)


def run_potts_nn_mapping_payload_job(
    analysis_id: str,
    analysis_dir: str,
    row: int,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    queue_name = getattr(job, "origin", "phase-jobs")
    keys = _potts_nn_mapping_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
    payloads = prepared.get("payloads") or []
    row = int(row)
    if row < 0 or row >= len(payloads):
        raise IndexError(f"Invalid potts NN mapping row index: {row}")
    try:
        out = _potts_nn_row_worker(payloads[row])
        write_partial_result(analysis_dir_path, row, out)
    except Exception as exc:
        write_partial_error(analysis_dir_path, row, f"{type(exc).__name__}: {exc}")
        if redis_conn is not None and redis_conn.get(keys["child_error"]) is None:
            redis_conn.set(
                keys["child_error"],
                f"row {row}: {type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
        raise
    finally:
        if redis_conn is not None:
            remaining = int(redis_conn.decr(keys["remaining"]))
            if remaining <= 0:
                redis_conn.set(keys["state"], "aggregating", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
                Queue(queue_name, connection=redis_conn).enqueue(
                    run_potts_nn_mapping_aggregate_job,
                    args=(str(analysis_id), str(analysis_dir_path), int(max(1, workers_used))),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"potts-nn-mapping-aggregate-{analysis_id}",
                )


def run_potts_nn_mapping_aggregate_job(
    analysis_id: str,
    analysis_dir: str,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    keys = _potts_nn_mapping_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    try:
        prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
        errors = load_partial_errors(analysis_dir_path)
        if errors:
            summary = "; ".join(f"row {err.get('row')}: {err.get('error')}" for err in errors[:3])
            if len(errors) > 3:
                summary = f"{summary}; ... ({len(errors)} worker failures total)"
            raise RuntimeError(f"Distributed potts NN mapping worker failures: {summary}")
        out_rows = load_partial_results(analysis_dir_path, expected_rows=len(prepared.get("payloads") or []))
        result = aggregate_potts_nn_mapping_batch(prepared, out_rows, workers_used=max(1, int(workers_used)))
        if redis_conn is not None:
            redis_conn.delete(keys["error"])
            redis_conn.set(keys["state"], "finished", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        return result
    except Exception as exc:
        if redis_conn is not None:
            redis_conn.set(keys["error"], f"{type(exc).__name__}: {exc}", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "failed", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        raise


def run_ligand_completion_frame_job(
    analysis_id: str,
    analysis_dir: str,
    row: int,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    queue_name = getattr(job, "origin", "phase-jobs")
    keys = _ligand_completion_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
    payloads = prepared.get("payloads") or []
    row = int(row)
    if row < 0 or row >= len(payloads):
        raise IndexError(f"Invalid ligand completion row index: {row}")
    try:
        out = _single_frame_ligand_completion_worker(payloads[row])
        write_partial_result(analysis_dir_path, row, out)
    except Exception as exc:
        write_partial_error(analysis_dir_path, row, f"{type(exc).__name__}: {exc}")
        if redis_conn is not None and redis_conn.get(keys["child_error"]) is None:
            redis_conn.set(
                keys["child_error"],
                f"row {row}: {type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
        raise
    finally:
        if redis_conn is not None:
            remaining = int(redis_conn.decr(keys["remaining"]))
            if remaining <= 0:
                redis_conn.set(
                    keys["state"],
                    "aggregating",
                    ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
                )
                Queue(queue_name, connection=redis_conn).enqueue(
                    run_ligand_completion_aggregate_job,
                    args=(str(analysis_id), str(analysis_dir_path), int(max(1, workers_used))),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"ligand-completion-aggregate-{analysis_id}",
                )


def run_ligand_completion_aggregate_job(
    analysis_id: str,
    analysis_dir: str,
    workers_used: int,
):
    job = get_current_job()
    redis_conn = getattr(job, "connection", None)
    keys = _ligand_completion_batch_keys(str(analysis_id))
    analysis_dir_path = Path(str(analysis_dir))
    try:
        prepared = pickle_load(orchestration_paths(analysis_dir_path)["prepared"])
        errors = load_partial_errors(analysis_dir_path)
        if errors:
            summary = "; ".join(
                f"row {err.get('row')}: {err.get('error')}" for err in errors[:3]
            )
            if len(errors) > 3:
                summary = f"{summary}; ... ({len(errors)} worker failures total)"
            raise RuntimeError(f"Distributed ligand completion worker failures: {summary}")
        out_rows = load_partial_results(
            analysis_dir_path,
            expected_rows=len(prepared.get("payloads") or []),
        )
        result = aggregate_ligand_completion_batch(
            prepared,
            out_rows,
            workers_used=max(1, int(workers_used)),
        )
        if redis_conn is not None:
            redis_conn.delete(keys["error"])
            redis_conn.set(keys["state"], "finished", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        return result
    except Exception as exc:
        if redis_conn is not None:
            redis_conn.set(
                keys["error"],
                f"{type(exc).__name__}: {exc}",
                ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS,
            )
            redis_conn.set(keys["state"], "failed", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["done"], "1", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
        raise


def run_ligand_completion_job(
    job_uuid: str,
    dataset_ref: Dict[str, str],
    params: Dict[str, Any],
):
    """
    Ligand-guided conditional completion analysis with endpoint models A/B.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"ligand-completion-{job_uuid}"

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    cluster_id = dataset_ref.get("cluster_id")
    if not project_id or not system_id or not cluster_id:
        raise ValueError("ligand_completion requires project_id/system_id/cluster_id.")

    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    result_payload: Dict[str, Any] = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id,
        "analysis_type": "ligand_completion",
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "results": None,
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
        },
        "error": None,
        "completed_at": None,
    }

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[LigandCompletion {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        try:
            with open(result_filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        model_a_ref = str(params.get("model_a_id") or params.get("model_a_path") or "").strip()
        model_b_ref = str(params.get("model_b_id") or params.get("model_b_path") or "").strip()
        if not model_a_ref or not model_b_ref:
            raise ValueError("ligand_completion requires model_a_id/model_b_id (or paths).")
        if model_a_ref == model_b_ref:
            raise ValueError("model_a and model_b must be different.")

        constrained_residues = params.get("constrained_residues")
        if isinstance(constrained_residues, str):
            constrained_residues = [s.strip() for s in constrained_residues.split(",") if s.strip()]
        constraint_mode = str(params.get("constraint_source_mode") or "manual").strip().lower()
        if constraint_mode not in {"manual", "delta_js_auto"}:
            raise ValueError("constraint_source_mode must be one of: manual, delta_js_auto.")
        if constraint_mode == "manual":
            if not isinstance(constrained_residues, list) or not constrained_residues:
                raise ValueError("constrained_residues must be a non-empty list for manual constraint mode.")
        elif not isinstance(constrained_residues, list):
            constrained_residues = []

        requested_workers = int(params.get("workers") or 0)
        analysis_kwargs = dict(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_a_ref=model_a_ref,
            model_b_ref=model_b_ref,
            md_sample_id=str(params.get("md_sample_id") or "").strip(),
            constrained_residues=constrained_residues,
            reference_sample_id_a=str(params.get("reference_sample_id_a") or "").strip() or None,
            reference_sample_id_b=str(params.get("reference_sample_id_b") or "").strip() or None,
            sampler=str(params.get("sampler") or "sa"),
            lambda_values=params.get("lambda_values") or [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            n_start_frames=int(params.get("n_start_frames", 100)),
            n_samples_per_frame=int(params.get("n_samples_per_frame", 100)),
            n_steps=int(params.get("n_steps", 1000)),
            tail_steps=int(params.get("tail_steps", 200)),
            target_window_size=int(params.get("target_window_size", 11)),
            target_pseudocount=float(params.get("target_pseudocount", 1e-3)),
            epsilon_logpenalty=float(params.get("epsilon_logpenalty", 1e-8)),
            constraint_weight_mode=str(params.get("constraint_weight_mode") or "uniform"),
            constraint_weights=params.get("constraint_weights"),
            constraint_weight_min=float(params.get("constraint_weight_min", 0.0)),
            constraint_weight_max=float(params.get("constraint_weight_max", 1.0)),
            constraint_source_mode=str(params.get("constraint_source_mode") or "manual"),
            constraint_delta_js_analysis_id=(
                str(params.get("constraint_delta_js_analysis_id") or "").strip() or None
            ),
            constraint_delta_js_sample_id=(
                str(params.get("constraint_delta_js_sample_id") or "").strip() or None
            ),
            constraint_auto_top_k=int(params.get("constraint_auto_top_k", 12)),
            constraint_auto_edge_alpha=float(params.get("constraint_auto_edge_alpha", 0.3)),
            constraint_auto_exclude_success=bool(params.get("constraint_auto_exclude_success", True)),
            gibbs_beta=float(params.get("gibbs_beta", 1.0)),
            sa_beta_hot=float(params.get("sa_beta_hot", 0.8)),
            sa_beta_cold=float(params.get("sa_beta_cold", 50.0)),
            sa_schedule=str(params.get("sa_schedule") or "geom"),
            md_label_mode=str(params.get("md_label_mode") or "assigned"),
            drop_invalid=not bool(params.get("keep_invalid", False)),
            success_metric_mode=str(params.get("success_metric_mode") or "deltae"),
            delta_js_experiment_id=(str(params.get("delta_js_experiment_id") or "").strip() or None),
            delta_js_analysis_id=(str(params.get("delta_js_analysis_id") or "").strip() or None),
            delta_js_filter_setup_id=(str(params.get("delta_js_filter_setup_id") or "").strip() or None),
            delta_js_filter_edge_alpha=float(params.get("delta_js_filter_edge_alpha", 0.75)),
            delta_js_d_residue_min=float(params.get("delta_js_d_residue_min", 0.0)),
            delta_js_d_residue_max=(
                float(params.get("delta_js_d_residue_max"))
                if params.get("delta_js_d_residue_max") is not None
                else None
            ),
            delta_js_d_edge_min=float(params.get("delta_js_d_edge_min", 0.0)),
            delta_js_d_edge_max=(
                float(params.get("delta_js_d_edge_max"))
                if params.get("delta_js_d_edge_max") is not None
                else None
            ),
            delta_js_node_edge_alpha=(
                float(params.get("delta_js_node_edge_alpha"))
                if params.get("delta_js_node_edge_alpha") is not None
                else None
            ),
            js_success_threshold=float(params.get("js_success_threshold", 0.15)),
            js_success_margin=float(params.get("js_success_margin", 0.02)),
            deltae_margin=float(params.get("deltae_margin", 0.0)),
            completion_target_success=float(params.get("completion_target_success", 0.7)),
            completion_cost_if_unreached=(
                float(params.get("completion_cost_if_unreached"))
                if params.get("completion_cost_if_unreached") is not None
                else None
            ),
            n_workers=max(1, requested_workers or 1),
            seed=int(params.get("seed", 0)),
        )
        save_progress("Preparing ligand completion analysis...", 10)
        prepared = prepare_ligand_completion_batch(**analysis_kwargs)
        payloads = prepared.get("payloads") or []
        n_payloads = len(payloads)
        analysis_id = str(prepared.get("analysis_id") or "")
        analysis_dir = Path(str(prepared.get("analysis_dir") or "")).resolve()
        redis_conn = getattr(job, "connection", None)
        queue_name = getattr(job, "origin", "phase-jobs")
        queue = Queue(queue_name, connection=redis_conn) if redis_conn is not None else None
        available_queue_workers = _count_queue_workers(queue) if queue is not None else 0
        distributed_parallelism = _distributed_batch_parallelism(
            requested_workers=requested_workers,
            available_queue_workers=available_queue_workers,
            n_payloads=n_payloads,
        )

        if distributed_parallelism > 1 and queue is not None:
            keys = _ligand_completion_batch_keys(analysis_id)
            redis_conn.delete(keys["done"], keys["error"], keys["child_error"])
            redis_conn.set(keys["remaining"], int(n_payloads), ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            redis_conn.set(keys["state"], "running", ex=_LIGAND_COMPLETION_BATCH_TTL_SECONDS)
            save_progress(
                f"Dispatching {n_payloads} frame tasks across {distributed_parallelism} queue workers...",
                15,
            )
            for row in range(n_payloads):
                queue.enqueue(
                    run_ligand_completion_frame_job,
                    args=(analysis_id, str(analysis_dir), int(row), int(distributed_parallelism)),
                    job_timeout="8h",
                    result_ttl=86400,
                    job_id=f"ligand-completion-frame-{analysis_id}-{row:06d}",
                )
            while True:
                done = _decode_redis_value(redis_conn.get(keys["done"]))
                remaining_raw = _decode_redis_value(redis_conn.get(keys["remaining"]))
                remaining = int(remaining_raw) if remaining_raw is not None else n_payloads
                state = _decode_redis_value(redis_conn.get(keys["state"])) or "running"
                child_error = _decode_redis_value(redis_conn.get(keys["child_error"]))
                if done:
                    break
                if remaining <= 0:
                    save_progress("Aggregating distributed ligand completion results...", 92)
                else:
                    completed = max(0, min(n_payloads, n_payloads - remaining))
                    pct = 15 + int(70 * (float(completed) / float(max(1, n_payloads))))
                    status = f"Running distributed ligand completion ({completed}/{n_payloads} frames)"
                    if child_error:
                        status = f"{status}; worker error recorded, waiting for aggregation"
                    elif state and state != "running":
                        status = f"{status}; state={state}"
                    save_progress(status, pct)
                time.sleep(1.0)
            error = _decode_redis_value(redis_conn.get(keys["error"]))
            if error:
                raise RuntimeError(error)
            out = pickle_load(orchestration_paths(analysis_dir)["aggregate"])
        else:
            save_progress("Running ligand completion analysis locally...", 15)

            def progress_cb(message: str, current: int, total: int):
                if total <= 0:
                    return
                ratio = max(0.0, min(1.0, float(current) / float(total)))
                pct = 15 + int(70 * ratio)
                save_progress(message, pct)

            out_rows = run_local_payload_batch(
                payloads,
                worker_fn=_single_frame_ligand_completion_worker,
                max_workers=1,
                progress_callback=progress_cb,
                progress_label="Running conditional completion",
            )
            out = aggregate_ligand_completion_batch(prepared, out_rows, workers_used=1)

        meta = out.get("metadata") or {}
        analysis_id = str(meta.get("analysis_id") or "")
        analysis_dir = Path(str(out.get("analysis_dir") or "")).resolve()
        npz_path = Path(str(out.get("analysis_npz") or "")).resolve()
        if not analysis_id or not analysis_dir.exists() or not npz_path.exists():
            raise RuntimeError("ligand_completion did not write analysis artifacts as expected.")

        result_payload["status"] = "finished"
        result_payload["results"] = {
            "analysis_type": "ligand_completion",
            "analysis_id": analysis_id,
            "analysis_dir": _relativize_path(analysis_dir),
            "analysis_npz": _relativize_path(npz_path),
        }

    except Exception as e:
        print(f"[LigandCompletion {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        raise e

    finally:
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        sanitized_payload = _convert_nan_to_none(result_payload)
        write_result_to_disk(sanitized_payload)

    save_progress("Ligand completion analysis completed", 100)
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

    project_id = dataset_ref.get("project_id")
    system_id = dataset_ref.get("system_id")
    if not project_id or not system_id:
        raise ValueError("Potts fit requires project_id/system_id.")
    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_filepath = results_dirs["jobs_dir"] / f"{job_uuid}.json"

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

        cluster_id = dataset_ref.get("cluster_id")
        if not cluster_id:
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

        fit_params = dict(params or {})
        model_name = fit_params.get("model_name")
        fit_mode = fit_params.get("fit_mode")
        if not fit_mode:
            if fit_params.get("base_model_id") or fit_params.get("base_model_path") or fit_params.get("active_state_id") or fit_params.get("inactive_state_id"):
                fit_mode = "delta"
            elif fit_params.get("active_npz") or fit_params.get("inactive_npz"):
                fit_mode = "delta"
            else:
                fit_mode = "standard"
        fit_mode = str(fit_mode)
        fit_params["fit_mode"] = fit_mode
        result_payload["params"] = fit_params

        if fit_mode == "delta":
            base_model_rel = fit_params.get("base_model_path")
            base_model_id = fit_params.get("base_model_id")
            base_model_name = None
            if base_model_id and isinstance(entry, dict):
                models = entry.get("potts_models")
                if isinstance(models, list):
                    base_entry = next((m for m in models if m.get("model_id") == base_model_id), None)
                    if base_entry:
                        base_model_rel = base_entry.get("path") or base_model_rel
                        base_model_name = base_entry.get("name") or base_model_name
            if not base_model_rel:
                raise FileNotFoundError("Base Potts model path missing for delta fit.")
            base_model_path = Path(base_model_rel)
            if not base_model_path.is_absolute():
                base_model_path = project_store.resolve_path(project_id, system_id, base_model_rel)
            if not base_model_path.exists():
                raise FileNotFoundError(f"Base Potts model is missing on disk: {base_model_path}")

            cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
            results_dir = cluster_dirs["potts_models_dir"] / f"delta_fit_{job_uuid}"
            results_dir.mkdir(parents=True, exist_ok=True)

            args_list = [
                "--base-model",
                str(base_model_path),
                "--results-dir",
                str(results_dir),
            ]
            state_ids = _coerce_str_list(fit_params.get("state_ids"))
            active_npz = fit_params.get("active_npz")
            inactive_npz = fit_params.get("inactive_npz")
            if state_ids:
                args_list += ["--npz", str(cluster_path), "--state-ids", ",".join(state_ids)]
            elif active_npz and inactive_npz:
                active_npz_path = Path(active_npz)
                if not active_npz_path.is_absolute():
                    active_npz_path = project_store.resolve_path(project_id, system_id, active_npz)
                inactive_npz_path = Path(inactive_npz)
                if not inactive_npz_path.is_absolute():
                    inactive_npz_path = project_store.resolve_path(project_id, system_id, inactive_npz)
                if not active_npz_path.exists() or not inactive_npz_path.exists():
                    raise FileNotFoundError("Active/inactive NPZ file(s) missing on disk.")
                args_list += ["--active-npz", str(active_npz_path), "--inactive-npz", str(inactive_npz_path)]
            else:
                active_state_id = fit_params.get("active_state_id")
                inactive_state_id = fit_params.get("inactive_state_id")
                args_list += [
                    "--npz",
                    str(cluster_path),
                    "--active-state-id",
                    str(active_state_id),
                    "--inactive-state-id",
                    str(inactive_state_id),
                ]

            unassigned_policy = fit_params.get("unassigned_policy")
            if unassigned_policy:
                args_list += ["--unassigned-policy", str(unassigned_policy)]

            for key, flag in (
                ("delta_epochs", "--epochs"),
                ("delta_lr", "--lr"),
                ("delta_lr_min", "--lr-min"),
                ("delta_lr_schedule", "--lr-schedule"),
                ("delta_batch_size", "--batch-size"),
                ("delta_grad_accum_steps", "--grad-accum-steps"),
                ("delta_seed", "--seed"),
                ("delta_device", "--device"),
                ("delta_l2", "--delta-l2"),
                ("delta_group_h", "--delta-group-h"),
                ("delta_group_j", "--delta-group-j"),
            ):
                val = fit_params.get(key)
                if val is not None:
                    args_list += [flag, str(val)]
            if fit_params.get("delta_no_combined"):
                args_list.append("--no-combined")

            save_progress("Fitting delta Potts models", 20)
            try:
                delta_fit_main.main(args_list)
            except SystemExit as exc:
                raise ValueError("Invalid delta potts fit arguments.") from exc

            cluster_name = entry.get("name") if isinstance(entry, dict) else None
            name_root = None
            if isinstance(model_name, str) and model_name.strip():
                name_root = model_name.strip()
            elif base_model_name:
                name_root = f"{base_model_name} Delta"
            elif cluster_name:
                name_root = f"{cluster_name} Delta"
            if not name_root:
                name_root = f"{cluster_id} Delta"
            if state_ids:
                name_root = f"{name_root} ({','.join(state_ids)})"

            persisted_models: List[Dict[str, Any]] = []

            def persist_delta_model(path: Path, kind: str, name: str) -> None:
                if not path.exists():
                    return
                model_id = str(uuid.uuid4())
                model_params = dict(fit_params)
                model_params.update(
                    {
                        "fit_mode": "delta",
                        "delta_kind": kind,
                        "state_ids": state_ids or None,
                        "base_model_id": base_model_id,
                        "base_model_path": base_model_rel,
                    }
                )
                rel = _persist_potts_model(
                    project_id,
                    system_id,
                    cluster_id,
                    path,
                    model_params,
                    source="potts_delta_fit",
                    model_id=model_id,
                    model_name=name,
                )
                persisted_models.append({"model_id": model_id, "name": name, "path": rel, "kind": kind})

            if state_ids:
                persist_delta_model(results_dir / "delta_model.npz", "delta_patch", f"{name_root} (delta)")
                persist_delta_model(results_dir / "model_combined.npz", "model_patch", f"{name_root} (combined)")
            else:
                persist_delta_model(results_dir / "delta_active.npz", "delta_active", f"{name_root} (delta active)")
                persist_delta_model(results_dir / "delta_inactive.npz", "delta_inactive", f"{name_root} (delta inactive)")
                persist_delta_model(results_dir / "model_active.npz", "model_active", f"{name_root} (combined active)")
                persist_delta_model(results_dir / "model_inactive.npz", "model_inactive", f"{name_root} (combined inactive)")

            meta_path = results_dir / "delta_fit_metadata.json"
            meta_rel = _relativize_path(meta_path) if meta_path.exists() else None
            if cluster_name:
                result_payload["system_reference"]["cluster_name"] = cluster_name

            result_payload["status"] = "finished"
            result_payload["results"] = {
                "results_dir": _relativize_path(results_dir),
                "potts_model": persisted_models[0]["path"] if persisted_models else None,
                "potts_models": persisted_models or None,
                "cluster_npz": _relativize_path(cluster_path),
                "metadata_json": meta_rel,
            }
        else:
            def _resolve_potts_model_path(value: object) -> str | None:
                if not value:
                    return None
                raw = str(value)
                if isinstance(entry, dict):
                    models = entry.get("potts_models") or []
                    match = next((m for m in models if m.get("model_id") == raw), None)
                    if match and match.get("path"):
                        raw = str(match.get("path"))
                path_obj = Path(raw)
                if not path_obj.is_absolute():
                    path_obj = project_store.resolve_path(project_id, system_id, raw)
                return str(path_obj)

            resolved_resume_model = _resolve_potts_model_path(fit_params.get("plm_resume_model"))
            resume_in_place = bool(resolved_resume_model)
            requested_sample_ids = _dedupe_preserve_order(_coerce_str_list(fit_params.get("sample_ids")))

            if resume_in_place:
                model_path = Path(resolved_resume_model)
                model_dir = model_path.parent
                model_id = None
            else:
                model_id = str(uuid.uuid4())
                display_name = None
                if isinstance(model_name, str) and model_name.strip():
                    display_name = model_name.strip()
                elif isinstance(entry, dict):
                    cluster_name = entry.get("name")
                    if isinstance(cluster_name, str) and cluster_name.strip():
                        display_name = f"{cluster_name} Potts Model"
                if not display_name:
                    display_name = f"{cluster_id} Potts Model"
                cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
                model_dir = cluster_dirs["potts_models_dir"] / model_id
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{_sanitize_model_filename(display_name)}.npz"

            fit_dataset_summary = None
            fit_dataset_path = cluster_path
            fit_sample_state_ids: List[str] = []
            if requested_sample_ids:
                fit_dataset_summary = _build_standard_fit_dataset_from_md_samples(
                    project_id=project_id,
                    system_id=system_id,
                    cluster_id=cluster_id,
                    cluster_path=cluster_path,
                    sample_entries=list(entry.get("samples") or project_store.list_samples(project_id, system_id, cluster_id)),
                    sample_ids=requested_sample_ids,
                    output_path=model_dir / "fit_dataset.npz",
                )
                fit_dataset_path = Path(fit_dataset_summary["path"])
                fit_sample_state_ids = list(fit_dataset_summary.get("state_ids") or [])
                fit_params["fit_dataset_type"] = "selected_md_samples"
                fit_params["fit_sample_ids"] = requested_sample_ids
                fit_params["fit_sample_state_ids"] = fit_sample_state_ids
                fit_params["data_npz"] = _relativize_path(fit_dataset_path)
            else:
                fit_params.pop("data_npz", None)

            args_list = [
                "--npz",
                str(fit_dataset_path),
                "--results-dir",
                str(model_dir),
                "--fit-only",
                "--model-out",
                str(model_path),
            ]

            fit_method = fit_params.get("fit_method")
            if fit_method is not None:
                args_list += ["--fit", str(fit_method)]
            plm_init_model = fit_params.get("plm_init_model")
            plm_resume_model = fit_params.get("plm_resume_model")
            resolved_init = _resolve_potts_model_path(plm_init_model)
            resolved_resume = _resolve_potts_model_path(plm_resume_model)
            if resolved_init:
                fit_params["plm_init_model"] = resolved_init
            if resolved_resume:
                fit_params["plm_resume_model"] = resolved_resume
            using_existing_model = bool(resolved_init or resolved_resume)

            if not using_existing_model:
                contact_mode = fit_params.get("contact_atom_mode") or fit_params.get("contact_mode") or "CA"
                contact_cutoff = fit_params.get("contact_cutoff") or 10.0
                contact_pdbs = _resolve_contact_pdbs(
                    project_id,
                    system_id,
                    _collect_contact_pdbs(
                        system_meta,
                        fit_sample_state_ids or entry.get("state_ids") or entry.get("metastable_ids") or [],
                        system_meta.analysis_mode,
                    ),
                )
                if contact_pdbs:
                    pdb_flag = None
                    if _supports_sim_arg("--pdbs"):
                        pdb_flag = "--pdbs"
                    elif _supports_sim_arg("--contact-pdb"):
                        pdb_flag = "--contact-pdb"
                    if pdb_flag:
                        args_list += [
                            pdb_flag,
                            ",".join(str(p) for p in contact_pdbs),
                            "--contact-cutoff",
                            str(float(contact_cutoff)),
                            "--contact-atom-mode",
                            str(contact_mode),
                        ]
                    else:
                        print("[potts-sample] warning: simulation args do not support contact PDBs; skipping edge build.")

            for key, flag in (
                ("plm_epochs", "--plm-epochs"),
                ("plm_lr", "--plm-lr"),
                ("plm_lr_min", "--plm-lr-min"),
                ("plm_lr_schedule", "--plm-lr-schedule"),
                ("plm_l2", "--plm-l2"),
                ("plm_batch_size", "--plm-batch-size"),
                ("plm_grad_accum_steps", "--plm-grad-accum-steps"),
                ("plm_progress_every", "--plm-progress-every"),
                ("plm_device", "--plm-device"),
                ("plm_init", "--plm-init"),
                ("plm_init_model", "--plm-init-model"),
                ("plm_resume_model", "--plm-resume-model"),
                ("plm_val_frac", "--plm-val-frac"),
            ):
                val = fit_params.get(key)
                if val is not None:
                    args_list += [flag, str(val)]

            save_progress("Fitting Potts model", 20)
            try:
                sim_args = parse_simulation_args(args_list)
            except SystemExit as exc:
                raise ValueError("Invalid potts fit arguments.") from exc
            run_result = run_simulation_pipeline(
                sim_args,
                progress_callback=save_progress,
                runtime=RuntimePolicy(allow_multiprocessing=False),
            )
            if fit_dataset_summary:
                metadata_path = model_dir / "model_metadata.json"
                try:
                    metadata_payload: Dict[str, Any] = {}
                    if metadata_path.exists():
                        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata_payload["data_npz"] = _relativize_path(fit_dataset_path)
                    metadata_payload["fit_dataset_type"] = "selected_md_samples"
                    metadata_payload["fit_sample_ids"] = requested_sample_ids
                    metadata_payload["fit_sample_state_ids"] = fit_sample_state_ids or None
                    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
                except Exception as exc:
                    print(f"[PottsFit {job_uuid}] warning: failed to update model metadata provenance: {exc}")
            if resume_in_place:
                potts_model_rel = _relativize_path(Path(model_path))
            else:
                potts_model_rel = _persist_potts_model(
                    project_id,
                    system_id,
                    cluster_id,
                    Path(model_path),
                    fit_params,
                    source="potts_fit",
                    model_id=model_id,
                    model_name=display_name,
                )

            result_payload["status"] = "finished"
            result_payload["results"] = {
                "results_dir": _relativize_path(model_dir),
                "potts_model": potts_model_rel,
                "cluster_npz": _relativize_path(cluster_path),
                "fit_dataset_npz": _relativize_path(fit_dataset_path) if fit_dataset_summary else None,
                "sample_ids": requested_sample_ids or None,
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


def run_backmapping_job(job_uuid: str, project_id: str, system_id: str, cluster_id: str) -> Dict[str, Any]:
    job = get_current_job()
    start_time = datetime.utcnow()
    rq_job_id = job.id if job else f"backmapping-{job_uuid}"

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[Backmapping {job_uuid}] {status_msg}")

    save_progress("Initializing...", 0)
    system_meta = project_store.get_system(project_id, system_id)
    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise ValueError(f"Cluster NPZ not found for cluster_id='{cluster_id}'.")
    rel_path = entry.get("path")
    if not rel_path:
        raise ValueError("Cluster NPZ path missing in system metadata.")
    cluster_path = Path(rel_path)
    if not cluster_path.is_absolute():
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ file is missing on disk: {cluster_path}")

    dirs = project_store.ensure_directories(project_id, system_id)
    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    out_dir = cluster_dirs["cluster_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backmapping.npz"

    def progress_callback(current: int, total: int):
        if not total:
            return
        ratio = max(0.0, min(1.0, current / float(total)))
        progress = 5 + int(ratio * 90)
        save_progress("Building backmapping NPZ...", min(progress, 95))

    build_backmapping_npz(
        project_id,
        system_id,
        cluster_path,
        out_path,
        progress_callback=progress_callback,
    )

    save_progress("Completed", 100)
    return {
        "job_id": rq_job_id,
        "status": "finished",
        "cluster_id": cluster_id,
        "path": str(out_path.relative_to(dirs["system_dir"])),
        "completed_at": datetime.utcnow().isoformat(),
        "started_at": start_time.isoformat(),
    }


def run_sample_backmapping_job(
    job_uuid: str,
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_id: str,
    trajectory_path: str,
) -> Dict[str, Any]:
    job = get_current_job()
    started_at = datetime.utcnow().isoformat()
    rq_job_id = job.id if job else f"sample-backmapping-{job_uuid}"

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta["status"] = status_msg
            job.meta["progress"] = progress
            job.save_meta()
        print(f"[SampleBackmapping {job_uuid}] {status_msg}")

    sample_meta = project_store.get_sample_entry(project_id, system_id, cluster_id, sample_id)
    existing = dict(sample_meta.get("backmapping_dataset") or {})
    temp_traj_path = Path(trajectory_path)
    if not temp_traj_path.is_absolute():
        temp_traj_path = project_store.resolve_path(project_id, system_id, str(trajectory_path))
    sample_dir = project_store._sample_dir(project_id, system_id, cluster_id, sample_id)
    system_dir = project_store.ensure_directories(project_id, system_id)["system_dir"]
    out_path = sample_dir / "backmapping_dataset.npz"
    relative_out = str(out_path.relative_to(system_dir))

    def persist(status: str, **extra: Any) -> None:
        payload = {
            **existing,
            "status": status,
            "job_id": rq_job_id,
            "updated_at": datetime.utcnow().isoformat(),
            "path": existing.get("path") or relative_out,
            **extra,
        }
        sample_meta["backmapping_dataset"] = payload
        project_store.save_sample_entry(project_id, system_id, cluster_id, sample_id, sample_meta)

    save_progress("Initializing...", 0)
    persist("running", started_at=started_at, error=None)

    def progress_callback(current: int, total: int):
        if not total:
            return
        ratio = max(0.0, min(1.0, current / float(total)))
        progress = 5 + int(ratio * 90)
        save_progress("Building backmapping dataset...", min(progress, 95))
        persist("running", progress=min(progress, 95))

    try:
        summary = build_sample_backmapping_dataset(
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            sample_id=sample_id,
            trajectory_path=temp_traj_path,
            output_path=out_path,
            progress_callback=progress_callback,
        )
        completed_at = datetime.utcnow().isoformat()
        persist(
            "finished",
            progress=100,
            path=relative_out,
            source_trajectory_name=temp_traj_path.name,
            started_at=started_at,
            completed_at=completed_at,
            n_frames=int(summary.get("n_frames") or 0),
            n_atoms=int(summary.get("n_atoms") or 0),
            n_residues=int(summary.get("n_residues") or 0),
            dihedral_keys=list(summary.get("dihedral_keys") or []),
            error=None,
        )
        save_progress("Completed", 100)
        return {
            "job_id": rq_job_id,
            "status": "finished",
            "project_id": project_id,
            "system_id": system_id,
            "cluster_id": cluster_id,
            "sample_id": sample_id,
            "path": relative_out,
            "started_at": started_at,
            "completed_at": completed_at,
            **summary,
        }
    except Exception as exc:
        persist("failed", progress=100, error=str(exc))
        save_progress(f"Failed: {exc}", 100)
        raise
    finally:
        try:
            temp_traj_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            parent = temp_traj_path.parent
            if parent != sample_dir and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass


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

        meta_ids = params.get("state_ids") or params.get("metastable_ids") or []
        if not meta_ids:
            raise ValueError("Provide at least one state_id.")

        cluster_name = params.get("cluster_name")
        output_path = build_cluster_output_path(
            project_id,
            system_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
        )

        save_progress("Clustering residues...", 10)
        npz_path, meta = generate_metastable_cluster_npz(
            project_id,
            system_id,
            meta_ids,
            output_path=output_path,
            cluster_name=cluster_name,
            max_cluster_frames=params.get("max_cluster_frames"),
            random_state=params.get("random_state", 0),
            cluster_algorithm="density_peaks",
            density_maxk=params.get("density_maxk", 100),
            density_z=params.get("density_z"),
            progress_callback=progress_callback,
        )

        save_progress("Saving cluster metadata...", 90)
        dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        try:
            rel_path = str(npz_path.relative_to(dirs["system_dir"]))
        except Exception:
            rel_path = str(npz_path)

        selected_state_ids = []
        try:
            system_meta = project_store.get_system(project_id, system_id)
            descriptor_state_ids = {
                str(sid)
                for sid, state in (system_meta.states or {}).items()
                if getattr(state, "descriptor_file", None)
            }
            selected_state_ids = [str(v) for v in meta_ids if str(v) in descriptor_state_ids]
        except Exception:
            selected_state_ids = []

        save_progress("Assigning clusters to MD states...", 92)
        assignments = build_md_eval_samples_for_cluster(
            project_id,
            system_id,
            cluster_id,
            cluster_path=npz_path,
            selected_state_ids=selected_state_ids,
            include_remaining_states=True,
            store=project_store,
        )

        _update_cluster_entry(
            project_id,
            system_id,
            cluster_id,
            {
                "status": "finished",
                "progress": 100,
                "path": rel_path,
                "samples": assignments.get("samples", []),
                "generated_at": meta.get("generated_at") if isinstance(meta, dict) else None,
                "contact_edge_count": meta.get("contact_edge_count") if isinstance(meta, dict) else None,
                "descriptor_symmetry": meta.get("descriptor_symmetry") if isinstance(meta, dict) else None,
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
