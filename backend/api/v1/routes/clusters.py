import logging
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

import numpy as np

from backend.api.v1.common import DATA_ROOT, get_queue, project_store, stream_upload, get_cluster_entry
from phase.workflows.clustering import (
    generate_metastable_cluster_npz,
    assign_cluster_labels_to_states,
    update_cluster_metadata_with_assignments,
    build_cluster_entry,
    build_cluster_output_path,
    _slug,
    evaluate_state_with_models,
)
from phase.workflows.backmapping import build_backmapping_npz
from backend.tasks import run_cluster_job, run_backmapping_job


router = APIRouter()


def _remove_results_dir(path_value: str | None, *, system_dir: Path | None = None) -> None:
    if not isinstance(path_value, str) or not path_value:
        return
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = DATA_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(DATA_ROOT)
    except ValueError:
        return
    if system_dir is not None:
        results_root = (system_dir / "results").resolve()
        try:
            candidate.relative_to(results_root)
        except ValueError:
            return
    if candidate.exists() and candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)


def _matches_sample(payload: dict, sample_id: str) -> bool:
    system_ref = payload.get("system_reference") or {}
    if system_ref.get("sample_id") == sample_id:
        return True
    results_payload = payload.get("results") or {}
    for key in ("results_dir", "summary_npz"):
        value = results_payload.get(key)
        if not isinstance(value, str) or not value:
            continue
        parts = Path(value).parts
        if sample_id in parts:
            return True
    return False


def _matches_model(payload: dict, model_id: str) -> bool:
    system_ref = payload.get("system_reference") or {}
    if system_ref.get("potts_model_id") == model_id:
        return True
    params_payload = payload.get("params") or {}
    if params_payload.get("potts_model_id") == model_id:
        return True
    results_payload = payload.get("results") or {}
    value = results_payload.get("potts_model")
    if isinstance(value, str) and value:
        parts = Path(value).parts
        if model_id in parts:
            return True
    return False


def _remove_simulation_results(
    project_id: str,
    system_id: str,
    cluster_id: str,
    *,
    sample_id: str | None = None,
    model_id: str | None = None,
) -> int:
    removed = 0
    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    jobs_dir = results_dirs["jobs_dir"]
    system_dir = results_dirs["system_dir"]
    if not jobs_dir.exists():
        return removed
    for result_file in jobs_dir.glob("*.json"):
        try:
            payload = json.loads(result_file.read_text())
        except Exception:
            continue
        if payload.get("analysis_type") != "simulation":
            continue
        system_ref = payload.get("system_reference") or {}
        if cluster_id and system_ref.get("cluster_id") != cluster_id:
            continue
        if sample_id and not _matches_sample(payload, sample_id):
            continue
        if model_id and not _matches_model(payload, model_id):
            continue
        try:
            _remove_results_dir((payload.get("results") or {}).get("results_dir"), system_dir=system_dir)
            result_file.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    if jobs_dir.exists():
        try:
            if not any(jobs_dir.iterdir()):
                jobs_dir.rmdir()
        except Exception:
            pass
    results_dir = system_dir / "results"
    if results_dir.exists():
        try:
            if not any(results_dir.iterdir()):
                results_dir.rmdir()
        except Exception:
            pass
    return removed


def _parse_state_ids(raw: str) -> List[str]:
    if not raw:
        return []
    return [val.strip() for val in raw.split(",") if val.strip()]


def _decode_metadata_json(meta_raw: Any) -> Dict[str, Any]:
    value = meta_raw
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.ravel()[0]
        else:
            try:
                value = value.tolist()
            except Exception as exc:
                raise ValueError("metadata_json has unexpected shape.") from exc
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, list):
        if value:
            value = value[0]
        else:
            raise ValueError("metadata_json list is empty.")
    if isinstance(value, str):
        if value.startswith("array("):
            match = re.search(r"array\\(['\"](.*)['\"],\\s*dtype=.*\\)$", value, re.S)
            if match:
                value = match.group(1)
        return json.loads(value)
    raise ValueError("metadata_json is not a valid JSON string.")


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/samples/{sample_id}/summary",
    summary="Load a sampling summary NPZ for visualization",
)
async def get_sampling_summary(
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_id: str,
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    entry = get_cluster_entry(system_meta, cluster_id)
    samples = entry.get("samples") if isinstance(entry, dict) else None
    if not isinstance(samples, list):
        raise HTTPException(status_code=404, detail="No samples recorded for this cluster.")
    sample_entry = next((s for s in samples if isinstance(s, dict) and s.get("sample_id") == sample_id), None)
    if not sample_entry:
        raise HTTPException(status_code=404, detail="Sample not found in cluster metadata.")
    paths = sample_entry.get("paths") if isinstance(sample_entry, dict) else None
    summary_rel = paths.get("summary_npz") if isinstance(paths, dict) else None
    if not summary_rel:
        raise HTTPException(status_code=404, detail="Sample summary NPZ is missing.")
    summary_path = project_store.resolve_path(project_id, system_id, summary_rel)
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Sample summary NPZ not found on disk.")

    try:
        with np.load(summary_path, allow_pickle=True) as data:
            def _get(name, default=None):
                if name not in data:
                    return default
                value = data[name]
                if isinstance(value, np.ndarray):
                    return value.tolist()
                return value

            payload = {
                "sample_id": sample_entry.get("sample_id"),
                "sample_name": sample_entry.get("name"),
                "sample_method": sample_entry.get("method"),
                "model_id": sample_entry.get("model_id"),
                "cluster_id": cluster_id,
                "summary_keys": data.files,
                "K": _get("K", []),
                "edges": _get("edges", []),
                "residue_labels": _get("residue_labels", []),
                "md_source_ids": _get("md_source_ids", []),
                "md_source_labels": _get("md_source_labels", []),
                "md_source_types": _get("md_source_types", []),
                "md_source_counts": _get("md_source_counts", []),
                "sample_source_ids": _get("sample_source_ids", []),
                "sample_source_labels": _get("sample_source_labels", []),
                "sample_source_types": _get("sample_source_types", []),
                "sample_source_counts": _get("sample_source_counts", []),
                "js_md_sample": _get("js_md_sample", []),
                "js2_md_sample": _get("js2_md_sample", []),
                "edge_strength": _get("edge_strength", []),
                "energy_bins": _get("energy_bins", []),
                "energy_hist_md": _get("energy_hist_md", []),
                "energy_hist_sample": _get("energy_hist_sample", []),
                "xlik_delta_active": _get("xlik_delta_active", []),
                "xlik_delta_inactive": _get("xlik_delta_inactive", []),
                "xlik_auc": _get("xlik_auc", []),
                "xlik_active_state_labels": _get("xlik_active_state_labels", []),
                "xlik_inactive_state_labels": _get("xlik_inactive_state_labels", []),
                "xlik_other_state_labels": _get("xlik_other_state_labels", []),
                "beta_eff_grid": _get("beta_eff_grid", []),
                "beta_eff_distances_by_schedule": _get("beta_eff_distances_by_schedule", []),
                "sa_schedule_labels": _get("sa_schedule_labels", []),
                "beta_eff": _get("beta_eff", []),
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sampling summary: {exc}") from exc

    try:
        state_label_map = {sid: state.name for sid, state in system_meta.states.items()}
        meta_label_map = {str(m.get("metastable_id")): m.get("name") for m in system_meta.metastable_states or []}
        md_source_ids = payload.get("md_source_ids") or []
        md_source_labels = payload.get("md_source_labels") or []
        normalized = []
        for idx, src_id in enumerate(md_source_ids):
            label = md_source_labels[idx] if idx < len(md_source_labels) else str(src_id)
            if isinstance(src_id, str) and src_id.startswith("state:"):
                state_id = src_id.split(":", 1)[-1]
                label = state_label_map.get(state_id, label)
            elif isinstance(src_id, str) and src_id.startswith("meta:"):
                meta_id = src_id.split(":", 1)[-1]
                label = meta_label_map.get(meta_id, label)
            normalized.append(label)
        payload["md_source_labels"] = normalized
    except Exception:
        pass

    return payload


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/samples/{sample_id}/artifacts/{artifact}",
    summary="Download a sampling artifact from a cluster sample",
)
async def download_sampling_artifact(
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_id: str,
    artifact: str,
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    entry = get_cluster_entry(system_meta, cluster_id)
    samples = entry.get("samples") if isinstance(entry, dict) else None
    if not isinstance(samples, list):
        raise HTTPException(status_code=404, detail="No samples recorded for this cluster.")
    sample_entry = next((s for s in samples if isinstance(s, dict) and s.get("sample_id") == sample_id), None)
    if not sample_entry:
        raise HTTPException(status_code=404, detail="Sample not found in cluster metadata.")

    key_map = {
        "summary_npz": "summary_npz",
        "metadata_json": "metadata_json",
        "marginals_plot": "marginals_plot",
        "sampling_report": "sampling_report",
        "cross_likelihood_report": "cross_likelihood_report",
        "beta_scan_plot": "beta_scan_plot",
    }
    artifact_key = key_map.get(artifact)
    if not artifact_key:
        raise HTTPException(status_code=404, detail="Unknown artifact.")

    paths = sample_entry.get("paths") if isinstance(sample_entry, dict) else None
    rel_path = paths.get(artifact_key) if isinstance(paths, dict) else None
    if not rel_path:
        raise HTTPException(status_code=404, detail="Artifact not available for this sample.")
    artifact_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk.")

    media_type = _artifact_media_type(artifact_path)
    headers = {}
    if media_type == "text/html":
        headers["Content-Disposition"] = f"inline; filename={artifact_path.name}"
    return FileResponse(artifact_path, filename=artifact_path.name, media_type=media_type, headers=headers)


@router.delete(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/samples/{sample_id}",
    summary="Delete a sampling sample and its artifacts",
)
async def delete_sampling_sample(
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_id: str,
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    entry = get_cluster_entry(system_meta, cluster_id)
    samples = entry.get("samples") if isinstance(entry, dict) else None
    if not isinstance(samples, list):
        raise HTTPException(status_code=404, detail="No samples recorded for this cluster.")

    sample_entry = next((s for s in samples if isinstance(s, dict) and s.get("sample_id") == sample_id), None)
    if not sample_entry:
        raise HTTPException(status_code=404, detail="Sample not found in cluster metadata.")

    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    sample_dir = cluster_dirs["samples_dir"] / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir, ignore_errors=True)

    paths = sample_entry.get("paths") if isinstance(sample_entry, dict) else None
    if isinstance(paths, dict):
        for rel_path in paths.values():
            if not rel_path:
                continue
            path = project_store.resolve_path(project_id, system_id, rel_path)
            if path.exists() and path.is_file():
                try:
                    path.unlink()
                except Exception:
                    pass

    entry["samples"] = [s for s in samples if not (isinstance(s, dict) and s.get("sample_id") == sample_id)]
    project_store.save_system(system_meta)
    _remove_simulation_results(
        project_id,
        system_id,
        cluster_id,
        sample_id=sample_id,
    )
    return {"status": "deleted", "sample_id": sample_id}


def _parse_cluster_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metastable_ids_raw = (payload or {}).get("state_ids")
    if metastable_ids_raw is None:
        metastable_ids_raw = (payload or {}).get("metastable_ids") or []
    cluster_name = (payload or {}).get("cluster_name")
    if not isinstance(metastable_ids_raw, list):
        raise HTTPException(status_code=400, detail="state_ids must be a list.")
    metastable_ids = [str(mid).strip() for mid in metastable_ids_raw if str(mid).strip()]
    if not metastable_ids:
        raise HTTPException(status_code=400, detail="Provide at least one state_id.")

    max_cluster_frames = None
    if (payload or {}).get("max_cluster_frames") is not None:
        try:
            max_cluster_frames = int((payload or {}).get("max_cluster_frames"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="max_cluster_frames must be an integer.")
        if max_cluster_frames < 1:
            raise HTTPException(status_code=400, detail="max_cluster_frames must be >= 1.")

    try:
        random_state = int((payload or {}).get("random_state", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="random_state must be an integer.")

    algo_params = (payload or {}).get("algorithm_params", {}) or {}
    try:
        density_maxk = algo_params.get("maxk", (payload or {}).get("density_maxk"))
        if density_maxk is not None:
            density_maxk = int(density_maxk)
        else:
            density_maxk = 100
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="density maxk must be an integer >=1.")
    density_z_raw = algo_params.get("Z")
    if density_z_raw is None:
        density_z_raw = (payload or {}).get("density_z")
    if density_z_raw is None:
        density_z = "auto"
    elif isinstance(density_z_raw, str) and density_z_raw.lower() == "auto":
        density_z = "auto"
    else:
        try:
            density_z = float(density_z_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="density_peaks Z must be a number or 'auto'.")

    return {
        "state_ids": metastable_ids,
        "metastable_ids": metastable_ids,
        "cluster_name": cluster_name,
        "max_cluster_frames": max_cluster_frames,
        "random_state": random_state,
        "cluster_algorithm": "density_peaks",
        "density_maxk": density_maxk,
        "density_z": density_z,
    }


def _build_algorithm_params(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "density_maxk": parsed["density_maxk"],
        "density_z": parsed["density_z"],
        "max_cluster_frames": parsed["max_cluster_frames"],
    }


def _build_cluster_entry(parsed: Dict[str, Any], cluster_id: str, status: str, progress: int, status_message: str):
    entry = build_cluster_entry(
        cluster_id=cluster_id,
        cluster_name=parsed.get("cluster_name").strip()
        if isinstance(parsed.get("cluster_name"), str) and parsed.get("cluster_name").strip()
        else None,
        state_ids=parsed.get("state_ids") or parsed.get("metastable_ids") or [],
        max_cluster_frames=parsed.get("max_cluster_frames"),
        random_state=parsed.get("random_state", 0),
        density_maxk=parsed.get("density_maxk"),
        density_z=parsed.get("density_z"),
    )
    entry.update({"status": status, "progress": progress, "status_message": status_message})
    return entry


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/cluster_vectors",
    summary="Cluster residue angles inside selected metastable states and download NPZ",
)
async def build_metastable_cluster_vectors(
    project_id: str,
    system_id: str,
    payload: Dict[str, Any],
):
    logger = logging.getLogger("cluster_build")
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro-states before clustering.")
    analysis_mode = getattr(system_meta, "analysis_mode", None)
    if not getattr(system_meta, "metastable_locked", False) and analysis_mode != "macro":
        raise HTTPException(
            status_code=400,
            detail="Lock metastable states or confirm macro-only before clustering.",
        )

    parsed = _parse_cluster_payload(payload)
    cluster_id = str(uuid.uuid4())
    cluster_name = parsed.get("cluster_name")
    output_path = build_cluster_output_path(
        project_id,
        system_id,
        cluster_id=cluster_id,
        cluster_name=cluster_name,
    )

    try:
        if logger:
            logger.error(
                "[cluster_build] project=%s system=%s meta_ids=%s algo=%s params=%s",
                project_id,
                system_id,
                parsed["state_ids"],
                parsed["cluster_algorithm"],
                {
                    "random_state": parsed["random_state"],
                    "density_peaks_maxk": parsed["density_maxk"],
                    "density_peaks_Z": parsed["density_z"],
                },
            )
        npz_path, meta = await run_in_threadpool(
            generate_metastable_cluster_npz,
            project_id,
            system_id,
            parsed["state_ids"],
            output_path=output_path,
            cluster_name=cluster_name,
            max_cluster_frames=parsed["max_cluster_frames"],
            random_state=parsed["random_state"],
            cluster_algorithm=parsed["cluster_algorithm"],
            density_maxk=parsed["density_maxk"],
            density_z=parsed["density_z"],
        )
    except ValueError as exc:
        if logger:
            logger.error("[cluster_build] validation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if logger:
            logger.error("[cluster_build] failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build metastable clusters: {exc}") from exc

    dirs = project_store.ensure_directories(project_id, system_id)
    try:
        rel_path = str(npz_path.relative_to(dirs["system_dir"]))
    except Exception:
        rel_path = str(npz_path)

    assignments = assign_cluster_labels_to_states(npz_path, project_id, system_id)
    update_cluster_metadata_with_assignments(npz_path, assignments)

    cluster_entry = _build_cluster_entry(parsed, cluster_id, "finished", 100, "Complete")
    cluster_entry.update(
        {
            "path": rel_path,
            "generated_at": meta.get("generated_at") if isinstance(meta, dict) else None,
            "contact_edge_count": meta.get("contact_edge_count") if isinstance(meta, dict) else None,
            "assigned_state_paths": assignments.get("assigned_state_paths", {}),
            "assigned_metastable_paths": assignments.get("assigned_metastable_paths", {}),
        }
    )
    system_meta.metastable_clusters = (system_meta.metastable_clusters or []) + [cluster_entry]
    project_store.save_system(system_meta)

    return FileResponse(
        npz_path,
        filename=npz_path.name,
        media_type="application/octet-stream",
    )


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/cluster_jobs",
    summary="Queue per-residue clustering and track progress",
)
async def submit_metastable_cluster_job(
    project_id: str,
    system_id: str,
    payload: Dict[str, Any],
    request: Request,
):
    task_queue = get_queue(request)
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro-states before clustering.")
    if not getattr(system_meta, "metastable_locked", False) and getattr(system_meta, "analysis_mode", None) != "macro":
        raise HTTPException(
            status_code=400,
            detail="Lock metastable states or confirm macro-only before clustering.",
        )

    parsed = _parse_cluster_payload(payload)
    cluster_id = str(uuid.uuid4())
    cluster_entry = _build_cluster_entry(parsed, cluster_id, "queued", 0, "Queued")
    system_meta.metastable_clusters = (system_meta.metastable_clusters or []) + [cluster_entry]
    project_store.save_system(system_meta)

    job_uuid = str(uuid.uuid4())
    try:
        job = task_queue.enqueue(
            run_cluster_job,
            args=(job_uuid, project_id, system_id, cluster_id, parsed),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"cluster-{job_uuid}",
        )
    except Exception as exc:
        system_meta.metastable_clusters = [
            c for c in system_meta.metastable_clusters or [] if c.get("cluster_id") != cluster_id
        ]
        project_store.save_system(system_meta)
        raise HTTPException(status_code=500, detail=f"Cluster job submission failed: {exc}") from exc

    cluster_entry["job_id"] = job.id
    cluster_entry["updated_at"] = datetime.utcnow().isoformat()
    project_store.save_system(system_meta)
    return {"status": "queued", "job_id": job.id, "cluster_id": cluster_id}


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}",
    summary="Download a previously generated metastable cluster NPZ",
)
async def download_metastable_cluster_npz(project_id: str, system_id: str, cluster_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")
    rel_path = entry.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="Cluster NPZ path missing.")
    abs_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Cluster NPZ file is missing on disk.")
    return FileResponse(abs_path, filename=abs_path.name, media_type="application/octet-stream")


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/backmapping_npz",
    summary="Download a backmapping-ready NPZ with coordinates and torsions",
)
async def download_backmapping_npz(project_id: str, system_id: str, cluster_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")
    rel_path = entry.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="Cluster NPZ path missing.")
    cluster_path = Path(rel_path)
    if not cluster_path.is_absolute():
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not cluster_path.exists():
        raise HTTPException(status_code=404, detail="Cluster NPZ file is missing on disk.")

    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    out_dir = cluster_dirs["cluster_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backmapping.npz"

    if not out_path.exists():
        try:
            await run_in_threadpool(build_backmapping_npz, project_id, system_id, cluster_path, out_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build backmapping NPZ: {exc}") from exc

    return FileResponse(out_path, filename=out_path.name, media_type="application/octet-stream")


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/backmapping_npz/job",
    summary="Queue backmapping NPZ generation",
)
async def queue_backmapping_npz_job(
    project_id: str,
    system_id: str,
    cluster_id: str,
    task_queue: Any = Depends(get_queue),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_backmapping_job,
            args=(job_uuid, project_id, system_id, cluster_id),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"backmapping-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "cluster_id": cluster_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/backmapping_npz/upload",
    summary="Upload trajectories on-demand to build a backmapping NPZ",
)
async def upload_backmapping_npz(
    project_id: str,
    system_id: str,
    cluster_id: str,
    trajectories: List[UploadFile] = File(...),
    state_ids: str = Form(...),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    parsed_state_ids = _parse_state_ids(state_ids)
    if not parsed_state_ids:
        raise HTTPException(status_code=400, detail="Provide state_ids for uploaded trajectories.")
    if len(parsed_state_ids) != len(trajectories):
        raise HTTPException(
            status_code=400,
            detail="State selection count must match the number of uploaded trajectories.",
        )

    rel_path = entry.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="Cluster NPZ path missing.")
    cluster_path = Path(rel_path)
    if not cluster_path.is_absolute():
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not cluster_path.exists():
        raise HTTPException(status_code=404, detail="Cluster NPZ file is missing on disk.")

    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    upload_root = cluster_dirs["cluster_dir"] / "backmapping_uploads" / str(uuid.uuid4())
    upload_root.mkdir(parents=True, exist_ok=True)
    out_path = upload_root / "backmapping.npz"

    overrides: Dict[str, Path] = {}
    try:
        for state_id, upload in zip(parsed_state_ids, trajectories):
            filename = upload.filename or "traj"
            dest = upload_root / f"{state_id}_{filename}"
            await stream_upload(upload, dest)
            overrides[str(state_id)] = dest

        await run_in_threadpool(
            build_backmapping_npz,
            project_id,
            system_id,
            cluster_path,
            out_path,
            trajectory_overrides=overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build backmapping NPZ: {exc}") from exc
    finally:
        for path in overrides.values():
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    return FileResponse(out_path, filename=out_path.name, media_type="application/octet-stream")


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/upload",
    summary="Upload a locally generated cluster NPZ and map it to macro-states",
)
async def upload_metastable_cluster_npz(
    project_id: str,
    system_id: str,
    cluster_npz: UploadFile = File(...),
    state_ids: str = Form(...),
    name: Optional[str] = Form(None),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    parsed_state_ids = _parse_state_ids(state_ids)
    if parsed_state_ids:
        for sid in parsed_state_ids:
            if sid not in (system_meta.states or {}):
                raise HTTPException(status_code=400, detail=f"Unknown macro-state '{sid}'.")
            state = system_meta.states[sid]
            if not state.descriptor_file:
                raise HTTPException(status_code=400, detail=f"State '{sid}' is missing descriptors.")

    filename = cluster_npz.filename or "cluster_upload.npz"
    if not filename.lower().endswith(".npz"):
        raise HTTPException(status_code=400, detail="Cluster upload must be an .npz file.")

    cluster_id = str(uuid.uuid4())
    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    cluster_dir = cluster_dirs["cluster_dir"]
    tmp_path = cluster_dir / "cluster.npz"
    await stream_upload(cluster_npz, tmp_path)

    try:
        data = np.load(tmp_path, allow_pickle=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded NPZ: {exc}") from exc

    payload: Dict[str, Any] = {}
    for key in data.files:
        payload[key] = data[key]

    meta_raw = payload.get("metadata_json")
    if meta_raw is None:
        raise HTTPException(status_code=400, detail="Uploaded NPZ is missing metadata_json.")
    try:
        meta = _decode_metadata_json(meta_raw)
    except Exception as exc:
        meta_preview = None
        try:
            meta_preview = str(meta_raw)
        except Exception:
            meta_preview = "<unprintable>"
        logger = logging.getLogger("cluster_upload")
        logger.error("metadata_json decode failed: %s", exc, exc_info=True)
        logger.error("metadata_json type=%s preview=%s", type(meta_raw), meta_preview[:500])
        if isinstance(meta_raw, np.ndarray):
            try:
                logger.error("metadata_json ndarray dtype=%s shape=%s", meta_raw.dtype, meta_raw.shape)
            except Exception:
                pass
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded NPZ metadata_json is invalid. type={type(meta_raw).__name__} preview={meta_preview[:200]}",
        ) from exc

    predictions = meta.get("predictions") or {}
    state_labels_meta = meta.get("state_labels") or {}
    old_state_ids = list(state_labels_meta.keys())
    if not old_state_ids:
        old_state_ids = [
            key.split(":", 1)[1]
            for key in predictions.keys()
            if isinstance(key, str) and key.startswith("state:")
        ]

    mapping: Dict[str, str] = {}
    local_labels = state_labels_meta or {}
    if parsed_state_ids:
        selected_lookup = {sid: system_meta.states[sid].name for sid in parsed_state_ids}
    else:
        selected_lookup = {
            sid: state.name
            for sid, state in (system_meta.states or {}).items()
            if state.descriptor_file
        }
    fallback_lookup = {
        sid: state.name
        for sid, state in (system_meta.states or {}).items()
        if state.descriptor_file
    }
    used_ids: set[str] = set()
    missing_matches: List[str] = []
    def _normalize_label(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _candidate_labels(raw: str) -> List[str]:
        raw = raw.strip()
        candidates = [raw]
        if "-" in raw:
            candidates.append(raw.split("-")[-1].strip())
        if "+" in raw:
            candidates.append(raw.split("+")[-1].strip())
        return candidates

    normalized_lookup = {
        sid: (_normalize_label(name), _normalize_label(sid))
        for sid, name in selected_lookup.items()
    }
    normalized_fallback = {
        sid: (_normalize_label(name), _normalize_label(sid))
        for sid, name in fallback_lookup.items()
    }

    for old_id in old_state_ids:
        label = local_labels.get(str(old_id)) or str(old_id)
        matched = None
        label_candidates = _candidate_labels(label)
        label_candidates = [
            c.replace("_descriptors", "").replace("descriptors", "").strip()
            for c in label_candidates
            if c
        ]
        normalized_candidates = {_normalize_label(c) for c in label_candidates if c}
        for sid, (norm_name, norm_id) in normalized_lookup.items():
            if sid in used_ids:
                continue
            if norm_name in normalized_candidates or norm_id in normalized_candidates:
                matched = sid
                break
        if matched is None:
            for sid, (norm_name, norm_id) in normalized_fallback.items():
                if sid in used_ids:
                    continue
                if norm_name in normalized_candidates or norm_id in normalized_candidates:
                    matched = sid
                    break
        if matched is None:
            missing_matches.append(label)
            continue
        mapping[str(old_id)] = matched
        used_ids.add(matched)

    if missing_matches:
        if parsed_state_ids and len(parsed_state_ids) == len(old_state_ids):
            # Fall back to selection order when labels do not match.
            mapping = dict(zip([str(s) for s in old_state_ids], parsed_state_ids))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Could not match NPZ states to system states: {', '.join(missing_matches)}",
            )
    new_predictions: Dict[str, Any] = {}

    ordered_new_ids: List[str] = []
    for old_id in old_state_ids:
        new_id = mapping.get(str(old_id))
        if not new_id:
            continue
        ordered_new_ids.append(new_id)
        entry = predictions.get(f"state:{old_id}")
        if not isinstance(entry, dict):
            continue
        halo_key = entry.get("labels_halo")
        assigned_key = entry.get("labels_assigned")
        frame_key = entry.get("frame_indices")
        if not halo_key or halo_key not in payload:
            continue
        new_slug = _slug(str(new_id))
        new_halo_key = f"state__{new_slug}__labels_halo"
        new_assigned_key = f"state__{new_slug}__labels_assigned"
        new_frame_key = f"state__{new_slug}__frame_indices"
        payload[new_halo_key] = payload[halo_key]
        if assigned_key and assigned_key in payload:
            payload[new_assigned_key] = payload[assigned_key]
        if frame_key and frame_key in payload:
            payload[new_frame_key] = payload[frame_key]
        new_predictions[f"state:{new_id}"] = {
            "type": "macro",
            "labels_halo": new_halo_key,
            "labels_assigned": new_assigned_key,
            "frame_indices": new_frame_key,
            "frame_count": entry.get("frame_count", None),
        }

    meta["selected_state_ids"] = parsed_state_ids
    meta["selected_metastable_ids"] = parsed_state_ids
    meta["analysis_mode"] = "macro"
    meta["state_labels"] = {
        sid: (system_meta.states[sid].name if sid in system_meta.states else sid)
        for sid in ordered_new_ids
    }
    meta["metastable_labels"] = {}
    meta["predictions"] = new_predictions

    halo_summary = meta.get("halo_summary") or {}
    npz_keys = (halo_summary or {}).get("npz_keys") or {}
    ids_key = npz_keys.get("condition_ids")
    labels_key = npz_keys.get("condition_labels")
    types_key = npz_keys.get("condition_types")
    if ids_key in payload:
        payload[ids_key] = np.array([f"state:{sid}" for sid in ordered_new_ids], dtype=str)
    if labels_key in payload:
        payload[labels_key] = np.array(
            [meta["state_labels"].get(sid, sid) for sid in ordered_new_ids],
            dtype=str,
        )
    if types_key in payload:
        payload[types_key] = np.array(["macro"] * len(ordered_new_ids), dtype=str)

    if "merged__frame_state_ids" in payload:
        raw_ids = payload["merged__frame_state_ids"]
        mapped_ids = [mapping.get(str(val), str(val)) for val in raw_ids]
        payload["merged__frame_state_ids"] = np.array(mapped_ids, dtype=str)
    if "merged__frame_metastable_ids" in payload:
        raw_meta_ids = payload["merged__frame_metastable_ids"]
        mapped_meta_ids = [mapping.get(str(val), str(val)) for val in raw_meta_ids]
        payload["merged__frame_metastable_ids"] = np.array(mapped_meta_ids, dtype=str)

    payload["metadata_json"] = np.array(json.dumps(meta))

    final_name = name.strip() if isinstance(name, str) and name.strip() else None
    out_path = cluster_dir / "cluster.npz"
    np.savez_compressed(out_path, **payload)

    assignments = assign_cluster_labels_to_states(
        out_path,
        project_id,
        system_id,
        output_dir=cluster_dirs["samples_dir"],
    )
    update_cluster_metadata_with_assignments(out_path, assignments)

    cluster_entry = {
        "cluster_id": cluster_id,
        "name": final_name,
        "status": "finished",
        "progress": 100,
        "status_message": "Uploaded",
        "job_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "path": str(out_path.relative_to(cluster_dirs["system_dir"])),
        "state_ids": parsed_state_ids,
        "metastable_ids": parsed_state_ids,
        "max_cluster_frames": meta.get("cluster_params", {}).get("max_cluster_frames"),
        "random_state": meta.get("cluster_params", {}).get("random_state"),
        "generated_at": meta.get("generated_at"),
        "contact_edge_count": meta.get("contact_edge_count"),
        "cluster_algorithm": "density_peaks",
        "algorithm_params": {
            "density_maxk": meta.get("cluster_params", {}).get("density_maxk"),
            "density_z": meta.get("cluster_params", {}).get("density_z"),
            "max_cluster_frames": meta.get("cluster_params", {}).get("max_cluster_frames"),
        },
        "assigned_state_paths": assignments.get("assigned_state_paths", {}),
        "assigned_metastable_paths": assignments.get("assigned_metastable_paths", {}),
        "samples": assignments.get("samples", []),
        "potts_models": [],
    }
    system_meta.metastable_clusters = (system_meta.metastable_clusters or []) + [cluster_entry]
    project_store.save_system(system_meta)

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {"status": "uploaded", "cluster_id": cluster_id}


@router.patch(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}",
    summary="Rename a saved metastable cluster NPZ",
)
async def rename_metastable_cluster_npz(project_id: str, system_id: str, cluster_id: str, payload: Dict[str, Any]):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    name = (payload or {}).get("name")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="Cluster name is required.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    entry["name"] = name.strip()
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    return {"status": "renamed", "cluster_id": cluster_id, "name": entry["name"]}


@router.delete(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}",
    summary="Delete a saved metastable cluster NPZ",
)
async def delete_metastable_cluster_npz(project_id: str, system_id: str, cluster_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro-states before deleting cluster NPZ.")
    if not getattr(system_meta, "metastable_locked", False) and getattr(system_meta, "analysis_mode", None) != "macro":
        raise HTTPException(status_code=400, detail="Lock metastable states or confirm macro-only before deleting cluster NPZ.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    rel_path = entry.get("path")
    if rel_path:
        abs_path = project_store.resolve_path(project_id, system_id, rel_path)
        try:
            abs_path.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        if cluster_dirs["cluster_dir"].exists():
            shutil.rmtree(cluster_dirs["cluster_dir"], ignore_errors=True)
    except Exception:
        pass

    system_meta.metastable_clusters = [c for c in clusters if c.get("cluster_id") != cluster_id]
    project_store.save_system(system_meta)
    return {"status": "deleted", "cluster_id": cluster_id}


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/evaluate_state",
    summary="Evaluate a new state against an existing cluster (creates MD sample)",
)
async def evaluate_state_against_cluster(
    project_id: str,
    system_id: str,
    cluster_id: str,
    payload: Dict[str, Any],
):
    state_id = (payload or {}).get("state_id")
    if not isinstance(state_id, str) or not state_id.strip():
        raise HTTPException(status_code=400, detail="state_id is required.")
    state_id = state_id.strip()

    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    try:
        sample_entry = evaluate_state_with_models(project_id, system_id, cluster_id, state_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []
    samples.append(sample_entry)
    entry["samples"] = samples
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    return {"status": "ok", "cluster_id": cluster_id, "sample": sample_entry}


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/potts_models",
    summary="Upload a Potts model NPZ for a cluster",
)
async def upload_potts_model_npz(
    project_id: str,
    system_id: str,
    cluster_id: str,
    model: UploadFile = File(...),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = dirs["system_dir"]
    filename = model.filename or f"{cluster_id}_potts_model.npz"
    model_id = str(uuid.uuid4())
    model_dir = dirs["potts_models_dir"] / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / filename
    await stream_upload(model, dest_path)
    rel_path = str(dest_path.relative_to(system_dir))
    display_name = Path(filename).stem

    models = entry.get("potts_models")
    if not isinstance(models, list):
        models = []
    models.append(
        {
            "model_id": model_id,
            "name": display_name,
            "path": rel_path,
            "created_at": datetime.utcnow().isoformat(),
            "source": "upload",
            "params": {},
        }
    )
    entry["potts_models"] = models
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    return {"status": "uploaded", "cluster_id": cluster_id, "model_id": model_id, "path": rel_path}


@router.patch(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/potts_models/{model_id}",
    summary="Rename a Potts model NPZ",
)
async def rename_potts_model_npz(
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_id: str,
    payload: Dict[str, Any],
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    name = (payload or {}).get("name")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="Potts model name is required.")
    base = Path(name.strip()).name
    if base.lower().endswith(".npz"):
        base = base[:-4]
    base = base.strip()
    if not base:
        raise HTTPException(status_code=400, detail="Potts model name is required.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    models = entry.get("potts_models") or []
    model_entry = next((m for m in models if m.get("model_id") == model_id), None)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Potts model not available.")

    rel_path = model_entry.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="Potts model path missing.")

    dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = dirs["system_dir"]
    abs_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Potts model file is missing on disk.")

    new_filename = f"{base}.npz"
    new_path = abs_path.with_name(new_filename)
    if new_path.resolve() != abs_path.resolve():
        if new_path.exists():
            raise HTTPException(status_code=409, detail="Potts model name already exists.")
        abs_path.rename(new_path)
        rel_path = str(new_path.relative_to(system_dir))
        model_entry["path"] = rel_path

    model_entry["name"] = base
    model_entry["updated_at"] = datetime.utcnow().isoformat()
    entry["potts_models"] = models
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    return {"status": "renamed", "cluster_id": cluster_id, "model_id": model_id, "name": base}


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/potts_models/{model_id}",
    summary="Download a Potts model NPZ for a cluster",
)
async def download_potts_model_npz(project_id: str, system_id: str, cluster_id: str, model_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")
    models = entry.get("potts_models") or []
    model_entry = next((m for m in models if m.get("model_id") == model_id), None)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Potts model not available.")
    rel_path = model_entry.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="Potts model path missing.")
    abs_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Potts model file missing on disk.")
    return FileResponse(abs_path, filename=abs_path.name, media_type="application/octet-stream")


@router.delete(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/potts_models/{model_id}",
    summary="Delete a Potts model NPZ for a cluster",
)
async def delete_potts_model_npz(project_id: str, system_id: str, cluster_id: str, model_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")
    models = entry.get("potts_models") or []
    model_entry = next((m for m in models if m.get("model_id") == model_id), None)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Potts model not available.")
    rel_path = model_entry.get("path")
    if rel_path:
        abs_path = project_store.resolve_path(project_id, system_id, rel_path)
        try:
            abs_path.unlink(missing_ok=True)
        except Exception:
            pass
    entry["potts_models"] = [m for m in models if m.get("model_id") != model_id]
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    _remove_simulation_results(
        project_id,
        system_id,
        cluster_id,
        model_id=model_id,
    )
    return {"status": "deleted", "cluster_id": cluster_id, "model_id": model_id}
