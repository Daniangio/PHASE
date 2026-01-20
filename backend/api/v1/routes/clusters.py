import logging
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.api.v1.common import get_queue, project_store
from backend.services.metastable_clusters import generate_metastable_cluster_npz
from backend.tasks import run_cluster_job


router = APIRouter()


def _parse_cluster_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metastable_ids_raw = (payload or {}).get("metastable_ids") or []
    cluster_name = (payload or {}).get("cluster_name")
    if not isinstance(metastable_ids_raw, list):
        raise HTTPException(status_code=400, detail="metastable_ids must be a list.")
    metastable_ids = [str(mid).strip() for mid in metastable_ids_raw if str(mid).strip()]
    if not metastable_ids:
        raise HTTPException(status_code=400, detail="Provide at least one metastable_id.")

    try:
        max_clusters = int((payload or {}).get("max_clusters_per_residue", 6))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_clusters_per_residue must be an integer.")
    if max_clusters < 1:
        raise HTTPException(status_code=400, detail="max_clusters_per_residue must be >= 1.")
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
    try:
        contact_cutoff = float((payload or {}).get("contact_cutoff", 10.0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="contact_cutoff must be a number.")
    contact_atom_mode = str(
        (payload or {}).get("contact_atom_mode", payload.get("contact_mode", "CA") if isinstance(payload, dict) else "CA")
        or "CA"
    ).upper()
    if contact_atom_mode not in {"CA", "CM"}:
        raise HTTPException(status_code=400, detail="contact_atom_mode must be 'CA' or 'CM'.")

    algo_raw = (payload or {}).get("cluster_algorithm", "density_peaks")
    cluster_algorithm = str(algo_raw or "density_peaks").lower()
    if cluster_algorithm not in {"tomato", "density_peaks", "dbscan", "kmeans", "hierarchical"}:
        raise HTTPException(
            status_code=400,
            detail="cluster_algorithm must be tomato, density_peaks, dbscan, kmeans, or hierarchical.",
        )
    algo_params = (payload or {}).get("algorithm_params", {}) or {}
    try:
        dbscan_eps = float(algo_params.get("eps", (payload or {}).get("dbscan_eps", 0.5)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="dbscan eps must be a number.")
    try:
        dbscan_min_samples = int(algo_params.get("min_samples", (payload or {}).get("dbscan_min_samples", 5)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="dbscan min_samples must be an integer.")
    try:
        hierarchical_n_clusters = algo_params.get("n_clusters")
        if hierarchical_n_clusters is not None:
            hierarchical_n_clusters = int(hierarchical_n_clusters)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="hierarchical n_clusters must be an integer.")
    hierarchical_linkage = str(
        algo_params.get("linkage", (payload or {}).get("hierarchical_linkage", "ward")) or "ward"
    ).lower()
    density_z_raw = algo_params.get("Z")
    try:
        density_maxk = algo_params.get("maxk", (payload or {}).get("density_maxk"))
        if density_maxk is not None:
            density_maxk = int(density_maxk)
        else:
            density_maxk = 100
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="density maxk must be an integer >=1.")
    density_z = None
    if cluster_algorithm == "density_peaks":
        if density_z_raw is None:
            density_z = "auto"
        elif isinstance(density_z_raw, str) and density_z_raw.lower() == "auto":
            density_z = "auto"
        else:
            try:
                density_z = float(density_z_raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="density_peaks Z must be a number or 'auto'.")
    try:
        tomato_k = int(algo_params.get("k_neighbors", (payload or {}).get("tomato_k", 15)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="tomato k_neighbors must be an integer.")
    tomato_tau_raw = algo_params.get("tau", (payload or {}).get("tomato_tau", "auto"))
    if tomato_tau_raw is None:
        tomato_tau = "auto"
    elif isinstance(tomato_tau_raw, str) and tomato_tau_raw.lower() == "auto":
        tomato_tau = "auto"
    else:
        try:
            tomato_tau = float(tomato_tau_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="tomato tau must be a number or 'auto'.")
    try:
        tomato_k_max = int(algo_params.get("k_max", (payload or {}).get("tomato_k_max", max_clusters)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="tomato k_max must be an integer.")

    return {
        "metastable_ids": metastable_ids,
        "cluster_name": cluster_name,
        "max_clusters_per_residue": max_clusters,
        "max_cluster_frames": max_cluster_frames,
        "random_state": random_state,
        "contact_cutoff": contact_cutoff,
        "contact_atom_mode": contact_atom_mode,
        "cluster_algorithm": cluster_algorithm,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "hierarchical_n_clusters": hierarchical_n_clusters,
        "hierarchical_linkage": hierarchical_linkage,
        "density_maxk": density_maxk,
        "density_z": density_z,
        "tomato_k": tomato_k,
        "tomato_tau": tomato_tau,
        "tomato_k_max": tomato_k_max,
    }


def _build_algorithm_params(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "eps": parsed["dbscan_eps"],
        "min_samples": parsed["dbscan_min_samples"],
        "n_clusters": parsed["hierarchical_n_clusters"],
        "linkage": parsed["hierarchical_linkage"],
        "k_neighbors": parsed["tomato_k"],
        "tau": parsed["tomato_tau"],
        "k_max": parsed["tomato_k_max"],
        "density_maxk": parsed["density_maxk"],
        "density_z": parsed["density_z"],
        "max_cluster_frames": parsed["max_cluster_frames"],
    }


def _build_cluster_entry(parsed: Dict[str, Any], cluster_id: str, status: str, progress: int, status_message: str):
    return {
        "cluster_id": cluster_id,
        "name": parsed["cluster_name"].strip()
        if isinstance(parsed["cluster_name"], str) and parsed["cluster_name"].strip()
        else None,
        "status": status,
        "progress": progress,
        "status_message": status_message,
        "job_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "path": None,
        "metastable_ids": parsed["metastable_ids"],
        "max_clusters_per_residue": parsed["max_clusters_per_residue"],
        "max_cluster_frames": parsed["max_cluster_frames"],
        "random_state": parsed["random_state"],
        "generated_at": None,
        "contact_cutoff": parsed["contact_cutoff"],
        "contact_atom_mode": parsed["contact_atom_mode"],
        "contact_edge_count": None,
        "cluster_algorithm": parsed["cluster_algorithm"],
        "algorithm_params": _build_algorithm_params(parsed),
    }


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

    try:
        if logger:
            logger.error(
                "[cluster_build] project=%s system=%s meta_ids=%s algo=%s params=%s",
                project_id,
                system_id,
                parsed["metastable_ids"],
                parsed["cluster_algorithm"],
                {
                    "max_clusters": parsed["max_clusters_per_residue"],
                    "random_state": parsed["random_state"],
                    "contact_cutoff": parsed["contact_cutoff"],
                    "contact_atom_mode": parsed["contact_atom_mode"],
                    "dbscan_eps": parsed["dbscan_eps"],
                    "dbscan_min_samples": parsed["dbscan_min_samples"],
                    "hierarchical_n_clusters": parsed["hierarchical_n_clusters"],
                    "hierarchical_linkage": parsed["hierarchical_linkage"],
                    "density_peaks_maxk": parsed["density_maxk"],
                    "density_peaks_Z": parsed["density_z"],
                    "tomato_k": parsed["tomato_k"],
                    "tomato_tau": parsed["tomato_tau"],
                },
            )
        npz_path, meta = await run_in_threadpool(
            generate_metastable_cluster_npz,
            project_id,
            system_id,
            parsed["metastable_ids"],
            max_clusters_per_residue=parsed["max_clusters_per_residue"],
            max_cluster_frames=parsed["max_cluster_frames"],
            random_state=parsed["random_state"],
            contact_cutoff=parsed["contact_cutoff"],
            contact_atom_mode=parsed["contact_atom_mode"],
            cluster_algorithm=parsed["cluster_algorithm"],
            dbscan_eps=parsed["dbscan_eps"],
            dbscan_min_samples=parsed["dbscan_min_samples"],
            hierarchical_n_clusters=parsed["hierarchical_n_clusters"],
            hierarchical_linkage=parsed["hierarchical_linkage"],
            density_maxk=parsed["density_maxk"],
            density_z=parsed["density_z"],
            tomato_k=parsed["tomato_k"],
            tomato_tau=parsed["tomato_tau"],
            tomato_k_max=parsed["tomato_k_max"],
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

    cluster_id = str(uuid.uuid4())
    cluster_entry = _build_cluster_entry(parsed, cluster_id, "finished", 100, "Complete")
    cluster_entry.update(
        {
            "path": rel_path,
            "generated_at": meta.get("generated_at") if isinstance(meta, dict) else None,
            "contact_edge_count": meta.get("contact_edge_count") if isinstance(meta, dict) else None,
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

    system_meta.metastable_clusters = [c for c in clusters if c.get("cluster_id") != cluster_id]
    project_store.save_system(system_meta)
    return {"status": "deleted", "cluster_id": cluster_id}
