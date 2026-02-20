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
from backend.api.v1.schemas import LambdaPottsModelCreateRequest, UiSetupUpsertRequest
from phase.workflows.clustering import (
    generate_metastable_cluster_npz,
    build_md_eval_samples_for_cluster,
    build_cluster_entry,
    build_cluster_output_path,
    list_cluster_patches,
    create_cluster_residue_patch,
    discard_cluster_residue_patch,
    confirm_cluster_residue_patch,
    _slug,
    evaluate_state_with_models,
)
from phase.workflows.backmapping import build_backmapping_npz
from backend.tasks import run_cluster_job, run_backmapping_job
from phase.potts.potts_model import interpolate_potts_models, load_potts_model, save_potts_model, zero_sum_gauge_model


router = APIRouter()

def _convert_nan_to_none(obj: Any):
    """
    Recursively converts NaN/Inf values into None so API responses remain valid JSON.
    """
    if isinstance(obj, dict):
        return {k: _convert_nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_nan_to_none(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _convert_nan_to_none(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and not np.isfinite(val):
            return None
        return val
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


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


def _replace_md_samples(existing_samples: Any, new_md_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    if isinstance(existing_samples, list):
        for entry in existing_samples:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("type") or "") == "md_eval":
                continue
            kept.append(entry)
    kept.extend(new_md_samples or [])
    return kept


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


def _ui_setups_dir(project_id: str, system_id: str, cluster_id: str) -> Path:
    dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    setup_dir = dirs["cluster_dir"] / "ui_setups"
    setup_dir.mkdir(parents=True, exist_ok=True)
    return setup_dir


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/ui_setups",
    summary="List persisted UI setups for a cluster",
)
async def list_cluster_ui_setups(
    project_id: str,
    system_id: str,
    cluster_id: str,
    setup_type: Optional[str] = None,
    page: Optional[str] = None,
):
    try:
        project_store.get_cluster_entry(project_id, system_id, cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    setup_dir = _ui_setups_dir(project_id, system_id, cluster_id)
    entries: List[Dict[str, Any]] = []
    for path in sorted(setup_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if setup_type and str(payload.get("setup_type") or "") != str(setup_type):
            continue
        if page and str(payload.get("page") or "") != str(page):
            continue
        payload.setdefault("setup_id", path.stem)
        payload.setdefault("name", path.stem)
        payload.setdefault("setup_type", None)
        payload.setdefault("page", None)
        payload.setdefault("created_at", None)
        payload.setdefault("updated_at", None)
        payload.setdefault("payload", {})
        entries.append(payload)
    entries.sort(key=lambda x: (str(x.get("name") or "").lower(), str(x.get("setup_id") or "")))
    return {"setups": entries}


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/ui_setups",
    summary="Create or update a persisted UI setup for a cluster",
)
async def upsert_cluster_ui_setup(
    project_id: str,
    system_id: str,
    cluster_id: str,
    payload: UiSetupUpsertRequest,
):
    try:
        project_store.get_cluster_entry(project_id, system_id, cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    name = str(payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required.")
    setup_type = str(payload.setup_type or "").strip()
    if not setup_type:
        raise HTTPException(status_code=400, detail="setup_type is required.")
    page_name = str(payload.page or "").strip() or None

    setup_id = str(payload.setup_id or "").strip() or str(uuid.uuid4())
    setup_dir = _ui_setups_dir(project_id, system_id, cluster_id)
    out_path = setup_dir / f"{setup_id}.json"

    now = datetime.utcnow().isoformat()
    created_at = now
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            created_at = str(old.get("created_at") or created_at)
        except Exception:
            created_at = now

    obj = {
        "setup_id": setup_id,
        "name": name,
        "setup_type": setup_type,
        "page": page_name,
        "created_at": created_at,
        "updated_at": now,
        "payload": payload.payload if isinstance(payload.payload, dict) else {},
    }
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return obj


@router.delete(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/ui_setups/{setup_id}",
    summary="Delete a persisted UI setup for a cluster",
)
async def delete_cluster_ui_setup(
    project_id: str,
    system_id: str,
    cluster_id: str,
    setup_id: str,
):
    try:
        project_store.get_cluster_entry(project_id, system_id, cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    sid = str(setup_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="setup_id is required.")
    setup_dir = _ui_setups_dir(project_id, system_id, cluster_id)
    path = setup_dir / f"{sid}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Setup '{sid}' not found.")
    try:
        path.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete setup '{sid}': {exc}") from exc
    return {"status": "deleted", "setup_id": sid}


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
        # Legacy metadata sometimes stored paths relative to the cluster directory.
        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        alt = cluster_dirs["cluster_dir"] / summary_rel
        if alt.exists():
            summary_path = alt
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Sample summary NPZ not found on disk.")

    try:
        with np.load(summary_path, allow_pickle=True) as data:
            # New sampling runs store a minimal `sample.npz` (labels only). The legacy visualization
            # endpoint expects a full run_summary bundle; return an explicit error instead of
            # silently serving empty arrays.
            if "labels" in data and "js_md_sample" not in data and "X_gibbs" not in data and "X_sa" not in data:
                raise HTTPException(
                    status_code=409,
                    detail="This sample contains a minimal sample.npz (labels only). Run a Potts analysis job to generate analyses for visualization.",
                )

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

    return _convert_nan_to_none(payload)


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/potts/cluster_info",
    summary="Load cluster info needed for Potts sample visualization (residues, K, edges).",
)
async def get_potts_cluster_info(
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_id: str | None = None,
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    entry = get_cluster_entry(system_meta, cluster_id)
    rel_path = entry.get("path") if isinstance(entry, dict) else None
    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    cluster_path = cluster_dirs["cluster_dir"] / "cluster.npz"
    if not cluster_path.exists() and rel_path:
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not cluster_path.exists():
        raise HTTPException(status_code=404, detail="Cluster NPZ not found on disk.")

    residue_keys: list[str] = []
    cluster_counts: list[int] = []
    edges: list[list[int]] = []
    edges_source = "none"
    model_name: str | None = None

    try:
        with np.load(cluster_path, allow_pickle=True) as data:
            residue_keys = data["residue_keys"].tolist() if "residue_keys" in data else []
            if "cluster_counts" in data:
                cluster_counts = data["cluster_counts"].tolist()
            elif "merged__cluster_counts" in data:
                cluster_counts = data["merged__cluster_counts"].tolist()
            else:
                cluster_counts = []
            if "contact_edge_index" in data:
                edge_index = np.asarray(data["contact_edge_index"], dtype=int)
                if edge_index.ndim == 2 and edge_index.shape[0] == 2:
                    edges = edge_index.T.tolist()
            elif "edges" in data:
                edges = np.asarray(data["edges"], dtype=int).tolist()
            if edges:
                edges_source = "cluster"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load cluster NPZ: {exc}") from exc

    # If a model_id is provided, use the Potts model edges (these define the "meaningful" edge set).
    if isinstance(model_id, str) and model_id.strip():
        model_id = model_id.strip()
        models = project_store.list_potts_models(project_id, system_id, cluster_id)
        model_entry = next((m for m in models if m.get("model_id") == model_id), None)
        if not model_entry or not model_entry.get("path"):
            raise HTTPException(status_code=404, detail=f"Potts model '{model_id}' not found.")
        model_name = model_entry.get("name")
        model_path = project_store.resolve_path(project_id, system_id, str(model_entry.get("path")))
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Potts model NPZ not found on disk.")
        try:
            with np.load(model_path, allow_pickle=False) as data:
                if "edges" in data:
                    edges = np.asarray(data["edges"], dtype=int).tolist()
                    edges_source = "potts_model"
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load Potts model NPZ: {exc}") from exc

    return {
        "cluster_id": cluster_id,
        "n_residues": int(len(residue_keys)),
        "n_edges": int(len(edges)),
        "residue_keys": residue_keys,
        "cluster_counts": cluster_counts,
        "edges": edges,
        "edges_source": edges_source,
        "model_id": model_id,
        "model_name": model_name,
    }


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/patches",
    summary="List preview clustering patches stored for a cluster.",
)
async def get_cluster_patches(
    project_id: str,
    system_id: str,
    cluster_id: str,
):
    try:
        project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")
    try:
        result = await run_in_threadpool(
            list_cluster_patches,
            project_id,
            system_id,
            cluster_id,
            store=project_store,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list cluster patches: {exc}") from exc
    return _convert_nan_to_none(result)


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/patches",
    summary="Create a preview cluster patch on selected residues (hierarchical + frozen GMM).",
)
async def create_cluster_patch(
    project_id: str,
    system_id: str,
    cluster_id: str,
    payload: Dict[str, Any],
):
    logger = logging.getLogger("cluster_patch")
    try:
        project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    residue_keys = payload.get("residue_keys")
    if isinstance(residue_keys, str):
        residue_keys = [v.strip() for v in residue_keys.split(",") if v.strip()]
    residue_indices = payload.get("residue_indices")
    if isinstance(residue_indices, str):
        residue_indices = [int(v.strip()) for v in residue_indices.split(",") if v.strip()]
    if not isinstance(residue_keys, list):
        residue_keys = None
    if not isinstance(residue_indices, list):
        residue_indices = None

    n_clusters = payload.get("n_clusters")
    if n_clusters is not None:
        try:
            n_clusters = int(n_clusters)
        except Exception:
            raise HTTPException(status_code=400, detail="n_clusters must be an integer.")
    cluster_selection_mode = str(payload.get("cluster_selection_mode") or "maxclust").strip().lower()
    if cluster_selection_mode not in {"maxclust", "inconsistent"}:
        raise HTTPException(status_code=400, detail="cluster_selection_mode must be one of: maxclust, inconsistent.")
    inconsistent_threshold = payload.get("inconsistent_threshold")
    if inconsistent_threshold is not None:
        try:
            inconsistent_threshold = float(inconsistent_threshold)
        except Exception:
            raise HTTPException(status_code=400, detail="inconsistent_threshold must be numeric.")
    inconsistent_depth = payload.get("inconsistent_depth", 2)
    try:
        inconsistent_depth = int(inconsistent_depth)
    except Exception:
        raise HTTPException(status_code=400, detail="inconsistent_depth must be an integer.")
    if inconsistent_depth < 1:
        raise HTTPException(status_code=400, detail="inconsistent_depth must be >= 1.")
    if cluster_selection_mode == "inconsistent" and inconsistent_threshold is None:
        raise HTTPException(
            status_code=400,
            detail="inconsistent_threshold is required when cluster_selection_mode='inconsistent'.",
        )
    linkage_method = str(payload.get("linkage_method") or "ward")
    covariance_type = str(payload.get("covariance_type") or "full")
    reg_covar = float(payload.get("reg_covar", 1e-5))
    halo_percentile = float(payload.get("halo_percentile", 5.0))
    max_cluster_frames = payload.get("max_cluster_frames")
    if max_cluster_frames is not None:
        try:
            max_cluster_frames = int(max_cluster_frames)
        except Exception:
            raise HTTPException(status_code=400, detail="max_cluster_frames must be an integer.")
    patch_name = payload.get("name")
    if patch_name is not None and not isinstance(patch_name, str):
        patch_name = str(patch_name)

    logger.info(
        "[cluster_patch] create start project=%s system=%s cluster=%s residues=%s mode=%s n_clusters=%s inconsistent_threshold=%s max_cluster_frames=%s",
        project_id,
        system_id,
        cluster_id,
        len(residue_keys or residue_indices or []),
        cluster_selection_mode,
        n_clusters,
        inconsistent_threshold,
        max_cluster_frames,
    )
    try:
        result = await run_in_threadpool(
            create_cluster_residue_patch,
            project_id,
            system_id,
            cluster_id,
            residue_indices=residue_indices,
            residue_keys=residue_keys,
            n_clusters=n_clusters,
            cluster_selection_mode=cluster_selection_mode,
            inconsistent_threshold=inconsistent_threshold,
            inconsistent_depth=inconsistent_depth,
            linkage_method=linkage_method,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            halo_percentile=halo_percentile,
            max_cluster_frames=max_cluster_frames,
            patch_name=patch_name,
            store=project_store,
        )
        logger.info(
            "[cluster_patch] create done project=%s system=%s cluster=%s patch_id=%s",
            project_id,
            system_id,
            cluster_id,
            result.get("patch_id"),
        )
    except FileNotFoundError as exc:
        logger.error("[cluster_patch] create file not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error("[cluster_patch] create validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("[cluster_patch] create failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create cluster patch: {exc}") from exc
    return _convert_nan_to_none(result)


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/patches/{patch_id}/confirm",
    summary="Confirm a preview patch, swap labels, and recompute MD cluster memberships.",
)
async def confirm_cluster_patch(
    project_id: str,
    system_id: str,
    cluster_id: str,
    patch_id: str,
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    entry = get_cluster_entry(system_meta, cluster_id)
    if not isinstance(entry, dict):
        raise HTTPException(status_code=404, detail="Cluster not found.")

    try:
        result = await run_in_threadpool(
            confirm_cluster_residue_patch,
            project_id,
            system_id,
            cluster_id,
            patch_id=patch_id,
            recompute_assignments=True,
            store=project_store,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to confirm cluster patch: {exc}") from exc

    assignments = result.get("assignments") or {}
    if isinstance(assignments, dict):
        new_md_samples = [s for s in (assignments.get("samples") or []) if isinstance(s, dict)]
        entry["samples"] = _replace_md_samples(entry.get("samples"), new_md_samples)
        entry["updated_at"] = datetime.utcnow().isoformat()
        project_store.save_system(system_meta)

    return _convert_nan_to_none(result)


@router.delete(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/patches/{patch_id}",
    summary="Discard a preview cluster patch.",
)
async def discard_cluster_patch(
    project_id: str,
    system_id: str,
    cluster_id: str,
    patch_id: str,
):
    try:
        project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")
    try:
        result = await run_in_threadpool(
            discard_cluster_residue_patch,
            project_id,
            system_id,
            cluster_id,
            patch_id=patch_id,
            store=project_store,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to discard cluster patch: {exc}") from exc
    return _convert_nan_to_none(result)


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/analyses",
    summary="List derived Potts analyses stored under clusters/<cluster_id>/analyses/.",
)
async def list_cluster_analyses(
    project_id: str,
    system_id: str,
    cluster_id: str,
    analysis_type: str | None = None,
):
    try:
        project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    analyses_root = cluster_dirs["cluster_dir"] / "analyses"
    if not analyses_root.exists():
        return {"cluster_id": cluster_id, "analyses": []}

    wanted = analysis_type.strip().lower() if isinstance(analysis_type, str) and analysis_type.strip() else None

    analyses: list[dict[str, Any]] = []
    for kind_dir in sorted((p for p in analyses_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        kind = kind_dir.name
        if wanted and kind != wanted:
            continue
        for analysis_dir in sorted((p for p in kind_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            meta_path = analysis_dir / "analysis_metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            # Ensure stable keys for the UI.
            meta.setdefault("analysis_type", kind)
            meta.setdefault("analysis_id", analysis_dir.name)
            analyses.append(_convert_nan_to_none(meta))

    def _sort_key(m: dict[str, Any]):
        return str(m.get("created_at") or "")

    analyses.sort(key=_sort_key, reverse=True)
    return {"cluster_id": cluster_id, "analyses": analyses}


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/analyses/{analysis_type}/{analysis_id}/data",
    summary="Load a stored analysis.npz as JSON arrays for visualization.",
)
async def get_cluster_analysis_data(
    project_id: str,
    system_id: str,
    cluster_id: str,
    analysis_type: str,
    analysis_id: str,
):
    try:
        project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    analysis_dir = cluster_dirs["cluster_dir"] / "analyses" / analysis_type / analysis_id
    meta_path = analysis_dir / "analysis_metadata.json"
    npz_path = analysis_dir / "analysis.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found on disk.")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read analysis metadata: {exc}") from exc

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            payload: dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                if isinstance(value, np.ndarray):
                    payload[key] = value.tolist()
                else:
                    payload[key] = value
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load analysis NPZ: {exc}") from exc

    return {"metadata": _convert_nan_to_none(meta), "data": _convert_nan_to_none(payload)}


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/samples/{sample_id}/stats",
    summary="Load basic stats for a saved sample NPZ (sample.npz or legacy).",
)
async def get_sample_stats(
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_id: str,
):
    try:
        project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    samples = project_store.list_samples(project_id, system_id, cluster_id)
    if not isinstance(samples, list) or not samples:
        raise HTTPException(status_code=404, detail="No samples recorded for this cluster.")
    sample_entry = next((s for s in samples if isinstance(s, dict) and s.get("sample_id") == sample_id), None)
    if not sample_entry:
        raise HTTPException(status_code=404, detail="Sample not found in cluster metadata.")
    paths = sample_entry.get("paths") if isinstance(sample_entry, dict) else None
    summary_rel = paths.get("summary_npz") if isinstance(paths, dict) else None
    summary_rel = summary_rel or sample_entry.get("path")
    if not summary_rel:
        raise HTTPException(status_code=404, detail="Sample NPZ path is missing.")

    npz_path = project_store.resolve_path(project_id, system_id, str(summary_rel))
    if not npz_path.exists():
        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        alt = cluster_dirs["cluster_dir"] / str(summary_rel)
        if alt.exists():
            npz_path = alt
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail="Sample NPZ not found on disk.")

    keys: list[str] = []
    n_frames = 0
    n_residues = 0
    invalid_count = 0
    has_halo = False
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            keys = list(data.files)
            labels = None
            if "labels" in data:
                labels = np.asarray(data["labels"])
                has_halo = "labels_halo" in data
                if "invalid_mask" in data:
                    invalid_mask = np.asarray(data["invalid_mask"], dtype=bool).ravel()
                    invalid_count = int(np.count_nonzero(invalid_mask))
            elif "assigned__labels_assigned" in data:
                labels = np.asarray(data["assigned__labels_assigned"])
                has_halo = "assigned__labels" in data
            elif "assigned__labels" in data:
                labels = np.asarray(data["assigned__labels"])
                has_halo = True
            elif "X_sa" in data:
                labels = np.asarray(data["X_sa"])
                if "sa_invalid_mask" in data:
                    invalid_mask = np.asarray(data["sa_invalid_mask"], dtype=bool).ravel()
                    invalid_count = int(np.count_nonzero(invalid_mask))
            elif "X_gibbs" in data:
                labels = np.asarray(data["X_gibbs"])
            if isinstance(labels, np.ndarray) and labels.ndim == 2:
                n_frames = int(labels.shape[0])
                n_residues = int(labels.shape[1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read sample NPZ: {exc}") from exc

    invalid_fraction = float(invalid_count) / float(n_frames) if n_frames else 0.0
    return {
        "sample_id": sample_id,
        "cluster_id": cluster_id,
        "path": str(summary_rel),
        "keys": keys,
        "n_frames": n_frames,
        "n_residues": n_residues,
        "invalid_count": invalid_count,
        "invalid_fraction": invalid_fraction,
        "has_halo": bool(has_halo),
    }


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/samples/{sample_id}/residue_profile",
    summary="Compute residue and edge cluster distributions for one sample.",
)
async def get_sample_residue_profile(
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_id: str,
    payload: Dict[str, Any],
):
    residue_index_raw = (payload or {}).get("residue_index")
    if residue_index_raw is None:
        raise HTTPException(status_code=400, detail="residue_index is required.")
    try:
        residue_index = int(residue_index_raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="residue_index must be an integer.") from exc

    label_mode = str((payload or {}).get("label_mode") or "assigned").strip().lower()
    if label_mode not in {"assigned", "labels", "halo", "labels_halo"}:
        raise HTTPException(status_code=400, detail="label_mode must be one of: assigned, halo.")

    edge_pairs_raw = (payload or {}).get("edge_pairs") or []
    if not isinstance(edge_pairs_raw, list):
        raise HTTPException(status_code=400, detail="edge_pairs must be a list of [r,s] pairs.")

    edge_pairs: List[tuple[int, int]] = []
    for item in edge_pairs_raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            r = int(item[0])
            s = int(item[1])
        except Exception:
            continue
        if r == s:
            continue
        edge_pairs.append((r, s))

    try:
        project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="System not found.")

    samples = project_store.list_samples(project_id, system_id, cluster_id)
    if not isinstance(samples, list) or not samples:
        raise HTTPException(status_code=404, detail="No samples recorded for this cluster.")
    sample_entry = next((s for s in samples if isinstance(s, dict) and s.get("sample_id") == sample_id), None)
    if not sample_entry:
        raise HTTPException(status_code=404, detail="Sample not found in sample directory.")
    paths = sample_entry.get("paths") if isinstance(sample_entry, dict) else None
    sample_rel = paths.get("summary_npz") if isinstance(paths, dict) else None
    sample_rel = sample_rel or sample_entry.get("path")
    if not sample_rel:
        raise HTTPException(status_code=404, detail="Sample NPZ path is missing.")

    sample_path = project_store.resolve_path(project_id, system_id, str(sample_rel))
    if not sample_path.exists():
        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        alt = cluster_dirs["cluster_dir"] / str(sample_rel)
        if alt.exists():
            sample_path = alt
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample NPZ not found on disk.")

    try:
        with np.load(sample_path, allow_pickle=True) as sample_npz:
            labels = None
            if label_mode in {"halo", "labels_halo"}:
                if "labels_halo" in sample_npz:
                    labels = np.asarray(sample_npz["labels_halo"], dtype=int)
                elif "assigned__labels" in sample_npz:
                    labels = np.asarray(sample_npz["assigned__labels"], dtype=int)
            if labels is None:
                if "labels" in sample_npz:
                    labels = np.asarray(sample_npz["labels"], dtype=int)
                elif "assigned__labels_assigned" in sample_npz:
                    labels = np.asarray(sample_npz["assigned__labels_assigned"], dtype=int)
                elif "assigned__labels" in sample_npz:
                    labels = np.asarray(sample_npz["assigned__labels"], dtype=int)
            if labels is None:
                raise HTTPException(status_code=400, detail="Sample NPZ has no labels.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sample NPZ: {exc}") from exc

    if labels.ndim != 2:
        raise HTTPException(status_code=400, detail="Sample labels array must be 2D.")

    n_res = int(labels.shape[1])
    if residue_index < 0 or residue_index >= n_res:
        raise HTTPException(status_code=400, detail=f"residue_index out of range [0,{n_res - 1}].")

    valid_pos = np.where(labels >= 0, labels, -1)
    K = np.max(valid_pos, axis=0).astype(int, copy=False) + 1 if valid_pos.size else np.zeros((n_res,), dtype=int)
    K = np.maximum(K, 1)

    Ki = int(K[residue_index])
    col = np.asarray(labels[:, residue_index], dtype=int)
    valid = (col >= 0) & (col < Ki)
    valid_count = int(np.count_nonzero(valid))
    counts = np.bincount(col[valid], minlength=Ki).astype(float, copy=False) if valid_count > 0 else np.zeros((Ki,), dtype=float)
    probs = counts / float(valid_count) if valid_count > 0 else np.zeros((Ki,), dtype=float)

    edge_profiles: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for r, s in edge_pairs:
        if r < 0 or s < 0 or r >= n_res or s >= n_res:
            continue
        rr, ss = (r, s) if r < s else (s, r)
        if (rr, ss) in seen_pairs:
            continue
        seen_pairs.add((rr, ss))
        Kr = int(K[rr])
        Ks = int(K[ss])
        if Kr <= 0 or Ks <= 0:
            continue
        ar = np.asarray(labels[:, rr], dtype=int)
        bs = np.asarray(labels[:, ss], dtype=int)
        valid_edge = (ar >= 0) & (ar < Kr) & (bs >= 0) & (bs < Ks)
        edge_valid_count = int(np.count_nonzero(valid_edge))
        if edge_valid_count > 0:
            flat = np.asarray(ar[valid_edge] * Ks + bs[valid_edge], dtype=int)
            joint_counts = np.bincount(flat, minlength=Kr * Ks).astype(float, copy=False).reshape(Kr, Ks)
            joint_probs = joint_counts / float(edge_valid_count)
        else:
            joint_probs = np.zeros((Kr, Ks), dtype=float)
        edge_profiles.append(
            {
                "r": int(rr),
                "s": int(ss),
                "k_r": int(Kr),
                "k_s": int(Ks),
                "cluster_ids_r": list(range(int(Kr))),
                "cluster_ids_s": list(range(int(Ks))),
                "valid_count": int(edge_valid_count),
                "invalid_count": int(labels.shape[0] - edge_valid_count),
                "joint_probs": joint_probs.tolist(),
            }
        )

    residue_label = f"res_{residue_index}"
    return {
        "sample_id": sample_id,
        "sample_name": sample_entry.get("name"),
        "label_mode": "halo" if label_mode in {"halo", "labels_halo"} else "assigned",
        "n_frames": int(labels.shape[0]),
        "residue_index": int(residue_index),
        "residue_label": str(residue_label),
        "k": int(Ki),
        "node_cluster_ids": list(range(int(Ki))),
        "node_probs": probs.tolist(),
        "node_valid_count": int(valid_count),
        "node_invalid_count": int(labels.shape[0] - valid_count),
        "edge_profiles": edge_profiles,
    }


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
        cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
        alt = cluster_dirs["cluster_dir"] / rel_path
        if alt.exists():
            artifact_path = alt
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
        density_z = 2.0
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

    selected_state_ids = []
    try:
        descriptor_state_ids = {
            str(sid)
            for sid, state in (system_meta.states or {}).items()
            if getattr(state, "descriptor_file", None)
        }
        selected_state_ids = [str(v) for v in parsed["state_ids"] if str(v) in descriptor_state_ids]
    except Exception:
        selected_state_ids = []

    assignments = build_md_eval_samples_for_cluster(
        project_id,
        system_id,
        cluster_id,
        cluster_path=npz_path,
        selected_state_ids=selected_state_ids,
        include_remaining_states=True,
        store=project_store,
    )

    cluster_entry = _build_cluster_entry(parsed, cluster_id, "finished", 100, "Complete")
    cluster_entry.update(
        {
            "path": rel_path,
            "generated_at": meta.get("generated_at") if isinstance(meta, dict) else None,
            "contact_edge_count": meta.get("contact_edge_count") if isinstance(meta, dict) else None,
            "samples": assignments.get("samples", []),
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

    selected_state_ids = []
    try:
        descriptor_state_ids = {
            str(sid)
            for sid, state in (system_meta.states or {}).items()
            if getattr(state, "descriptor_file", None)
        }
        selected_state_ids = [str(v) for v in parsed_state_ids if str(v) in descriptor_state_ids]
    except Exception:
        selected_state_ids = []

    assignments = build_md_eval_samples_for_cluster(
        project_id,
        system_id,
        cluster_id,
        cluster_path=out_path,
        selected_state_ids=selected_state_ids,
        include_remaining_states=True,
        store=project_store,
    )

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

    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []
    existing = [
        s
        for s in samples
        if isinstance(s, dict)
        and (s.get("type") or "") == "md_eval"
        and (s.get("state_id") or "") == state_id
        and s.get("sample_id")
    ]
    reuse_id = None
    if existing:
        existing.sort(key=lambda s: str(s.get("created_at") or ""))
        reuse_id = str(existing[-1].get("sample_id"))
        dup_ids = {str(s.get("sample_id")) for s in existing[:-1] if s.get("sample_id")}
        if dup_ids:
            samples = [s for s in samples if not (isinstance(s, dict) and s.get("sample_id") in dup_ids)]

    try:
        sample_entry = evaluate_state_with_models(project_id, system_id, cluster_id, state_id, sample_id=reuse_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    out_id = sample_entry.get("sample_id")
    replaced = False
    for idx, s in enumerate(samples):
        if isinstance(s, dict) and s.get("sample_id") == out_id:
            samples[idx] = sample_entry
            replaced = True
            break
    if not replaced:
        samples.append(sample_entry)
    entry["samples"] = samples
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    return {"status": "ok", "cluster_id": cluster_id, "sample": sample_entry}


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/assign_states",
    summary="Assign selected macro states to an existing cluster and create/update MD samples",
)
async def assign_states_against_cluster(
    project_id: str,
    system_id: str,
    cluster_id: str,
    payload: Dict[str, Any],
):
    raw_state_ids = (payload or {}).get("state_ids")
    if raw_state_ids is None:
        raw_state_ids = []
    if not isinstance(raw_state_ids, list):
        raise HTTPException(status_code=400, detail="state_ids must be a list.")
    state_ids = []
    seen_ids: set[str] = set()
    for raw in raw_state_ids:
        sid = str(raw).strip()
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        state_ids.append(sid)

    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")

    descriptor_state_ids = [
        str(sid)
        for sid, state in (system_meta.states or {}).items()
        if getattr(state, "descriptor_file", None)
    ]
    if not state_ids:
        state_ids = descriptor_state_ids
    invalid = [sid for sid in state_ids if sid not in descriptor_state_ids]
    if invalid:
        raise HTTPException(status_code=400, detail=f"State(s) missing descriptors: {', '.join(invalid)}")

    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []

    out_samples: List[Dict[str, Any]] = list(samples)
    refreshed: List[Dict[str, Any]] = []
    for state_id in state_ids:
        existing = [
            s
            for s in out_samples
            if isinstance(s, dict)
            and (s.get("type") or "") == "md_eval"
            and (s.get("state_id") or "") == state_id
            and s.get("sample_id")
        ]
        reuse_id = None
        if existing:
            existing.sort(key=lambda s: str(s.get("created_at") or ""))
            reuse_id = str(existing[-1].get("sample_id"))
            dup_ids = {str(s.get("sample_id")) for s in existing[:-1] if s.get("sample_id")}
            if dup_ids:
                out_samples = [s for s in out_samples if not (isinstance(s, dict) and s.get("sample_id") in dup_ids)]

        try:
            sample_entry = evaluate_state_with_models(
                project_id,
                system_id,
                cluster_id,
                state_id,
                sample_id=reuse_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        out_id = sample_entry.get("sample_id")
        replaced = False
        for idx, s in enumerate(out_samples):
            if isinstance(s, dict) and s.get("sample_id") == out_id:
                out_samples[idx] = sample_entry
                replaced = True
                break
        if not replaced:
            out_samples = [
                s
                for s in out_samples
                if not (
                    isinstance(s, dict)
                    and (s.get("type") or "") == "md_eval"
                    and (s.get("state_id") or "") == state_id
                )
            ]
            out_samples.append(sample_entry)
        refreshed.append(sample_entry)

    entry["samples"] = out_samples
    entry["updated_at"] = datetime.utcnow().isoformat()
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    return {"status": "ok", "cluster_id": cluster_id, "states": state_ids, "samples": refreshed}


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


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}/potts_models/lambda",
    summary="Create a derived lambda-interpolated Potts model (saved under potts_models/)",
)
async def create_lambda_potts_model(
    project_id: str,
    system_id: str,
    cluster_id: str,
    payload: LambdaPottsModelCreateRequest,
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not isinstance(entry, dict):
        raise HTTPException(status_code=404, detail="Cluster not found.")

    lam = float(payload.lam)
    if not (0.0 <= lam <= 1.0):
        raise HTTPException(status_code=400, detail="lam must be in [0,1].")
    model_a_id = str(payload.model_a_id).strip()
    model_b_id = str(payload.model_b_id).strip()
    if not model_a_id or not model_b_id:
        raise HTTPException(status_code=400, detail="model_a_id and model_b_id are required.")
    if model_a_id == model_b_id:
        raise HTTPException(status_code=400, detail="Select two different endpoint models.")

    models = entry.get("potts_models") or []
    model_a_meta = next((m for m in models if m.get("model_id") == model_a_id), None)
    model_b_meta = next((m for m in models if m.get("model_id") == model_b_id), None)
    if not isinstance(model_a_meta, dict) or not isinstance(model_b_meta, dict):
        raise HTTPException(status_code=404, detail="Could not locate both endpoint models in this cluster.")

    for mid, meta in [(model_a_id, model_a_meta), (model_b_id, model_b_meta)]:
        params = meta.get("params") or {}
        if isinstance(params, dict):
            dk = str(params.get("delta_kind") or "").strip().lower()
            if dk.startswith("delta"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Endpoint model {mid} appears delta-only (params.delta_kind={dk!r}). "
                        "Please select a sampleable endpoint model (standard or combined)."
                    ),
                )

    rel_a = model_a_meta.get("path")
    rel_b = model_b_meta.get("path")
    if not rel_a or not rel_b:
        raise HTTPException(status_code=404, detail="Endpoint model path missing in metadata.")

    dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = dirs["system_dir"]

    abs_a = project_store.resolve_path(project_id, system_id, str(rel_a))
    abs_b = project_store.resolve_path(project_id, system_id, str(rel_b))
    if not abs_a.exists() or not abs_b.exists():
        raise HTTPException(status_code=404, detail="Endpoint model NPZ missing on disk.")

    try:
        model_a = load_potts_model(str(abs_a))
        model_b = load_potts_model(str(abs_b))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load endpoint models: {exc}") from exc

    do_gauge = bool(payload.zero_sum_gauge) if payload.zero_sum_gauge is not None else True
    if do_gauge:
        model_a = zero_sum_gauge_model(model_a)
        model_b = zero_sum_gauge_model(model_b)
    try:
        derived = interpolate_potts_models(model_b, model_a, lam)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot interpolate endpoint models: {exc}") from exc
    if do_gauge:
        derived = zero_sum_gauge_model(derived)

    a_name = model_a_meta.get("name") or model_a_id
    b_name = model_b_meta.get("name") or model_b_id
    default_name = f"Lambda {lam:.3f} {b_name} -> {a_name}"
    display_name = (payload.name or "").strip() or default_name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name).strip("._-") or "lambda_model"

    model_id = str(uuid.uuid4())
    model_dir = dirs["potts_models_dir"] / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / f"{safe}.npz"
    if dest_path.exists():
        dest_path = model_dir / f"{safe}-{model_id[:8]}.npz"

    params = {
        "fit_mode": "derived",
        "derived_kind": "lambda_interpolation",
        "lambda": lam,
        "endpoint_model_a_id": model_a_id,
        "endpoint_model_b_id": model_b_id,
        "endpoint_model_a_name": a_name,
        "endpoint_model_b_name": b_name,
        "zero_sum_gauge": do_gauge,
    }
    save_potts_model(derived, dest_path, metadata=params)
    rel_path = str(dest_path.relative_to(system_dir))

    if not isinstance(models, list):
        models = []
    models.append(
        {
            "model_id": model_id,
            "name": display_name,
            "path": rel_path,
            "created_at": datetime.utcnow().isoformat(),
            "source": "derived",
            "params": params,
        }
    )
    entry["potts_models"] = models
    system_meta.metastable_clusters = clusters
    project_store.save_system(system_meta)
    return {"status": "created", "cluster_id": cluster_id, "model_id": model_id, "path": rel_path, "name": display_name}
