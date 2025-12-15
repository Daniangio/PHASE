"""
API Routers for V1
Defines endpoints for job submission, status polling, and result retrieval.
"""

import shutil
import functools
import json, os
from dataclasses import asdict
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form, Response, Query
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any, List, Optional, Tuple
import uuid
from pathlib import Path
from rq.job import Job
from redis import RedisError
import MDAnalysis as mda
import numpy as np

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
from backend.services.metastable import recompute_metastable_states
from backend.services.metastable_clusters import generate_metastable_cluster_npz
from backend.services.preprocessing import DescriptorPreprocessor
from backend.services.descriptors import save_descriptor_npz, load_descriptor_npz

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


def _ensure_not_macro_locked(system_meta: SystemMetadata):
    if getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="System macro-states are locked; no further edits allowed.")


def _ensure_not_metastable_locked(system_meta: SystemMetadata):
    if getattr(system_meta, "metastable_locked", False):
        raise HTTPException(status_code=400, detail="Metastable states are locked; recomputation is disabled.")


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
    # Trigger metastable recomputation (best-effort)
    try:
        recompute_metastable_states(system_meta.project_id, system_meta.system_id)
        system_meta = project_store.get_system(system_meta.project_id, system_meta.system_id)
    except Exception as exc:
        print(f"[metastable] Recompute failed: {exc}")
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


@api_router.get(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/descriptors",
    summary="Preview descriptor angles for a state (for visualization)",
)
async def get_state_descriptors(
    project_id: str,
    system_id: str,
    state_id: str,
    residue_keys: Optional[str] = Query(
        None,
        description="Comma-separated residue keys to include; defaults to all keys for the state.",
    ),
    metastable_ids: Optional[str] = Query(
        None,
        description="Comma-separated metastable IDs to filter frames; defaults to all frames.",
    ),
    cluster_id: Optional[str] = Query(
        None,
        description="ID of a saved cluster NPZ to use for coloring (optional).",
    ),
    cluster_mode: Optional[str] = Query(
        "merged",
        description="Cluster mode: 'merged' or 'per_meta' (only if cluster_id provided).",
    ),
    max_points: int = Query(
        2000,
        ge=10,
        le=50000,
        description="Maximum number of points returned per residue (down-sampled evenly).",
    ),
):
    """
    Returns a down-sampled set of phi/psi/chi1 angles (in degrees) for the requested state.
    Intended for client-side scatter plotting; not for bulk export.
    """
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = _get_state_or_404(system, state_id)
    if not state_meta.descriptor_file:
        raise HTTPException(status_code=404, detail="No descriptors stored for this state.")

    descriptor_path = project_store.resolve_path(project_id, system_id, state_meta.descriptor_file)
    if not descriptor_path.exists():
        raise HTTPException(status_code=404, detail="Descriptor file missing on disk.")

    try:
        feature_dict = load_descriptor_npz(descriptor_path)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to load descriptor file: {exc}") from exc

    keys_to_use = list(feature_dict.keys())
    if residue_keys:
        requested = [key.strip() for key in residue_keys.split(",") if key.strip()]
        keys_to_use = [k for k in keys_to_use if k in requested]
        if not keys_to_use:
            raise HTTPException(status_code=400, detail="No matching residue keys found in descriptor file.")

    angles_payload: Dict[str, Any] = {}
    residue_labels: Dict[str, str] = {}
    sample_stride = 1
    n_frames = 0

    # Try to resolve residue names from the stored PDB for nicer labels
    resname_map: Dict[int, str] = {}
    if state_meta.pdb_file:
        try:
            pdb_path = project_store.resolve_path(project_id, system_id, state_meta.pdb_file)
            if pdb_path.exists():
                u = mda.Universe(str(pdb_path))
                for res in u.residues:
                    resname_map[int(res.resid)] = str(res.resname).strip()
        except Exception:
            resname_map = {}

    # --- Metastable filtering ---
    metastable_filter_ids = []
    if metastable_ids:
        metastable_filter_ids = [mid.strip() for mid in metastable_ids.split(",") if mid.strip()]
    meta_id_to_index = {}
    index_to_meta_id = {}
    if system.metastable_states:
        for m in system.metastable_states:
            mid = m.get("metastable_id")
            if mid is None:
                continue
            meta_id_to_index[mid] = m.get("metastable_index")
            if m.get("metastable_index") is not None:
                index_to_meta_id[m.get("metastable_index")] = mid

    # --- Cluster NPZ (optional) ---
    cluster_npz = None
    cluster_meta = None
    cluster_mode_final = None
    merged_lookup = {}
    per_meta_lookup: Dict[str, Dict[Tuple[str, int], int]] = {}
    cluster_residue_indices: Dict[str, int] = {}
    merged_labels_arr: Optional[np.ndarray] = None
    per_meta_label_arrays: Dict[str, np.ndarray] = {}
    cluster_legend: List[Dict[str, Any]] = []
    cluster_color_ids: Dict[Tuple[str, int], int] = {}

    def _build_lookup(entry_key: str, npz_dict, keys_dict):
        """
        Build a mapping (state_id, frame_idx) -> row index.
        Falls back to sequential frame indices for backward-compatible NPZs that
        lack the explicit frame_indices array.
        """
        if not keys_dict or "frame_state_ids" not in keys_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster NPZ is missing frame index metadata for '{entry_key}'. Regenerate clusters.",
            )
        if keys_dict["frame_state_ids"] not in npz_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster NPZ is missing array '{keys_dict['frame_state_ids']}'. Regenerate clusters.",
            )

        frame_states = np.asarray(npz_dict[keys_dict["frame_state_ids"]])
        if "frame_indices" in keys_dict and keys_dict.get("frame_indices") in npz_dict:
            frame_indices = np.asarray(npz_dict[keys_dict["frame_indices"]])
        else:
            # Backward compatibility: assume rows correspond to sequential frame indices.
            frame_indices = np.arange(len(frame_states), dtype=int)

        lookup = {}
        for i, (sid, fidx) in enumerate(zip(frame_states, frame_indices)):
            lookup[(str(sid), int(fidx))] = i
        return lookup

    logger = None
    try:
        import logging
        logger = logging.getLogger("descriptor_debug")
    except Exception:
        logger = None
    if logger:
        logger.error(
            "[desc] state=%s project=%s system=%s metastable_ids=%s cluster_id=%s cluster_mode=%s max_points=%s",
            state_id,
            project_id,
            system_id,
            metastable_ids,
            cluster_id,
            cluster_mode,
            max_points,
        )

    if cluster_id:
        entry = next((c for c in system.metastable_clusters or [] if c.get("cluster_id") == cluster_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Cluster NPZ not found.")
        rel_path = entry.get("path")
        if not rel_path:
            raise HTTPException(status_code=404, detail="Cluster NPZ path missing.")
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
        if not cluster_path.exists():
            raise HTTPException(status_code=404, detail="Cluster NPZ file missing.")
        if logger:
            logger.error("[desc] loading cluster npz: %s", cluster_path)
        cluster_npz = np.load(cluster_path, allow_pickle=True)
        try:
            cluster_meta = json.loads(cluster_npz["metadata_json"].item())
        except Exception:
            cluster_meta = None
        if not isinstance(cluster_meta, dict) or not cluster_meta:
            raise HTTPException(status_code=400, detail="Cluster NPZ missing metadata_json. Regenerate clusters.")
        cluster_mode_final = (cluster_mode or "merged").lower()
        if cluster_mode_final not in {"merged", "per_meta"}:
            raise HTTPException(status_code=400, detail="cluster_mode must be 'merged' or 'per_meta'.")
        cluster_res_keys = list(cluster_meta.get("residue_keys", []))
        cluster_residue_indices = {k: i for i, k in enumerate(cluster_res_keys)}

        if cluster_mode_final == "merged":
            merged_keys = cluster_meta.get("merged", {}).get("npz_keys", {})
            if (
                not merged_keys
                or "labels" not in merged_keys
                or "frame_state_ids" not in merged_keys
                or "frame_indices" not in merged_keys
            ):
                raise HTTPException(status_code=400, detail="Cluster NPZ missing merged frame metadata. Regenerate clusters.")
            merged_lookup = _build_lookup("merged", cluster_npz, merged_keys)
            merged_labels_arr = cluster_npz[merged_keys.get("labels")]
            unique_clusters = sorted({int(v) for v in np.unique(merged_labels_arr) if int(v) >= 0})
            cluster_legend = [{"id": c, "label": f"Merged c{c}"} for c in unique_clusters]
            if logger:
                logger.error(
                    "[desc] merged lookup ready rows=%d clusters=%s",
                    merged_labels_arr.shape[0],
                    unique_clusters,
                )
        else:
            per_meta = cluster_meta.get("per_metastable", {})
            global_counter = 0
            legend_entries = []
            for mid, info in per_meta.items():
                keys_dict = info.get("npz_keys", {})
                if (
                    not keys_dict
                    or "labels" not in keys_dict
                    or "frame_state_ids" not in keys_dict
                    or "frame_indices" not in keys_dict
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cluster NPZ missing frame metadata for metastable '{mid}'. Regenerate clusters.",
                    )
                per_meta_lookup[mid] = _build_lookup(mid, cluster_npz, keys_dict)
                labels_arr = cluster_npz[keys_dict.get("labels")]
                per_meta_label_arrays[mid] = labels_arr
                clusters_local = sorted({int(v) for v in np.unique(labels_arr) if int(v) >= 0})
                meta_name = next((m.get("name") or m.get("default_name") for m in system.metastable_states or [] if m.get("metastable_id") == mid), mid)
                for c in clusters_local:
                    cluster_color_ids[(mid, c)] = global_counter
                    legend_entries.append({"id": global_counter, "label": f"{meta_name} c{c}", "metastable_id": mid, "local_cluster": c})
                    global_counter += 1
            cluster_legend = legend_entries

    # Shared frame selection (metastable filter + sampling) computed once
    labels_meta = None
    if metastable_filter_ids or cluster_mode_final == "per_meta":
        labels_meta = feature_dict.get("metastable_labels")
        if labels_meta is None and state_meta.metastable_labels_file:
            label_path = project_store.resolve_path(project_id, system_id, state_meta.metastable_labels_file)
            if label_path.exists():
                labels_meta = np.load(label_path)
        if labels_meta is None:
            raise HTTPException(status_code=400, detail="Metastable labels missing for this state.")

    first_arr = feature_dict[keys_to_use[0]]
    total_frames = first_arr.shape[0] if hasattr(first_arr, "shape") else 0
    indices = np.arange(total_frames)
    if metastable_filter_ids:
        selected_idx = {meta_id_to_index.get(mid) for mid in metastable_filter_ids if mid in meta_id_to_index}
        if not selected_idx:
            raise HTTPException(status_code=400, detail="Selected metastable IDs not found on this system.")
        mask = np.isin(labels_meta, list(selected_idx))
        indices = np.where(mask)[0]
        if indices.size == 0:
            raise HTTPException(status_code=400, detail="No frames match selected metastable states for this state.")

    n_frames_filtered = indices.size
    sample_stride = max(1, n_frames_filtered // max_points) if n_frames_filtered > max_points else 1
    sample_indices = indices[::sample_stride]
    n_frames_out = n_frames_filtered
    if logger:
        logger.error(
            "[desc] frames_filtered=%d sampled=%d stride=%d",
            n_frames_filtered,
            len(sample_indices),
            sample_stride,
        )

    # Precompute merged/per-meta row mappings for sampled frames
    merged_rows_for_samples = None
    if cluster_mode_final == "merged" and cluster_npz is not None and merged_lookup:
        merged_rows_for_samples = np.array(
            [merged_lookup.get((state_meta.state_id, int(f)), -1) for f in sample_indices], dtype=int
        )

    per_meta_rows_for_samples: Dict[str, np.ndarray] = {}
    frame_meta_ids: Optional[np.ndarray] = None
    if cluster_mode_final == "per_meta" and cluster_npz is not None and labels_meta is not None:
        frame_meta_ids = np.array([index_to_meta_id.get(int(labels_meta[f]), None) for f in sample_indices], dtype=object)
        for mid, lookup_dict in per_meta_lookup.items():
            per_meta_rows_for_samples[mid] = np.array(
                [lookup_dict.get((state_meta.state_id, int(f)), -1) for f in sample_indices], dtype=int
            )

    for key in keys_to_use:
        arr = feature_dict[key]
        # Expected shape: (n_frames, 1, 3) in radians
        if arr.ndim != 3 or arr.shape[2] < 3:
            continue

        sampled = arr[sample_indices, 0, :]
        phi = (sampled[:, 0] * 180.0 / 3.141592653589793).tolist()
        psi = (sampled[:, 1] * 180.0 / 3.141592653589793).tolist()
        chi1 = (sampled[:, 2] * 180.0 / 3.141592653589793).tolist()
        angles_payload[key] = {"phi": phi, "psi": psi, "chi1": chi1}
        if logger:
            logger.error("[desc] residue=%s frames=%d sampled=%d stride=%d", key, n_frames_filtered, len(sample_indices), sample_stride)

        # Cluster labels (optional, vectorized)
        if cluster_npz is not None:
            res_idx = cluster_residue_indices.get(key, None)
            if res_idx is not None:
                if cluster_mode_final == "merged" and merged_rows_for_samples is not None:
                    if merged_labels_arr is not None:
                        safe_rows = np.clip(merged_rows_for_samples, 0, merged_labels_arr.shape[0] - 1)
                        labels_for_res = merged_labels_arr[safe_rows, res_idx].astype(int)
                        labels_for_res[merged_rows_for_samples < 0] = -1
                        angles_payload[key]["cluster_labels"] = labels_for_res.tolist()
                elif cluster_mode_final == "per_meta" and frame_meta_ids is not None:
                    labels_for_res = np.full(sample_indices.shape[0], -1, dtype=int)
                    for mid, rows in per_meta_rows_for_samples.items():
                        mask = frame_meta_ids == mid
                        if not np.any(mask):
                            continue
                        valid_mask = (rows >= 0) & mask
                        if not np.any(valid_mask):
                            continue
                        labels_arr = per_meta_label_arrays.get(mid)
                        if labels_arr is not None:
                            labels_for_res[valid_mask] = labels_arr[rows[valid_mask], res_idx]
                    angles_payload[key]["cluster_labels"] = labels_for_res.tolist()

        label = key
        selection = (state_meta.residue_mapping or {}).get(key) or ""
        resid_tokens = [
            tok for tok in selection.replace("resid", "").split() if tok.strip().lstrip("-").isdigit()
        ]
        resid_val = int(resid_tokens[0]) if resid_tokens else None
        if resid_val is not None and resid_val in resname_map:
            label = f"{key}_{resname_map[resid_val]}"
        residue_labels[key] = label

    if not angles_payload:
        raise HTTPException(status_code=500, detail="Descriptor file contained no usable angle data.")

    return {
        "residue_keys": keys_to_use,
        "residue_mapping": state_meta.residue_mapping or {},
        "residue_labels": residue_labels,
        "n_frames": n_frames_out,
        "sample_stride": sample_stride,
        "angles": angles_payload,
        "cluster_mode": cluster_mode_final,
        "cluster_legend": cluster_legend,
        "metastable_filter_applied": bool(metastable_filter_ids),
    }


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
    _ensure_not_macro_locked(system_meta)

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
    _ensure_not_macro_locked(system_meta)

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
    _ensure_not_macro_locked(system_meta)

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
    _ensure_not_macro_locked(system_meta)

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


@api_router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/recompute",
    summary="Recompute metastable states across all descriptor-ready trajectories",
)
async def recompute_metastable(
    project_id: str,
    system_id: str,
    n_microstates: int = Query(20, ge=2, le=500),
    k_meta_min: int = Query(1, ge=1, le=10),
    k_meta_max: int = Query(4, ge=1, le=10),
    tica_lag_frames: int = Query(5, ge=1),
    tica_dim: int = Query(5, ge=1),
    random_state: int = Query(0),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro-states before running metastable analysis.")
    _ensure_not_metastable_locked(system_meta)

    try:
        result = await run_in_threadpool(
            recompute_metastable_states,
            project_id,
            system_id,
            n_microstates=n_microstates,
            k_meta_min=k_meta_min,
            k_meta_max=max(k_meta_min, k_meta_max),
            tica_lag_frames=tica_lag_frames,
            tica_dim=tica_dim,
            random_state=random_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metastable recompute failed: {exc}") from exc

    return result


@api_router.get(
    "/projects/{project_id}/systems/{system_id}/metastable",
    summary="List metastable states for a system",
)
async def list_metastable_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    return {
        "metastable_states": system_meta.metastable_states or [],
        "model_dir": system_meta.metastable_model_dir,
    }


@api_router.post(
    "/projects/{project_id}/systems/{system_id}/states/confirm",
    summary="Lock macro-states to proceed to metastable analysis",
)
async def confirm_macro_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if getattr(system_meta, "macro_locked", False):
        return _serialize_system(system_meta)

    if not system_meta.states:
        raise HTTPException(status_code=400, detail="Add at least one state before locking.")

    system_meta.macro_locked = True
    project_store.save_system(system_meta)
    return _serialize_system(system_meta)


@api_router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/confirm",
    summary="Lock metastable states to proceed to clustering and analysis",
)
async def confirm_metastable_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro-states first.")

    if getattr(system_meta, "metastable_locked", False):
        return _serialize_system(system_meta)

    if not (system_meta.metastable_states or []):
        raise HTTPException(status_code=400, detail="Run metastable recompute before locking.")

    system_meta.metastable_locked = True
    project_store.save_system(system_meta)
    return _serialize_system(system_meta)


@api_router.post(
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
    if not getattr(system_meta, "metastable_locked", False):
        raise HTTPException(status_code=400, detail="Lock metastable states before clustering.")

    metastable_ids_raw = (payload or {}).get("metastable_ids") or []
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

    try:
        random_state = int((payload or {}).get("random_state", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="random_state must be an integer.")
    try:
        contact_cutoff = float((payload or {}).get("contact_cutoff", 10.0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="contact_cutoff must be a number.")
    contact_atom_mode = str((payload or {}).get("contact_atom_mode", payload.get("contact_mode", "CA") if isinstance(payload, dict) else "CA") or "CA").upper()
    if contact_atom_mode not in {"CA", "CM"}:
        raise HTTPException(status_code=400, detail="contact_atom_mode must be 'CA' or 'CM'.")

    algo_raw = (payload or {}).get("cluster_algorithm", "tomato")
    cluster_algorithm = str(algo_raw or "tomato").lower()
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
    hierarchical_linkage = str(algo_params.get("linkage", (payload or {}).get("hierarchical_linkage", "ward")) or "ward").lower()
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

    try:
        if logger:
            logger.error(
                "[cluster_build] project=%s system=%s meta_ids=%s algo=%s params=%s",
                project_id,
                system_id,
                metastable_ids,
                cluster_algorithm,
                {
                    "max_clusters": max_clusters,
                    "random_state": random_state,
                    "contact_cutoff": contact_cutoff,
                    "contact_atom_mode": contact_atom_mode,
                    "dbscan_eps": dbscan_eps,
                    "dbscan_min_samples": dbscan_min_samples,
                    "hierarchical_n_clusters": hierarchical_n_clusters,
                    "hierarchical_linkage": hierarchical_linkage,
                    "tomato_k": tomato_k,
                    "tomato_tau": tomato_tau,
                },
            )
        npz_path, meta = await run_in_threadpool(
            generate_metastable_cluster_npz,
            project_id,
            system_id,
            metastable_ids,
            max_clusters_per_residue=max_clusters,
            random_state=random_state,
            contact_cutoff=contact_cutoff,
            contact_atom_mode=contact_atom_mode,
            cluster_algorithm=cluster_algorithm,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            hierarchical_n_clusters=hierarchical_n_clusters,
            hierarchical_linkage=hierarchical_linkage,
            tomato_k=tomato_k,
            tomato_tau=tomato_tau,
            tomato_k_max=tomato_k_max,
        )
    except ValueError as exc:
        if logger:
            logger.error("[cluster_build] validation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if logger:
            logger.error("[cluster_build] failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build metastable clusters: {exc}") from exc

    # Persist reference to the generated cluster NPZ
    dirs = project_store.ensure_directories(project_id, system_id)
    try:
        rel_path = str(npz_path.relative_to(dirs["system_dir"]))
    except Exception:
        rel_path = str(npz_path)

    cluster_entry = {
        "cluster_id": str(uuid.uuid4()),
        "path": rel_path,
        "metastable_ids": metastable_ids,
        "max_clusters_per_residue": max_clusters,
        "random_state": random_state,
        "generated_at": meta.get("generated_at") if isinstance(meta, dict) else None,
        "contact_cutoff": contact_cutoff,
        "contact_atom_mode": contact_atom_mode,
        "contact_edge_count": meta.get("contact_edge_count") if isinstance(meta, dict) else None,
        "cluster_algorithm": cluster_algorithm,
        "algorithm_params": {
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "n_clusters": hierarchical_n_clusters,
            "linkage": hierarchical_linkage,
            "k_neighbors": tomato_k,
            "tau": tomato_tau,
            "k_max": tomato_k_max,
        },
    }
    system_meta.metastable_clusters = (system_meta.metastable_clusters or []) + [cluster_entry]
    project_store.save_system(system_meta)

    return FileResponse(
        npz_path,
        filename=npz_path.name,
        media_type="application/octet-stream",
    )


@api_router.get(
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


@api_router.delete(
    "/projects/{project_id}/systems/{system_id}/metastable/clusters/{cluster_id}",
    summary="Delete a saved metastable cluster NPZ",
)
async def delete_metastable_cluster_npz(project_id: str, system_id: str, cluster_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    if not getattr(system_meta, "macro_locked", False) or not getattr(system_meta, "metastable_locked", False):
        raise HTTPException(status_code=400, detail="Lock macro and metastable states before deleting cluster NPZ.")

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


@api_router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/{metastable_id}/pdb",
    summary="Download representative PDB for a metastable state",
)
async def fetch_metastable_pdb(project_id: str, system_id: str, metastable_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    metas = system_meta.metastable_states or []
    target = next((m for m in metas if m.get("metastable_id") == metastable_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Metastable state '{metastable_id}' not found.")
    pdb_rel = target.get("representative_pdb")
    if not pdb_rel:
        raise HTTPException(status_code=404, detail="No representative PDB stored for this metastable state.")
    pdb_path = project_store.resolve_path(project_id, system_id, pdb_rel)
    if not pdb_path.exists():
        raise HTTPException(status_code=404, detail="Representative PDB file is missing on disk.")
    return FileResponse(pdb_path, filename=pdb_path.name, media_type="chemical/x-pdb")


@api_router.patch(
    "/projects/{project_id}/systems/{system_id}/metastable/{metastable_id}",
    summary="Rename a metastable state",
)
async def rename_metastable_state(
    project_id: str,
    system_id: str,
    metastable_id: str,
    payload: Dict[str, Any],
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    new_name = (payload or {}).get("name")
    if not new_name or not str(new_name).strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    new_name = str(new_name).strip()

    updated = False
    metas = system_meta.metastable_states or []
    for meta in metas:
        if meta.get("metastable_id") == metastable_id:
            meta["name"] = new_name
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail=f"Metastable state '{metastable_id}' not found.")

    system_meta.metastable_states = metas
    project_store.save_system(system_meta)
    return {"metastable_states": metas}


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
