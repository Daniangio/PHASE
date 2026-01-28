"""Cluster per-residue angles inside selected metastable states and persist as NPZ."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree
import MDAnalysis as mda
from dadapy import Data

from backend.services.descriptors import load_descriptor_npz
from backend.services.project_store import DescriptorState, ProjectStore, SystemMetadata


def _slug(value: str) -> str:
    """Create a filesystem/NPZ-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_") or "metastable"


def build_cluster_output_path(
    project_id: str,
    system_id: str,
    *,
    cluster_id: str,
    cluster_name: Optional[str] = None,
) -> Path:
    store = ProjectStore()
    dirs = store.ensure_directories(project_id, system_id)
    cluster_dir = dirs["system_dir"] / "metastable" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(cluster_name) if cluster_name else "cluster"
    return cluster_dir / f"{slug}__{cluster_id}.npz"


def build_cluster_entry(
    *,
    cluster_id: str,
    cluster_name: Optional[str],
    state_ids: List[str],
    max_cluster_frames: Optional[int],
    random_state: int,
    density_maxk: int,
    density_z: float | str,
) -> Dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "name": cluster_name if cluster_name else None,
        "status": "finished",
        "progress": 100,
        "status_message": "Complete",
        "job_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "path": None,
        "state_ids": state_ids,
        "metastable_ids": state_ids,
        "max_cluster_frames": max_cluster_frames,
        "random_state": random_state,
        "generated_at": None,
        "contact_edge_count": None,
        "cluster_algorithm": "density_peaks",
        "algorithm_params": {
            "density_maxk": density_maxk,
            "density_z": density_z,
            "max_cluster_frames": max_cluster_frames,
        },
    }


def _build_state_name_maps(system_meta: SystemMetadata) -> tuple[Dict[str, str], Dict[str, str]]:
    state_labels = {}
    for state_id, state in (system_meta.states or {}).items():
        label = getattr(state, "name", None) or state_id
        state_labels[str(state_id)] = str(label)

    metastable_labels = {}
    for meta in system_meta.metastable_states or []:
        meta_id = meta.get("metastable_id") or meta.get("id")
        if not meta_id:
            continue
        label = meta.get("name") or meta.get("default_name") or meta_id
        metastable_labels[str(meta_id)] = str(label)

    return state_labels, metastable_labels


def _build_metastable_kind_map(system_meta: SystemMetadata) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for meta in system_meta.metastable_states or []:
        meta_id = meta.get("metastable_id") or meta.get("id")
        if not meta_id:
            continue
        kinds[str(meta_id)] = "metastable"
    for state_id in (system_meta.states or {}).keys():
        kinds.setdefault(str(state_id), "macro")
    return kinds


def _fit_density_peaks(
    samples: np.ndarray,
    *,
    density_maxk: int,
    density_z: float | str,
    halo: bool,
    n_jobs: int | None = None,
) -> tuple[Data, np.ndarray, int, Dict[str, Any]]:
    """Fit ADP density-peak clustering on embedded samples."""
    if samples.size == 0:
        raise ValueError("No samples provided for density peaks.")
    emb = _angles_to_embedding(samples)
    n = emb.shape[0]
    if n == 1:
        labels = np.zeros(1, dtype=np.int32)
        diag = {"density_peaks_k": 1, "density_peaks_maxk": 1, "density_peaks_Z": density_z}
        dp_data = Data(coordinates=emb, maxk=1, verbose=False, n_jobs=1)
        dp_data.cluster_assignment = labels  # type: ignore[attr-defined]
        dp_data.N_clusters = 1  # type: ignore[attr-defined]
        return dp_data, labels, 1, diag

    dp_maxk = max(1, min(int(density_maxk), n - 1))
    dp_data = Data(coordinates=emb, maxk=dp_maxk, verbose=False, n_jobs=1)
    dp_data.compute_distances()
    dp_data.compute_id_2NN()
    dp_data.compute_density_kstarNN()
    if isinstance(density_z, str) and density_z.lower() == "auto":
        dp_data.compute_clustering_ADP(halo=halo)
        density_z_val: float | str = "auto"
    else:
        density_z_val = float(density_z)
        dp_data.compute_clustering_ADP(Z=float(density_z_val), halo=halo)
    labels = np.asarray(dp_data.cluster_assignment, dtype=np.int32)
    k_final = int(dp_data.N_clusters) if hasattr(dp_data, "N_clusters") else int(
        len([c for c in np.unique(labels) if c >= 0])
    )
    diag: Dict[str, Any] = {
        "density_peaks_method": "dadapy_adp",
        "density_peaks_k": k_final,
        "density_peaks_maxk": dp_maxk,
        "density_peaks_Z": density_z_val,
    }
    return dp_data, labels, k_final, diag


def _predict_cluster_adp(
    dp_data: Data,
    samples: np.ndarray,
    *,
    density_maxk: int,
    halo: bool,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Predict ADP cluster labels for new samples using a fitted Data object."""
    if not hasattr(dp_data, "predict_cluster_ADP"):
        raise ValueError("DADApy predict_cluster_ADP is not available. Update the DADApy installation.")
    emb = _angles_to_embedding(samples)
    labels, _ = dp_data.predict_cluster_ADP(
        emb,
        maxk=max(1, min(int(density_maxk), emb.shape[0] - 1)),
        density_est="kstarNN",
        halo=halo,
        n_jobs=1
    )
    return np.asarray(labels, dtype=np.int32)


def _angles_to_embedding(samples: np.ndarray) -> np.ndarray:
    """Convert angle triplets to sin/cos embedding for periodic clustering."""
    angles = samples[:, :3]
    emb = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)


def _cluster_residue_samples(
    samples: np.ndarray,
    *,
    density_maxk: int,
    density_z: float | str,
    halo: bool,
    n_jobs: int | None = None,
) -> Tuple[np.ndarray, int, Dict[str, Any], Data]:
    """Cluster angles with ADP density peaks."""
    if samples.size == 0:
        return np.array([], dtype=np.int32), 0, {}, Data(coordinates=np.zeros((1, 1)))
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[1] < 3:
        raise ValueError("Residue samples must be (n_frames, >=3) shaped.")

    dp_data, labels, k_final, diagnostics = _fit_density_peaks(
        samples,
        density_maxk=density_maxk,
        density_z=density_z,
        halo=halo,
        n_jobs=n_jobs,
    )
    return labels, int(k_final), diagnostics, dp_data


def _uniform_subsample_indices(n_frames: int, max_frames: int) -> np.ndarray:
    """Pick roughly uniform indices up to max_frames."""
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    idx = np.linspace(0, n_frames - 1, num=max_frames, dtype=int)
    return np.unique(idx)


def _cluster_with_subsample(
    samples: np.ndarray,
    *,
    density_maxk: int,
    density_z: float | str,
    max_cluster_frames: Optional[int],
    n_jobs: int | None = None,
    subsample_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any], int, Data]:
    """Fit ADP on a subsample if requested, then predict labels on all frames."""
    n_frames = samples.shape[0]
    if not max_cluster_frames or max_cluster_frames <= 0 or n_frames <= max_cluster_frames:
        labels_halo, k, diag, dp_data = _cluster_residue_samples(
            samples,
            density_maxk=density_maxk,
            density_z=density_z,
            halo=True,
            n_jobs=n_jobs,
        )
        labels_assigned = _predict_cluster_adp(
            dp_data,
            samples,
            density_maxk=density_maxk,
            halo=False,
            n_jobs=n_jobs,
        )
        diag["subsampled"] = False
        diag["subsample_size"] = int(n_frames)
        diag["total_frames"] = int(n_frames)
        return labels_halo, labels_assigned, k, diag, int(n_frames), dp_data

    subsample_indices = (
        _uniform_subsample_indices(n_frames, int(max_cluster_frames))
        if subsample_indices is None
        else subsample_indices
    )
    subsample_indices = np.asarray(subsample_indices, dtype=int)
    sub_samples = samples[subsample_indices]
    _, k, diag, dp_data = _cluster_residue_samples(
        sub_samples,
        density_maxk=density_maxk,
        density_z=density_z,
        halo=True,
        n_jobs=n_jobs,
    )
    labels_halo = _predict_cluster_adp(
        dp_data,
        samples,
        density_maxk=density_maxk,
        halo=True,
        n_jobs=n_jobs,
    )
    labels_assigned = _predict_cluster_adp(
        dp_data,
        samples,
        density_maxk=density_maxk,
        halo=False,
        n_jobs=n_jobs,
    )
    diag["subsampled"] = True
    diag["subsample_size"] = int(subsample_indices.size)
    diag["total_frames"] = int(n_frames)
    return labels_halo, labels_assigned, k, diag, int(subsample_indices.size), dp_data


def _cluster_residue_worker(
    col: int,
    samples: np.ndarray,
    density_maxk: int,
    density_z: float | str,
    max_cluster_frames: Optional[int],
    subsample_indices: Optional[np.ndarray],
) -> Tuple[int, np.ndarray, np.ndarray, int]:
    labels_halo, labels_assigned, k, _, _, _ = _cluster_with_subsample(
        samples,
        density_maxk=density_maxk,
        density_z=density_z,
        max_cluster_frames=max_cluster_frames,
        subsample_indices=subsample_indices,
        n_jobs=1,
    )
    return col, labels_halo, labels_assigned, k


def _resolve_states_for_meta(meta: Dict[str, Any], system: SystemMetadata) -> List[DescriptorState]:
    """Return all states contributing to a metastable macro-state."""
    macro_state_id = meta.get("macro_state_id")
    macro_state_name = meta.get("macro_state")
    states: List[DescriptorState] = []
    for st in system.states.values():
        if macro_state_id and st.state_id == macro_state_id:
            states.append(st)
        elif macro_state_name and st.name == macro_state_name:
            states.append(st)
    return states


def _extract_labels_for_state(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    state: DescriptorState,
    features: Dict[str, Any],
) -> np.ndarray:
    """Extract metastable labels array, preferring embedded NPZ key."""
    labels = features.pop("metastable_labels", None)
    if labels is not None:
        labels = np.asarray(labels)
    elif state.metastable_labels_file:
        label_path = store.resolve_path(project_id, system_id, state.metastable_labels_file)
        if label_path.exists():
            labels = np.load(label_path)
    if labels is None:
        raise ValueError(f"No metastable labels found for state '{state.state_id}'.")
    return np.asarray(labels).astype(np.int32)


def _infer_frame_count(features: Dict[str, Any]) -> int:
    """Best-effort frame count inference from descriptor arrays."""
    for key, arr in features.items():
        if key == "metastable_labels":
            continue
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    return 0


def _coerce_residue_keys(
    residue_keys: List[str], features: Dict[str, Any], state: DescriptorState
) -> List[str]:
    """Prefer stored residue_keys but fall back to feature keys."""
    if residue_keys:
        return residue_keys
    if state.residue_keys:
        return _sort_residue_keys(state.residue_keys)
    feature_keys = [k for k in features.keys() if k != "metastable_labels"]
    return _sort_residue_keys(feature_keys)


def _sort_residue_keys(keys: List[str]) -> List[str]:
    """Sort keys by numeric resid extracted from patterns like 'res_123'."""
    def _extract_num(k: str) -> int:
        m = re.search(r"(\d+)$", k)
        return int(m.group(1)) if m else 0
    return sorted(keys, key=_extract_num)


def _extract_residue_positions(
    pdb_path: Path,
    residue_keys: List[str],
    residue_mapping: Dict[str, str],
    contact_mode: str,
) -> List[Optional[np.ndarray]]:
    """Return per-residue positions (CA or center-of-mass) for contact computation."""
    positions: List[Optional[np.ndarray]] = []
    u = mda.Universe(str(pdb_path))
    for key in residue_keys:
        sel = residue_mapping.get(key) or key
        try:
            res_atoms = u.select_atoms(sel)
        except Exception:
            positions.append(None)
            continue
        if res_atoms.n_atoms == 0:
            positions.append(None)
            continue
        if contact_mode == "CA":
            ca_atoms = res_atoms.select_atoms("name CA")
            if ca_atoms.n_atoms > 0:
                positions.append(np.array(ca_atoms[0].position, dtype=float))
            else:
                positions.append(np.array(res_atoms.center_of_mass(), dtype=float))
        else:
            positions.append(np.array(res_atoms.center_of_mass(), dtype=float))
    return positions


def _compute_contact_edges(
    pdb_path: Path,
    residue_keys: List[str],
    residue_mapping: Dict[str, str],
    cutoff: float,
    contact_mode: str,
) -> set:
    """Compute contact edges (i,j) for one PDB."""
    positions = _extract_residue_positions(pdb_path, residue_keys, residue_mapping, contact_mode)
    valid_indices = [i for i, pos in enumerate(positions) if pos is not None]
    edges: set = set()
    if len(valid_indices) < 2:
        return edges
    coords = np.stack([positions[i] for i in valid_indices], axis=0)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    for a_idx, i in enumerate(valid_indices):
        for b_idx in range(a_idx + 1, len(valid_indices)):
            j = valid_indices[b_idx]
            if dist[a_idx, b_idx] < cutoff:
                edges.add((min(i, j), max(i, j)))
    return edges


def _extract_angles_array(features: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    arr = features.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim >= 3:
        arr = arr[:, 0, :3]
    elif arr.ndim == 2:
        arr = arr[:, :3]
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 3:
        padded = np.zeros((arr.shape[0], 3), dtype=float)
        padded[:, : arr.shape[1]] = arr
        arr = padded
    return np.asarray(arr, dtype=float)


def _assign_labels_from_reference(
    target_emb: np.ndarray,
    ref_tree: KDTree,
    ref_labels: np.ndarray,
    k_neighbors: int,
) -> np.ndarray:
    if target_emb.size == 0 or ref_labels.size == 0:
        return np.full(target_emb.shape[0], -1, dtype=np.int32)
    k_eff = min(int(k_neighbors), ref_labels.shape[0])
    _, idxs = ref_tree.query(target_emb, k=k_eff)
    if idxs.ndim == 1:
        idxs = idxs[:, None]
    out = np.full(target_emb.shape[0], -1, dtype=np.int32)
    for i in range(target_emb.shape[0]):
        neigh = ref_labels[idxs[i]]
        vals, counts = np.unique(neigh, return_counts=True)
        if vals.size:
            out[i] = int(vals[np.argmax(counts)])
    return out


def assign_cluster_labels_to_states(
    cluster_path: Path,
    project_id: str,
    system_id: str,
    *,
    k_neighbors: int = 10,
) -> Dict[str, Dict[str, str]]:
    store = ProjectStore()
    system = store.get_system(project_id, system_id)
    data = np.load(cluster_path, allow_pickle=True)

    residue_keys = [str(k) for k in data["residue_keys"]]
    merged_counts = np.asarray(data["merged__cluster_counts"], dtype=np.int32)

    meta = {}
    if "metadata_json" in data:
        try:
            meta_raw = data["metadata_json"]
            meta_val = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
            meta = json.loads(str(meta_val))
        except Exception:
            meta = {}

    predictions = meta.get("predictions") or {}

    assign_dir = cluster_path.parent / f"{cluster_path.stem}_assigned"
    assign_dir.mkdir(parents=True, exist_ok=True)

    assigned_state_paths: Dict[str, str] = {}
    for state_id, state in system.states.items():
        key = f"state:{state_id}"
        entry = predictions.get(key)
        if not isinstance(entry, dict):
            continue
        labels_key = entry.get("labels_halo")
        if not labels_key or labels_key not in data:
            continue
        labels_halo = np.asarray(data[labels_key], dtype=np.int32)
        assigned_key = entry.get("labels_assigned")
        labels_assigned = (
            np.asarray(data[assigned_key], dtype=np.int32)
            if isinstance(assigned_key, str) and assigned_key in data
            else None
        )
        frame_indices_key = entry.get("frame_indices")
        if isinstance(frame_indices_key, str) and frame_indices_key in data:
            frame_indices = np.asarray(data[frame_indices_key], dtype=np.int64)
        else:
            frame_indices = np.arange(labels_halo.shape[0], dtype=np.int64)

        payload: Dict[str, Any] = {
            "residue_keys": np.asarray(residue_keys),
            "assigned__labels": labels_halo,
            "assigned__cluster_counts": merged_counts,
            "assigned__frame_state_id": np.array([state_id]),
            "assigned__frame_indices": frame_indices,
            "metadata_json": np.array(
                json.dumps(
                    {
                        "source_cluster": cluster_path.name,
                        "state_id": state_id,
                        "generated_at": datetime.utcnow().isoformat(),
                    }
                )
            ),
        }
        if labels_assigned is not None:
            payload["assigned__labels_assigned"] = labels_assigned

        out_path = assign_dir / f"state_{state_id}.npz"
        np.savez_compressed(out_path, **payload)
        assigned_state_paths[state_id] = str(out_path.relative_to(cluster_path.parent))

    assigned_meta_paths: Dict[str, str] = {}
    meta_lookup = {m.get("metastable_id") or m.get("id"): m for m in system.metastable_states or []}
    for meta_id in meta_lookup.keys():
        key = f"meta:{meta_id}"
        entry = predictions.get(key)
        if not isinstance(entry, dict):
            continue
        labels_key = entry.get("labels_halo")
        if not labels_key or labels_key not in data:
            continue
        labels_halo = np.asarray(data[labels_key], dtype=np.int32)
        assigned_key = entry.get("labels_assigned")
        labels_assigned = (
            np.asarray(data[assigned_key], dtype=np.int32)
            if isinstance(assigned_key, str) and assigned_key in data
            else None
        )
        frame_state_ids_key = entry.get("frame_state_ids")
        frame_indices_key = entry.get("frame_indices")
        frame_state_ids = (
            np.asarray(data[frame_state_ids_key], dtype=str)
            if isinstance(frame_state_ids_key, str) and frame_state_ids_key in data
            else np.array([], dtype=str)
        )
        frame_indices = (
            np.asarray(data[frame_indices_key], dtype=np.int64)
            if isinstance(frame_indices_key, str) and frame_indices_key in data
            else np.arange(labels_halo.shape[0], dtype=np.int64)
        )

        payload = {
            "residue_keys": np.asarray(residue_keys),
            "assigned__labels": labels_halo,
            "assigned__cluster_counts": merged_counts,
            "assigned__frame_state_id": frame_state_ids,
            "assigned__frame_indices": frame_indices,
            "assigned__frame_metastable_id": np.array([str(meta_id)]),
            "metadata_json": np.array(
                json.dumps(
                    {
                        "source_cluster": cluster_path.name,
                        "metastable_id": str(meta_id),
                        "generated_at": datetime.utcnow().isoformat(),
                    }
                )
            ),
        }
        if labels_assigned is not None:
            payload["assigned__labels_assigned"] = labels_assigned

        out_path = assign_dir / f"meta_{meta_id}.npz"
        np.savez_compressed(out_path, **payload)
        assigned_meta_paths[str(meta_id)] = str(out_path.relative_to(cluster_path.parent))

    return {
        "assigned_state_paths": assigned_state_paths,
        "assigned_metastable_paths": assigned_meta_paths,
    }


def update_cluster_metadata_with_assignments(
    cluster_path: Path,
    assignments: Dict[str, Dict[str, str]],
) -> None:
    data = np.load(cluster_path, allow_pickle=True)
    payload: Dict[str, Any] = {}
    for key in data.files:
        payload[key] = data[key]
    meta_raw = payload.get("metadata_json")
    meta = {}
    if meta_raw is not None:
        try:
            meta_val = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
            meta = json.loads(str(meta_val))
        except Exception:
            meta = {}
    meta.update(assignments or {})
    payload["metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(cluster_path, **payload)


def _collect_cluster_inputs(
    project_id: str,
    system_id: str,
    metastable_ids: List[str],
) -> Dict[str, Any]:
    unique_meta_ids = list(dict.fromkeys([str(mid) for mid in metastable_ids]))

    store = ProjectStore()
    system = store.get_system(project_id, system_id)
    metastable_lookup = {
        m.get("metastable_id"): {**m, "meta_kind": "metastable"} for m in system.metastable_states or []
    }
    for st in system.states.values():
        metastable_lookup.setdefault(
            st.state_id,
            {
                "metastable_id": st.state_id,
                "metastable_index": 0,
                "macro_state_id": st.state_id,
                "macro_state": st.name,
                "name": st.name,
                "default_name": st.name,
                "representative_pdb": st.pdb_file,
                "meta_kind": "macro",
            },
        )

    residue_keys: List[str] = []
    residue_mapping: Dict[str, str] = {}
    merged_angles_per_residue: List[List[np.ndarray]] = []
    merged_frame_state_ids: List[str] = []
    merged_frame_meta_ids: List[str] = []
    merged_frame_indices: List[int] = []
    contact_edges: set = set()
    contact_sources: List[str] = []

    for meta_id in unique_meta_ids:
        meta = metastable_lookup.get(meta_id)
        if not meta:
            raise ValueError(f"State '{meta_id}' not found on this system.")
        is_macro = meta.get("meta_kind") == "macro"
        meta_index = meta.get("metastable_index")
        if meta_index is None:
            raise ValueError(f"State '{meta_id}' is missing its index.")

        candidate_states = _resolve_states_for_meta(meta, system)
        if not candidate_states:
            raise ValueError(f"No descriptor-ready states found for metastable '{meta_id}'.")
        matched_frames = 0

        for state in candidate_states:
            if not state.descriptor_file:
                continue
            desc_path = store.resolve_path(project_id, system_id, state.descriptor_file)
            features = load_descriptor_npz(desc_path)
            if is_macro:
                frame_count = _infer_frame_count(features)
                if frame_count <= 0:
                    raise ValueError(f"Could not determine frame count for macro-state '{state.state_id}'.")
                labels = np.zeros(frame_count, dtype=np.int32)
            else:
                labels = _extract_labels_for_state(store, project_id, system_id, state, features)

            residue_keys = _coerce_residue_keys(residue_keys, features, state)
            if not residue_keys:
                raise ValueError("Could not determine residue keys for clustering.")
            if not residue_mapping:
                residue_mapping = dict(state.residue_mapping or system.residue_selections_mapping or {})

            if not merged_angles_per_residue:
                merged_angles_per_residue = [[] for _ in residue_keys]

            if labels.shape[0] == 0:
                continue
            mask = labels == int(meta_index)
            if not np.any(mask):
                continue

            matched_indices = np.where(mask)[0]
            for idx in matched_indices:
                merged_frame_state_ids.append(state.state_id)
                merged_frame_meta_ids.append(meta_id)
                merged_frame_indices.append(int(idx))
                for col, key in enumerate(residue_keys):
                    arr = np.asarray(features.get(key))
                    if arr is None or arr.shape[0] != labels.shape[0]:
                        raise ValueError(
                            f"Descriptor array for '{key}' is missing or misaligned in state '{state.state_id}'."
                        )
                    if arr.ndim >= 3:
                        vec = arr[idx, 0, :3]
                    elif arr.ndim == 2:
                        vec = arr[idx, :3]
                    else:
                        vec = arr[idx : idx + 1]
                    vec = np.asarray(vec, dtype=float).reshape(-1)
                    if vec.size < 3:
                        padded = np.zeros(3, dtype=float)
                        padded[: vec.size] = vec
                        vec = padded
                    else:
                        vec = vec[:3]
                    merged_angles_per_residue[col].append(vec)
                matched_frames += 1

        if matched_frames == 0:
            raise ValueError(f"No frames matched metastable '{meta_id}'.")

    return {
        "unique_meta_ids": unique_meta_ids,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "merged_angles_per_residue": merged_angles_per_residue,
        "merged_frame_state_ids": merged_frame_state_ids,
        "merged_frame_meta_ids": merged_frame_meta_ids,
        "merged_frame_indices": merged_frame_indices,
        "contact_edges": contact_edges,
        "contact_sources": contact_sources,
    }


def _build_halo_summary(
    *,
    condition_ids: List[str],
    condition_labels: List[str],
    condition_types: List[str],
    halo_matrix: List[np.ndarray],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not halo_matrix:
        matrix = np.zeros((0, 0), dtype=float)
    else:
        matrix = np.stack(halo_matrix, axis=0)
    payload = {
        "halo_rate__matrix": matrix,
        "halo_rate__condition_ids": np.array(condition_ids, dtype=str),
        "halo_rate__condition_labels": np.array(condition_labels, dtype=str),
        "halo_rate__condition_types": np.array(condition_types, dtype=str),
    }
    meta = {
        "npz_keys": {
            "matrix": "halo_rate__matrix",
            "condition_ids": "halo_rate__condition_ids",
            "condition_labels": "halo_rate__condition_labels",
            "condition_types": "halo_rate__condition_types",
        }
    }
    return payload, meta


def _build_condition_predictions(
    *,
    project_id: str,
    system_id: str,
    residue_keys: List[str],
    dp_models: List[Data],
    density_maxk: int,
    density_z: float | str,
    state_labels: Dict[str, str],
    metastable_labels: Dict[str, str],
    analysis_mode: Optional[str],
    n_jobs: int | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    store = ProjectStore()
    system_meta = store.get_system(project_id, system_id)
    n_residues = len(residue_keys)

    payload: Dict[str, Any] = {}
    predictions_meta: Dict[str, Any] = {}
    halo_condition_ids: List[str] = []
    halo_condition_labels: List[str] = []
    halo_condition_types: List[str] = []
    halo_matrix: List[np.ndarray] = []

    state_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    state_meta_labels: Dict[str, np.ndarray] = {}

    for state_id, state in (system_meta.states or {}).items():
        if not state.descriptor_file:
            continue
        desc_path = store.resolve_path(project_id, system_id, state.descriptor_file)
        if not desc_path.exists():
            continue
        features = load_descriptor_npz(desc_path)
        frame_count = _infer_frame_count(features)
        if frame_count <= 0:
            continue
        labels_halo = np.full((frame_count, n_residues), -1, dtype=np.int32)
        labels_assigned = np.full((frame_count, n_residues), -1, dtype=np.int32)

        for res_idx, key in enumerate(residue_keys):
            angles = _extract_angles_array(features, key)
            if angles is None or angles.shape[0] != frame_count:
                continue
            labels_halo[:, res_idx] = _predict_cluster_adp(
                dp_models[res_idx],
                angles,
                density_maxk=density_maxk,
                halo=True,
                n_jobs=n_jobs,
            )
            labels_assigned[:, res_idx] = _predict_cluster_adp(
                dp_models[res_idx],
                angles,
                density_maxk=density_maxk,
                halo=False,
                n_jobs=n_jobs,
            )

        key_slug = _slug(str(state_id))
        payload[f"state__{key_slug}__labels_halo"] = labels_halo
        payload[f"state__{key_slug}__labels_assigned"] = labels_assigned
        payload[f"state__{key_slug}__frame_indices"] = np.arange(frame_count, dtype=np.int64)
        predictions_meta[f"state:{state_id}"] = {
            "type": "macro",
            "labels_halo": f"state__{key_slug}__labels_halo",
            "labels_assigned": f"state__{key_slug}__labels_assigned",
            "frame_indices": f"state__{key_slug}__frame_indices",
            "frame_count": int(frame_count),
        }
        halo_condition_ids.append(f"state:{state_id}")
        halo_condition_labels.append(state_labels.get(str(state_id), str(state_id)))
        halo_condition_types.append("macro")
        halo_matrix.append(np.mean(labels_halo == -1, axis=0))
        state_predictions[str(state_id)] = {
            "labels_halo": labels_halo,
            "labels_assigned": labels_assigned,
        }

        if analysis_mode != "macro":
            meta_labels = features.get("metastable_labels")
            if meta_labels is None and state.metastable_labels_file:
                label_path = store.resolve_path(project_id, system_id, state.metastable_labels_file)
                if label_path.exists():
                    meta_labels = np.load(label_path)
            if meta_labels is not None:
                state_meta_labels[str(state_id)] = np.asarray(meta_labels).astype(np.int32)

    if analysis_mode != "macro":
        meta_lookup = {m.get("metastable_id") or m.get("id"): m for m in (system_meta.metastable_states or [])}
        for meta_id, meta in meta_lookup.items():
            if meta_id is None:
                continue
            meta_index = meta.get("metastable_index")
            if meta_index is None:
                continue
            labels_list = []
            labels_assigned_list = []
            frame_state_ids: List[str] = []
            frame_indices: List[int] = []
            for state_id, preds in state_predictions.items():
                labels_meta = state_meta_labels.get(state_id)
                if labels_meta is None:
                    continue
                mask = labels_meta == int(meta_index)
                if not np.any(mask):
                    continue
                idxs = np.where(mask)[0]
                labels_list.append(preds["labels_halo"][idxs])
                labels_assigned_list.append(preds["labels_assigned"][idxs])
                frame_state_ids.extend([state_id] * int(idxs.size))
                frame_indices.extend(idxs.tolist())

            if labels_list:
                labels_halo = np.concatenate(labels_list, axis=0)
                labels_assigned = np.concatenate(labels_assigned_list, axis=0)
            else:
                labels_halo = np.zeros((0, n_residues), dtype=np.int32)
                labels_assigned = np.zeros((0, n_residues), dtype=np.int32)

            key_slug = _slug(str(meta_id))
            payload[f"meta__{key_slug}__labels_halo"] = labels_halo
            payload[f"meta__{key_slug}__labels_assigned"] = labels_assigned
            payload[f"meta__{key_slug}__frame_state_ids"] = np.array(frame_state_ids, dtype=str)
            payload[f"meta__{key_slug}__frame_indices"] = np.array(frame_indices, dtype=np.int64)
            payload[f"meta__{key_slug}__frame_metastable_ids"] = np.array([str(meta_id)] * len(frame_indices), dtype=str)
            predictions_meta[f"meta:{meta_id}"] = {
                "type": "metastable",
                "labels_halo": f"meta__{key_slug}__labels_halo",
                "labels_assigned": f"meta__{key_slug}__labels_assigned",
                "frame_state_ids": f"meta__{key_slug}__frame_state_ids",
                "frame_indices": f"meta__{key_slug}__frame_indices",
                "frame_metastable_ids": f"meta__{key_slug}__frame_metastable_ids",
                "frame_count": int(labels_halo.shape[0]),
            }
            halo_condition_ids.append(f"meta:{meta_id}")
            halo_condition_labels.append(metastable_labels.get(str(meta_id), str(meta_id)))
            halo_condition_types.append("metastable")
            if labels_halo.size:
                halo_matrix.append(np.mean(labels_halo == -1, axis=0))
            else:
                halo_matrix.append(np.full(n_residues, np.nan))

    halo_payload, halo_meta = _build_halo_summary(
        condition_ids=halo_condition_ids,
        condition_labels=halo_condition_labels,
        condition_types=halo_condition_types,
        halo_matrix=halo_matrix,
    )
    payload.update(halo_payload)
    return payload, predictions_meta, {"halo_summary": halo_meta}


def _build_state_predictions_from_features(
    *,
    residue_keys: List[str],
    features_by_state: Dict[str, Dict[str, np.ndarray]],
    dp_models: List[Data],
    density_maxk: int,
    density_z: float | str,
    state_labels: Dict[str, str],
    n_jobs: int | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    n_residues = len(residue_keys)
    payload: Dict[str, Any] = {}
    predictions_meta: Dict[str, Any] = {}
    halo_condition_ids: List[str] = []
    halo_condition_labels: List[str] = []
    halo_condition_types: List[str] = []
    halo_matrix: List[np.ndarray] = []

    for state_id, features in features_by_state.items():
        frame_count = _infer_frame_count(features)
        if frame_count <= 0:
            continue
        labels_halo = np.full((frame_count, n_residues), -1, dtype=np.int32)
        labels_assigned = np.full((frame_count, n_residues), -1, dtype=np.int32)

        for res_idx, key in enumerate(residue_keys):
            angles = _extract_angles_array(features, key)
            if angles is None or angles.shape[0] != frame_count:
                continue
            labels_halo[:, res_idx] = _predict_cluster_adp(
                dp_models[res_idx],
                angles,
                density_maxk=density_maxk,
                halo=True,
                n_jobs=n_jobs,
            )
            labels_assigned[:, res_idx] = _predict_cluster_adp(
                dp_models[res_idx],
                angles,
                density_maxk=density_maxk,
                halo=False,
                n_jobs=n_jobs,
            )

        key_slug = _slug(str(state_id))
        payload[f"state__{key_slug}__labels_halo"] = labels_halo
        payload[f"state__{key_slug}__labels_assigned"] = labels_assigned
        payload[f"state__{key_slug}__frame_indices"] = np.arange(frame_count, dtype=np.int64)
        predictions_meta[f"state:{state_id}"] = {
            "type": "macro",
            "labels_halo": f"state__{key_slug}__labels_halo",
            "labels_assigned": f"state__{key_slug}__labels_assigned",
            "frame_indices": f"state__{key_slug}__frame_indices",
            "frame_count": int(frame_count),
        }
        halo_condition_ids.append(f"state:{state_id}")
        halo_condition_labels.append(state_labels.get(str(state_id), str(state_id)))
        halo_condition_types.append("macro")
        halo_matrix.append(np.mean(labels_halo == -1, axis=0))

    halo_payload, halo_meta = _build_halo_summary(
        condition_ids=halo_condition_ids,
        condition_labels=halo_condition_labels,
        condition_types=halo_condition_types,
        halo_matrix=halo_matrix,
    )
    payload.update(halo_payload)
    return payload, predictions_meta, {"halo_summary": halo_meta}


def _build_state_predictions_from_merged(
    *,
    merged_labels_halo: np.ndarray,
    merged_labels_assigned: np.ndarray,
    merged_frame_state_ids: List[str],
    merged_frame_indices: List[int],
    state_frame_counts: Dict[str, int],
    state_labels: Dict[str, str],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    n_frames = merged_labels_halo.shape[0]
    n_residues = merged_labels_halo.shape[1] if merged_labels_halo.ndim == 2 else 0
    payload: Dict[str, Any] = {}
    predictions_meta: Dict[str, Any] = {}
    halo_condition_ids: List[str] = []
    halo_condition_labels: List[str] = []
    halo_condition_types: List[str] = []
    halo_matrix: List[np.ndarray] = []

    state_buffers: Dict[str, Dict[str, np.ndarray]] = {}
    for state_id, count in state_frame_counts.items():
        if count <= 0:
            continue
        state_buffers[state_id] = {
            "labels_halo": np.full((count, n_residues), -1, dtype=np.int32),
            "labels_assigned": np.full((count, n_residues), -1, dtype=np.int32),
        }

    for row in range(n_frames):
        state_id = str(merged_frame_state_ids[row])
        frame_idx = int(merged_frame_indices[row])
        buf = state_buffers.get(state_id)
        if buf is None:
            continue
        if frame_idx < 0 or frame_idx >= buf["labels_halo"].shape[0]:
            continue
        buf["labels_halo"][frame_idx] = merged_labels_halo[row]
        buf["labels_assigned"][frame_idx] = merged_labels_assigned[row]

    for state_id, buf in state_buffers.items():
        labels_halo = buf["labels_halo"]
        labels_assigned = buf["labels_assigned"]
        key_slug = _slug(str(state_id))
        payload[f"state__{key_slug}__labels_halo"] = labels_halo
        payload[f"state__{key_slug}__labels_assigned"] = labels_assigned
        payload[f"state__{key_slug}__frame_indices"] = np.arange(labels_halo.shape[0], dtype=np.int64)
        predictions_meta[f"state:{state_id}"] = {
            "type": "macro",
            "labels_halo": f"state__{key_slug}__labels_halo",
            "labels_assigned": f"state__{key_slug}__labels_assigned",
            "frame_indices": f"state__{key_slug}__frame_indices",
            "frame_count": int(labels_halo.shape[0]),
        }
        halo_condition_ids.append(f"state:{state_id}")
        halo_condition_labels.append(state_labels.get(str(state_id), str(state_id)))
        halo_condition_types.append("macro")
        if labels_halo.size:
            halo_matrix.append(np.mean(labels_halo == -1, axis=0))
        else:
            halo_matrix.append(np.full(n_residues, np.nan))

    halo_payload, halo_meta = _build_halo_summary(
        condition_ids=halo_condition_ids,
        condition_labels=halo_condition_labels,
        condition_types=halo_condition_types,
        halo_matrix=halo_matrix,
    )
    payload.update(halo_payload)
    return payload, predictions_meta, {"halo_summary": halo_meta}


def generate_metastable_cluster_npz(
    project_id: str,
    system_id: str,
    metastable_ids: List[str],
    *,
    output_path: Optional[Path] = None,
    cluster_name: Optional[str] = None,
    max_cluster_frames: Optional[int] = None,
    random_state: int = 0,
    cluster_algorithm: str = "density_peaks",
    density_maxk: Optional[int] = 100,
    density_z: float | str | None = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    n_jobs: int | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Build per-residue cluster labels for selected metastable states and save NPZ.

    Returns the path to the NPZ and a metadata dictionary.
    """
    if not metastable_ids:
        raise ValueError("At least one metastable_id is required.")
    max_cluster_frames_val: Optional[int] = None
    if max_cluster_frames is not None:
        try:
            max_cluster_frames_val = int(max_cluster_frames)
        except Exception as exc:
            raise ValueError("max_cluster_frames must be an integer.") from exc
        if max_cluster_frames_val < 1:
            raise ValueError("max_cluster_frames must be >= 1.")
    algo = (cluster_algorithm or "density_peaks").lower()
    if algo != "density_peaks":
        raise ValueError("Only density_peaks clustering is supported.")
    if density_maxk is None:
        density_maxk_val = 100
    else:
        try:
            density_maxk_val = max(1, int(density_maxk))
        except Exception as exc:
            raise ValueError("density_maxk must be an integer >=1.") from exc
    if density_z is None or (isinstance(density_z, str) and density_z.lower() == "auto"):
        density_z_val: float | str = "auto"
    else:
        try:
            density_z_val = float(density_z)
        except Exception as exc:
            raise ValueError("density_z must be a number or 'auto'.") from exc

    inputs = _collect_cluster_inputs(project_id, system_id, metastable_ids)
    unique_meta_ids = inputs["unique_meta_ids"]
    residue_keys = inputs["residue_keys"]
    residue_mapping = inputs["residue_mapping"]
    merged_angles_per_residue = inputs["merged_angles_per_residue"]
    merged_frame_state_ids = inputs["merged_frame_state_ids"]
    merged_frame_meta_ids = inputs["merged_frame_meta_ids"]
    merged_frame_indices = inputs["merged_frame_indices"]
    contact_edges = inputs["contact_edges"]
    contact_sources = inputs["contact_sources"]
    total_residue_jobs = len(residue_keys)
    completed_residue_jobs = 0
    if progress_callback:
        progress_callback("Clustering residues...", 0, total_residue_jobs)

    store = ProjectStore()
    system_meta = store.get_system(project_id, system_id)
    state_labels, metastable_labels = _build_state_name_maps(system_meta)
    metastable_kinds = _build_metastable_kind_map(system_meta)

    if not merged_angles_per_residue:
        raise ValueError("No frames gathered across the selected metastable states.")

    merged_frame_count = len(merged_frame_state_ids)
    merged_labels_halo = np.zeros((merged_frame_count, len(residue_keys)), dtype=np.int32)
    merged_labels_assigned = np.zeros((merged_frame_count, len(residue_keys)), dtype=np.int32)
    merged_counts = np.zeros(len(residue_keys), dtype=np.int32)
    merged_subsample_indices = None
    if max_cluster_frames_val and merged_frame_count > max_cluster_frames_val:
        merged_subsample_indices = _uniform_subsample_indices(merged_frame_count, max_cluster_frames_val)
    merged_clustered_frames = (
        merged_subsample_indices.size if merged_subsample_indices is not None else merged_frame_count
    )

    dp_models: List[Data] = []

    for col, samples in enumerate(merged_angles_per_residue):
        sample_arr = np.asarray(samples, dtype=float)
        if sample_arr.shape[0] != merged_frame_count:
            raise ValueError("Merged residue samples have inconsistent frame counts.")
        labels_halo, labels_assigned, k, diag, _, dp_data = _cluster_with_subsample(
            sample_arr,
            density_maxk=density_maxk_val,
            density_z=density_z_val,
            max_cluster_frames=max_cluster_frames_val,
            subsample_indices=merged_subsample_indices,
            n_jobs=n_jobs,
        )
        if labels_halo.size == 0:
            merged_labels_halo[:, col] = -1
            merged_labels_assigned[:, col] = -1
            merged_counts[col] = 0
        else:
            merged_labels_halo[:, col] = labels_halo
            merged_labels_assigned[:, col] = labels_assigned
            if np.any(labels_assigned >= 0):
                k = max(k, int(labels_assigned.max()) + 1)
            merged_counts[col] = k
        dp_models.append(dp_data)
        if progress_callback and total_residue_jobs:
            completed_residue_jobs += 1
            progress_callback(
                f"Clustering residues: {completed_residue_jobs}/{total_residue_jobs}",
                completed_residue_jobs,
                total_residue_jobs,
            )

    condition_payload, predictions_meta, extra_meta = _build_condition_predictions(
        project_id=project_id,
        system_id=system_id,
        residue_keys=residue_keys,
        dp_models=dp_models,
        density_maxk=density_maxk_val,
        density_z=density_z_val,
        state_labels=state_labels,
        metastable_labels=metastable_labels,
        analysis_mode=getattr(system_meta, "analysis_mode", None),
        n_jobs=n_jobs,
    )

    # Persist NPZ
    dirs = store.ensure_directories(project_id, system_id)
    cluster_dir = dirs["system_dir"] / "metastable" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    if output_path is not None:
        out_path = Path(output_path)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        suffix = _slug(cluster_name) if cluster_name else "-".join(_slug(mid)[:24] for mid in unique_meta_ids)
        suffix = suffix or "cluster"
        out_path = cluster_dir / f"{suffix}_clusters_{timestamp}.npz"

    metadata = {
        "project_id": project_id,
        "system_id": system_id,
        "generated_at": datetime.utcnow().isoformat(),
        "selected_state_ids": unique_meta_ids,
        "selected_metastable_ids": unique_meta_ids,
        "analysis_mode": getattr(system_meta, "analysis_mode", None),
        "cluster_name": cluster_name,
        "state_labels": state_labels,
        "metastable_labels": metastable_labels,
        "metastable_kinds": metastable_kinds,
        "cluster_algorithm": "density_peaks",
        "cluster_params": {
            "density_maxk": density_maxk_val,
            "density_z": density_z_val,
            "max_cluster_frames": max_cluster_frames_val,
            "random_state": random_state,
        },
        "predictions": predictions_meta,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "random_state": random_state,
        "contact_sources": contact_sources,
        "contact_edge_count": len(contact_edges),
        "merged": {
            "n_frames": merged_frame_count,
            "clustered_frames": int(merged_clustered_frames),
            "npz_keys": {
                "labels": "merged__labels",
                "labels_halo": "merged__labels",
                "labels_assigned": "merged__labels_assigned",
                "cluster_counts": "merged__cluster_counts",
                "frame_state_ids": "merged__frame_state_ids",
                "frame_metastable_ids": "merged__frame_metastable_ids",
                "frame_indices": "merged__frame_indices",
            },
        },
    }
    metadata.update(extra_meta)

    payload: Dict[str, Any] = {
        "residue_keys": np.array(residue_keys),
        "metadata_json": np.array(json.dumps(metadata)),
        "merged__labels": merged_labels_halo,
        "merged__labels_assigned": merged_labels_assigned,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": np.array(merged_frame_state_ids),
        "merged__frame_metastable_ids": np.array(merged_frame_meta_ids),
        "merged__frame_indices": np.array(merged_frame_indices, dtype=np.int64),
    }
    if contact_edges:
        edge_index = np.array(sorted(contact_edges), dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    payload["contact_edge_index"] = edge_index
    # contact_edge_index remains to keep downstream loaders happy
    payload.update(condition_payload)

    np.savez_compressed(out_path, **payload)
    return out_path, metadata


def generate_cluster_npz_from_descriptors(
    descriptor_paths: Sequence[Path],
    *,
    labels: Optional[Sequence[str]] = None,
    eval_descriptor_paths: Optional[Sequence[Path]] = None,
    output_path: Optional[Path] = None,
    max_cluster_frames: Optional[int] = None,
    random_state: int = 0,
    density_maxk: Optional[int] = 100,
    density_z: float | str | None = None,
    n_jobs: int | None = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Cluster local descriptor NPZ files without a project/system context.
    Each descriptor file is treated as a macro-state; labels default to file stems.
    """
    if not descriptor_paths:
        raise ValueError("At least one descriptor NPZ path is required.")
    if labels and len(labels) != len(descriptor_paths):
        raise ValueError("Labels must match the number of descriptor NPZ paths.")
    max_cluster_frames_val: Optional[int] = None
    if max_cluster_frames is not None:
        try:
            max_cluster_frames_val = int(max_cluster_frames)
        except Exception as exc:
            raise ValueError("max_cluster_frames must be an integer.") from exc
        if max_cluster_frames_val < 1:
            raise ValueError("max_cluster_frames must be >= 1.")
    if density_maxk is None:
        density_maxk_val = 100
    else:
        try:
            density_maxk_val = max(1, int(density_maxk))
        except Exception as exc:
            raise ValueError("density_maxk must be an integer >=1.") from exc
    if density_z is None or (isinstance(density_z, str) and density_z.lower() == "auto"):
        density_z_val: float | str = "auto"
    else:
        try:
            density_z_val = float(density_z)
        except Exception as exc:
            raise ValueError("density_z must be a number or 'auto'.") from exc

    resolved_paths = [Path(p) for p in descriptor_paths]
    for path in resolved_paths:
        if not path.exists():
            raise ValueError(f"Descriptor NPZ not found: {path}")
    eval_paths = [Path(p) for p in (eval_descriptor_paths or [])]
    for path in eval_paths:
        if not path.exists():
            raise ValueError(f"Eval descriptor NPZ not found: {path}")

    if labels:
        raw_labels = [str(v).strip() or path.stem for v, path in zip(labels, resolved_paths)]
    else:
        raw_labels = [path.stem for path in resolved_paths]
    raw_eval_labels = [path.stem for path in eval_paths]
    state_ids: List[str] = []
    state_labels: Dict[str, str] = {}
    for idx, raw in enumerate(raw_labels):
        base = _slug(raw) or f"state_{idx + 1}"
        candidate = base
        suffix = 2
        while candidate in state_labels:
            candidate = f"{base}_{suffix}"
            suffix += 1
        state_ids.append(candidate)
        state_labels[candidate] = raw
    eval_state_ids: List[str] = []
    for idx, raw in enumerate(raw_eval_labels):
        base = _slug(raw) or f"eval_{idx + 1}"
        candidate = base
        suffix = 2
        while candidate in state_labels:
            candidate = f"{base}_{suffix}"
            suffix += 1
        eval_state_ids.append(candidate)
        state_labels[candidate] = raw

    features_by_state: Dict[str, Dict[str, np.ndarray]] = {}
    residue_keys: List[str] = []
    for state_id, path in zip(state_ids, resolved_paths):
        features = load_descriptor_npz(path)
        if not features:
            raise ValueError(f"No descriptor data found in '{path}'.")
        keys = sorted(features.keys())
        if not residue_keys:
            residue_keys = keys
        elif residue_keys != keys:
            missing = sorted(set(residue_keys) - set(keys))
            extra = sorted(set(keys) - set(residue_keys))
            raise ValueError(
                f"Descriptor keys mismatch for '{path}'. Missing={missing} Extra={extra}"
            )
        features_by_state[state_id] = features
    for state_id, path in zip(eval_state_ids, eval_paths):
        features = load_descriptor_npz(path)
        if not features:
            raise ValueError(f"No descriptor data found in '{path}'.")
        keys = sorted(features.keys())
        if residue_keys and residue_keys != keys:
            missing = sorted(set(residue_keys) - set(keys))
            extra = sorted(set(keys) - set(residue_keys))
            raise ValueError(
                f"Descriptor keys mismatch for '{path}'. Missing={missing} Extra={extra}"
            )
        features_by_state[state_id] = features

    if not residue_keys:
        raise ValueError("Could not determine residue keys for clustering.")

    merged_frame_state_ids: List[str] = []
    merged_frame_meta_ids: List[str] = []
    merged_frame_indices: List[int] = []
    merged_angles_per_residue: List[List[np.ndarray]] = [[] for _ in residue_keys]

    for state_id, features in features_by_state.items():
        frame_count = _infer_frame_count(features)
        if frame_count <= 0:
            raise ValueError(f"Could not determine frame count for '{state_id}'.")
        for idx in range(frame_count):
            merged_frame_state_ids.append(state_id)
            merged_frame_meta_ids.append(state_id)
            merged_frame_indices.append(int(idx))
            for col, key in enumerate(residue_keys):
                arr = np.asarray(features.get(key))
                if arr is None or arr.shape[0] != frame_count:
                    raise ValueError(
                        f"Descriptor array for '{key}' is missing or misaligned in '{state_id}'."
                    )
                if arr.ndim >= 3:
                    vec = arr[idx, 0, :3]
                elif arr.ndim == 2:
                    vec = arr[idx, :3]
                else:
                    vec = arr[idx : idx + 1]
                vec = np.asarray(vec, dtype=float).reshape(-1)
                if vec.size < 3:
                    padded = np.zeros(3, dtype=float)
                    padded[: vec.size] = vec
                    vec = padded
                else:
                    vec = vec[:3]
                merged_angles_per_residue[col].append(vec)

    merged_frame_count = len(merged_frame_state_ids)
    merged_labels_halo = np.zeros((merged_frame_count, len(residue_keys)), dtype=np.int32)
    merged_labels_assigned = np.zeros((merged_frame_count, len(residue_keys)), dtype=np.int32)
    merged_counts = np.zeros(len(residue_keys), dtype=np.int32)
    merged_subsample_indices = None
    if max_cluster_frames_val and merged_frame_count > max_cluster_frames_val:
        merged_subsample_indices = _uniform_subsample_indices(merged_frame_count, max_cluster_frames_val)
    merged_clustered_frames = (
        merged_subsample_indices.size if merged_subsample_indices is not None else merged_frame_count
    )

    total_residue_jobs = len(residue_keys)
    completed_residue_jobs = 0
    if progress_callback:
        progress_callback("Clustering residues...", 0, total_residue_jobs)

    use_processes = n_jobs is not None and int(n_jobs) > 1
    if use_processes:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
            futures = []
            for col, samples in enumerate(merged_angles_per_residue):
                sample_arr = np.asarray(samples, dtype=float)
                futures.append(
                    executor.submit(
                        _cluster_residue_worker,
                        col,
                        sample_arr,
                        density_maxk_val,
                        density_z_val,
                        max_cluster_frames_val,
                        merged_subsample_indices,
                    )
                )
            for fut in as_completed(futures):
                col, labels_halo, labels_assigned, k = fut.result()
                if labels_halo.size == 0:
                    merged_labels_halo[:, col] = -1
                    merged_labels_assigned[:, col] = -1
                    merged_counts[col] = 0
                else:
                    merged_labels_halo[:, col] = labels_halo
                    merged_labels_assigned[:, col] = labels_assigned
                    if np.any(labels_assigned >= 0):
                        k = max(k, int(labels_assigned.max()) + 1)
                    merged_counts[col] = k
                if progress_callback and total_residue_jobs:
                    completed_residue_jobs += 1
                    progress_callback(
                        f"Clustering residues: {completed_residue_jobs}/{total_residue_jobs}",
                        completed_residue_jobs,
                        total_residue_jobs,
                    )
    else:
        for col, samples in enumerate(merged_angles_per_residue):
            sample_arr = np.asarray(samples, dtype=float)
            labels_halo, labels_assigned, k, _, _, _ = _cluster_with_subsample(
                sample_arr,
                density_maxk=density_maxk_val,
                density_z=density_z_val,
                max_cluster_frames=max_cluster_frames_val,
                subsample_indices=merged_subsample_indices,
                n_jobs=1,
            )
            if labels_halo.size == 0:
                merged_labels_halo[:, col] = -1
                merged_labels_assigned[:, col] = -1
                merged_counts[col] = 0
            else:
                merged_labels_halo[:, col] = labels_halo
                merged_labels_assigned[:, col] = labels_assigned
                if np.any(labels_assigned >= 0):
                    k = max(k, int(labels_assigned.max()) + 1)
                merged_counts[col] = k
            if progress_callback and total_residue_jobs:
                completed_residue_jobs += 1
                progress_callback(
                    f"Clustering residues: {completed_residue_jobs}/{total_residue_jobs}",
                    completed_residue_jobs,
                    total_residue_jobs,
                )

    state_frame_counts = {state_id: _infer_frame_count(features) for state_id, features in features_by_state.items()}
    condition_payload, predictions_meta, extra_meta = _build_state_predictions_from_merged(
        merged_labels_halo=merged_labels_halo,
        merged_labels_assigned=merged_labels_assigned,
        merged_frame_state_ids=merged_frame_state_ids,
        merged_frame_indices=merged_frame_indices,
        state_frame_counts=state_frame_counts,
        state_labels=state_labels,
    )

    if output_path is None:
        output_path = Path.cwd() / f"cluster_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.npz"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.utcnow().isoformat(),
        "analysis_mode": "macro",
        "selected_state_ids": state_ids,
        "selected_metastable_ids": state_ids,
        "state_labels": state_labels,
        "metastable_labels": {},
        "cluster_algorithm": "density_peaks",
        "cluster_params": {
            "density_maxk": density_maxk_val,
            "density_z": density_z_val,
            "max_cluster_frames": max_cluster_frames_val,
            "random_state": random_state,
        },
        "predictions": predictions_meta,
        "residue_keys": residue_keys,
        "residue_mapping": {},
        "random_state": random_state,
        "contact_sources": [],
        "contact_edge_count": 0,
        "merged": {
            "n_frames": merged_frame_count,
            "clustered_frames": int(merged_clustered_frames),
            "npz_keys": {
                "labels": "merged__labels",
                "labels_halo": "merged__labels",
                "labels_assigned": "merged__labels_assigned",
                "cluster_counts": "merged__cluster_counts",
                "frame_state_ids": "merged__frame_state_ids",
                "frame_metastable_ids": "merged__frame_metastable_ids",
                "frame_indices": "merged__frame_indices",
            },
        },
    }
    metadata.update(extra_meta)

    payload: Dict[str, Any] = {
        "residue_keys": np.array(residue_keys),
        "metadata_json": np.array(json.dumps(metadata)),
        "merged__labels": merged_labels_halo,
        "merged__labels_assigned": merged_labels_assigned,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": np.array(merged_frame_state_ids),
        "merged__frame_metastable_ids": np.array(merged_frame_meta_ids),
        "merged__frame_indices": np.array(merged_frame_indices, dtype=np.int64),
        "contact_edge_index": np.zeros((2, 0), dtype=np.int64),
    }
    payload.update(condition_payload)

    np.savez_compressed(output_path, **payload)
    return output_path, metadata


def prepare_cluster_workspace(
    project_id: str,
    system_id: str,
    metastable_ids: List[str],
    *,
    max_cluster_frames: Optional[int],
    random_state: int,
    density_maxk: int,
    density_z: float | str | None,
    work_dir: Path,
) -> Dict[str, Any]:
    """Precompute and persist clustering inputs for fan-out chunk jobs."""
    work_dir.mkdir(parents=True, exist_ok=True)

    inputs = _collect_cluster_inputs(project_id, system_id, metastable_ids)
    residue_keys = inputs["residue_keys"]
    merged_angles_per_residue = inputs["merged_angles_per_residue"]
    merged_frame_state_ids = inputs["merged_frame_state_ids"]
    merged_frame_meta_ids = inputs["merged_frame_meta_ids"]
    merged_frame_indices = inputs["merged_frame_indices"]
    contact_edges = inputs["contact_edges"]
    contact_sources = inputs["contact_sources"]
    residue_mapping = inputs["residue_mapping"]
    unique_meta_ids = inputs["unique_meta_ids"]

    if not merged_angles_per_residue:
        raise ValueError("No frames gathered across the selected metastable states.")

    n_frames = len(merged_frame_state_ids)
    n_residues = len(residue_keys)
    angles_arr = np.stack(
        [np.asarray(samples, dtype=np.float32) for samples in merged_angles_per_residue], axis=1
    )
    angles_path = work_dir / "angles.npy"
    np.save(angles_path, angles_arr)

    np.save(work_dir / "frame_state_ids.npy", np.array(merged_frame_state_ids))
    np.save(work_dir / "frame_meta_ids.npy", np.array(merged_frame_meta_ids))
    np.save(work_dir / "frame_indices.npy", np.array(merged_frame_indices, dtype=np.int64))

    if contact_edges:
        edge_index = np.array(sorted(contact_edges), dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    np.save(work_dir / "contact_edge_index.npy", edge_index)
    manifest = {
        "project_id": project_id,
        "system_id": system_id,
        "selected_state_ids": unique_meta_ids,
        "selected_metastable_ids": unique_meta_ids,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "n_frames": n_frames,
        "n_residues": n_residues,
        "angles_path": "angles.npy",
        "frame_state_ids_path": "frame_state_ids.npy",
        "frame_meta_ids_path": "frame_meta_ids.npy",
        "frame_indices_path": "frame_indices.npy",
        "contact_edge_index_path": "contact_edge_index.npy",
        "contact_sources": contact_sources,
        "cluster_algorithm": "density_peaks",
        "cluster_params": {
            "density_maxk": int(density_maxk),
            "density_z": density_z,
            "max_cluster_frames": int(max_cluster_frames) if max_cluster_frames else None,
            "random_state": int(random_state),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_cluster_chunk(
    work_dir: Path,
    residue_index: int,
) -> Dict[str, Any]:
    """Run clustering for a single residue and persist labels."""
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    angles_path = work_dir / manifest["angles_path"]
    angles = np.load(angles_path, mmap_mode="r")
    n_frames, n_residues, _ = angles.shape
    if residue_index < 0 or residue_index >= n_residues:
        raise ValueError(f"Residue index {residue_index} out of range (0..{n_residues - 1}).")

    params = manifest.get("cluster_params", {})
    sample_arr = np.asarray(angles[:, residue_index, :], dtype=float)
    labels_halo, labels_assigned, k, diag, _, _ = _cluster_with_subsample(
        sample_arr,
        density_maxk=int(params.get("density_maxk", 100)),
        density_z=params.get("density_z", "auto"),
        max_cluster_frames=params.get("max_cluster_frames"),
    )

    out_path = work_dir / f"chunk_{residue_index:04d}.npz"
    payload: Dict[str, Any] = {
        "labels_halo": labels_halo.astype(np.int32),
        "labels_assigned": labels_assigned.astype(np.int32),
        "cluster_count": np.array([int(k)], dtype=np.int32),
    }

    np.savez_compressed(out_path, **payload)
    return {"residue_index": residue_index, "path": str(out_path)}


def reduce_cluster_workspace(work_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """Combine chunk outputs into a final cluster NPZ."""
    manifest = json.loads((work_dir / "manifest.json").read_text())
    project_id = manifest["project_id"]
    system_id = manifest["system_id"]
    residue_keys = manifest["residue_keys"]
    residue_mapping = manifest.get("residue_mapping") or {}
    n_frames = int(manifest.get("n_frames", 0))
    n_residues = int(manifest.get("n_residues", len(residue_keys)))
    selected_meta_ids = manifest.get("selected_state_ids") or manifest.get("selected_metastable_ids") or []
    cluster_params = manifest.get("cluster_params") or {}

    density_maxk_val = int(cluster_params.get("density_maxk", 100))
    density_z_val = cluster_params.get("density_z", "auto")
    max_cluster_frames_val = cluster_params.get("max_cluster_frames")

    angles_path = work_dir / manifest["angles_path"]
    angles = np.load(angles_path, mmap_mode="r")
    if angles.shape[0] != n_frames or angles.shape[1] != n_residues:
        raise ValueError("Angle array does not match manifest dimensions.")

    merged_labels_halo = np.zeros((n_frames, n_residues), dtype=np.int32)
    merged_labels_assigned = np.zeros((n_frames, n_residues), dtype=np.int32)
    merged_counts = np.zeros(n_residues, dtype=np.int32)
    merged_subsample_indices = None
    if max_cluster_frames_val and n_frames > int(max_cluster_frames_val):
        merged_subsample_indices = _uniform_subsample_indices(n_frames, int(max_cluster_frames_val))
    merged_clustered_frames = (
        merged_subsample_indices.size if merged_subsample_indices is not None else n_frames
    )

    dp_models: List[Data] = []

    for idx in range(n_residues):
        sample_arr = np.asarray(angles[:, idx, :], dtype=float)
        labels_halo, labels_assigned, k, diag, _, dp_data = _cluster_with_subsample(
            sample_arr,
            density_maxk=density_maxk_val,
            density_z=density_z_val,
            max_cluster_frames=max_cluster_frames_val,
            subsample_indices=merged_subsample_indices,
        )
        if labels_halo.size == 0:
            merged_labels_halo[:, idx] = -1
            merged_labels_assigned[:, idx] = -1
            merged_counts[idx] = 0
        else:
            merged_labels_halo[:, idx] = labels_halo
            merged_labels_assigned[:, idx] = labels_assigned
            if np.any(labels_assigned >= 0):
                k = max(k, int(labels_assigned.max()) + 1)
            merged_counts[idx] = k
        dp_models.append(dp_data)

    store = ProjectStore()
    dirs = store.ensure_directories(project_id, system_id)
    system_meta = store.get_system(project_id, system_id)
    state_labels, metastable_labels = _build_state_name_maps(system_meta)
    metastable_kinds = _build_metastable_kind_map(system_meta)
    cluster_dir = dirs["system_dir"] / "metastable" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = "-".join(_slug(mid)[:24] for mid in selected_meta_ids) or "metastable"
    out_path = cluster_dir / f"{suffix}_clusters_{timestamp}.npz"

    frame_state_ids = np.load(work_dir / manifest["frame_state_ids_path"], allow_pickle=True)
    frame_meta_ids = np.load(work_dir / manifest["frame_meta_ids_path"], allow_pickle=True)
    frame_indices = np.load(work_dir / manifest["frame_indices_path"], allow_pickle=True)
    contact_edge_index = np.load(work_dir / manifest["contact_edge_index_path"], allow_pickle=True)
    contact_sources = manifest.get("contact_sources") or []

    condition_payload, predictions_meta, extra_meta = _build_condition_predictions(
        project_id=project_id,
        system_id=system_id,
        residue_keys=residue_keys,
        dp_models=dp_models,
        density_maxk=density_maxk_val,
        density_z=density_z_val,
        state_labels=state_labels,
        metastable_labels=metastable_labels,
        analysis_mode=getattr(system_meta, "analysis_mode", None),
    )

    metadata = {
        "project_id": project_id,
        "system_id": system_id,
        "generated_at": datetime.utcnow().isoformat(),
        "selected_state_ids": selected_meta_ids,
        "selected_metastable_ids": selected_meta_ids,
        "analysis_mode": getattr(system_meta, "analysis_mode", None),
        "state_labels": state_labels,
        "metastable_labels": metastable_labels,
        "metastable_kinds": metastable_kinds,
        "cluster_algorithm": "density_peaks",
        "cluster_params": cluster_params,
        "predictions": predictions_meta,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "random_state": cluster_params.get("random_state"),
        "contact_sources": contact_sources,
        "contact_edge_count": int(contact_edge_index.shape[1]) if contact_edge_index.ndim == 2 else 0,
        "merged": {
            "n_frames": n_frames,
            "clustered_frames": int(merged_clustered_frames),
            "npz_keys": {
                "labels": "merged__labels",
                "labels_halo": "merged__labels",
                "labels_assigned": "merged__labels_assigned",
                "cluster_counts": "merged__cluster_counts",
                "frame_state_ids": "merged__frame_state_ids",
                "frame_metastable_ids": "merged__frame_metastable_ids",
                "frame_indices": "merged__frame_indices",
            },
        },
    }
    metadata.update(extra_meta)

    payload = {
        "residue_keys": np.array(residue_keys),
        "metadata_json": np.array(json.dumps(metadata)),
        "merged__labels": merged_labels_halo,
        "merged__labels_assigned": merged_labels_assigned,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": frame_state_ids,
        "merged__frame_metastable_ids": frame_meta_ids,
        "merged__frame_indices": frame_indices,
        "contact_edge_index": contact_edge_index,
    }
    payload.update(condition_payload)

    np.savez_compressed(out_path, **payload)
    return out_path, metadata
