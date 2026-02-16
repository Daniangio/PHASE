"""Cluster per-residue angles inside selected metastable states and persist as NPZ."""

from __future__ import annotations

import json
import re
import uuid
import pickle
from datetime import datetime
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fcluster, linkage
import MDAnalysis as mda
from dadapy import Data

from phase.io.descriptors import load_descriptor_npz
from phase.services.project_store import DescriptorState, ProjectStore, SystemMetadata
from phase.potts.sample_io import SAMPLE_NPZ_FILENAME, save_sample_npz


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
    dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    return dirs["cluster_dir"] / "cluster.npz"


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
        "potts_models": [],
        "samples": [],
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
    ) -> tuple[Data, np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Fit ADP density-peak clustering on embedded samples."""
    if samples.size == 0:
        raise ValueError("No samples provided for density peaks.")
    emb, period = _angles_to_periodic(samples)
    n = emb.shape[0]
    dp_maxk = max(1, min(int(density_maxk), n - 1))
    dp_data = Data(coordinates=emb, maxk=dp_maxk, verbose=False, n_jobs=1, period=period)
    dp_data.compute_distances()
    dp_data.compute_id_2NN()
    dp_data.compute_density_kstarNN()
    if isinstance(density_z, str) and density_z.lower() == "auto":
        dp_data.compute_clustering_ADP()
        density_z_val: float | str = "auto"
    else:
        density_z_val = float(density_z)
        dp_data.compute_clustering_ADP(Z=float(density_z_val))

    labels_assigned = getattr(dp_data, "cluster_assignment", None)
    labels_halo = getattr(dp_data, "cluster_assignment_halo", None)
    if labels_assigned is None or labels_halo is None:
        try:
            sig = inspect.signature(dp_data.compute_clustering_ADP)
        except (TypeError, ValueError):
            sig = None
        if sig and "halo" in sig.parameters:
            if density_z_val == "auto":
                dp_data.compute_clustering_ADP(halo=True)
            else:
                dp_data.compute_clustering_ADP(Z=float(density_z_val), halo=True)
            labels_halo = np.asarray(dp_data.cluster_assignment, dtype=np.int32)
            if density_z_val == "auto":
                dp_data.compute_clustering_ADP(halo=False)
            else:
                dp_data.compute_clustering_ADP(Z=float(density_z_val), halo=False)
            labels_assigned = np.asarray(dp_data.cluster_assignment, dtype=np.int32)
        else:
            labels_assigned = np.asarray(labels_assigned, dtype=np.int32) if labels_assigned is not None else None
            labels_halo = np.asarray(labels_halo, dtype=np.int32) if labels_halo is not None else labels_assigned

    if labels_assigned is None:
        raise ValueError("DADApy clustering did not produce cluster assignments.")
    labels_assigned = np.asarray(labels_assigned, dtype=np.int32)
    labels_halo = np.asarray(labels_halo, dtype=np.int32) if labels_halo is not None else labels_assigned
    k_final = int(dp_data.N_clusters) if hasattr(dp_data, "N_clusters") else int(
        len([c for c in np.unique(labels_assigned) if c >= 0])
    )
    diag: Dict[str, Any] = {
        "density_peaks_method": "dadapy_adp",
        "density_peaks_k": k_final,
        "density_peaks_maxk": dp_maxk,
        "density_peaks_Z": density_z_val,
    }
    return dp_data, labels_assigned, labels_halo, k_final, diag


def _predict_cluster_adp(
    dp_data: Data,
    samples: np.ndarray,
    *,
    density_maxk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Predict ADP cluster labels for new samples using a fitted Data object."""
    emb, _ = _angles_to_periodic(samples)
    maxk_val = max(1, min(int(density_maxk), emb.shape[0] - 1))
    result = dp_data.predict_cluster_ADP(
        emb,
        maxk=maxk_val,
        density_est="kstarNN",
        n_jobs=1,
    )
    labels_assigned = result[0] if isinstance(result, tuple) else result
    labels_halo = result[1] if isinstance(result, tuple) and len(result) > 1 else labels_assigned

    def _coerce_labels(labels: np.ndarray) -> np.ndarray:
        arr = np.asarray(labels, dtype=np.int32)
        if arr.ndim > 1:
            if arr.shape[1] == 0:
                return arr.reshape(-1)
            return arr[:, 0]
        return arr

    return _coerce_labels(labels_assigned), _coerce_labels(labels_halo)


def _angles_to_periodic(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center angle triplets into [0, 2pi) and return periodicity vector."""
    angles = samples[:, :3]
    two_pi = 2.0 * np.pi
    centered = np.mod(angles, two_pi)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    period = np.full(centered.shape[1], two_pi, dtype=np.float64)
    return centered, period


def _angles_to_circular_features(samples: np.ndarray) -> np.ndarray:
    """Map angle triplets to a wrap-safe embedding using sin/cos per angle."""
    centered, _ = _angles_to_periodic(samples)
    sin_part = np.sin(centered)
    cos_part = np.cos(centered)
    return np.concatenate([sin_part, cos_part], axis=1).astype(np.float64, copy=False)


def _gaussian_logpdf_matrix(
    X: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    covariance_type: str,
    reg_covar: float,
) -> np.ndarray:
    """Return component log densities log N(x | mean_k, cov_k) for all rows/components."""
    X = np.asarray(X, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    n, d = X.shape
    k = means.shape[0]
    out = np.full((n, k), -np.inf, dtype=np.float64)
    log2pi = d * np.log(2.0 * np.pi)
    reg = float(max(reg_covar, 1e-12))

    cov_kind = str(covariance_type or "full").lower()
    if cov_kind == "diag":
        for j in range(k):
            var = np.asarray(covariances[j], dtype=np.float64).reshape(-1)
            if var.size != d:
                var = np.resize(var, d)
            var = np.maximum(var, reg)
            diff = X - means[j]
            quad = np.sum((diff * diff) / var[None, :], axis=1)
            logdet = float(np.sum(np.log(var)))
            out[:, j] = -0.5 * (log2pi + logdet + quad)
        return out

    for j in range(k):
        cov = np.asarray(covariances[j], dtype=np.float64)
        if cov.shape != (d, d):
            eye = np.eye(d, dtype=np.float64)
            cov = eye * reg
        cov = cov + (np.eye(d, dtype=np.float64) * reg)
        try:
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise np.linalg.LinAlgError("non-positive definite covariance")
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov = cov + (np.eye(d, dtype=np.float64) * (reg * 10.0))
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                continue
            inv = np.linalg.inv(cov)

        diff = X - means[j]
        quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
        out[:, j] = -0.5 * (log2pi + float(logdet) + quad)
    return out


def _predict_cluster_frozen_gmm(
    model: Dict[str, Any],
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict assigned/halo labels using a frozen Gaussian approximation."""
    features = _angles_to_circular_features(samples)
    means = np.asarray(model.get("means", []), dtype=np.float64)
    covariances = np.asarray(model.get("covariances", []), dtype=np.float64)
    weights = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    thresholds = np.asarray(model.get("thresholds_logpdf", []), dtype=np.float64).reshape(-1)
    covariance_type = str(model.get("covariance_type") or "full").lower()
    reg_covar = float(model.get("reg_covar", 1e-5))

    n = features.shape[0]
    if means.ndim != 2 or means.shape[0] == 0:
        empty = np.full((n,), -1, dtype=np.int32)
        return empty, empty
    k = means.shape[0]
    if weights.size != k:
        weights = np.full((k,), 1.0 / float(k), dtype=np.float64)
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    if thresholds.size != k:
        thresholds = np.full((k,), -np.inf, dtype=np.float64)

    logpdf = _gaussian_logpdf_matrix(
        features,
        means,
        covariances,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
    )
    logpost = logpdf + np.log(weights[None, :])
    assigned = np.argmax(logpost, axis=1).astype(np.int32, copy=False)
    halo = assigned.copy()
    chosen = logpdf[np.arange(n), assigned]
    halo[chosen < thresholds[assigned]] = -1
    return assigned, halo


def _fit_hierarchical_frozen_gmm(
    samples: np.ndarray,
    *,
    n_clusters: Optional[int],
    cluster_selection_mode: str,
    inconsistent_threshold: Optional[float],
    inconsistent_depth: int,
    linkage_method: str,
    covariance_type: str,
    reg_covar: float,
    halo_percentile: float,
    max_cluster_frames: Optional[int],
    subsample_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, int, Dict[str, Any], int, Dict[str, Any]]:
    """Cluster a residue with scipy hierarchical and build a frozen Gaussian predictor."""
    if samples.ndim != 2 or samples.shape[0] == 0:
        empty = np.array([], dtype=np.int32)
        model = {"kind": "frozen_gmm", "means": [], "covariances": [], "weights": []}
        return empty, empty, 0, {"algorithm": "hierarchical_gmm"}, 0, model

    features = _angles_to_circular_features(samples)
    n_frames = features.shape[0]
    if max_cluster_frames and max_cluster_frames > 0 and n_frames > int(max_cluster_frames):
        sub_idx = (
            _uniform_subsample_indices(n_frames, int(max_cluster_frames))
            if subsample_indices is None
            else np.asarray(subsample_indices, dtype=int)
        )
    else:
        sub_idx = np.arange(n_frames, dtype=int)

    X_train = features[sub_idx]
    n_train = X_train.shape[0]
    d = X_train.shape[1]
    selection_mode = str(cluster_selection_mode or "maxclust").strip().lower()
    if selection_mode not in {"maxclust", "inconsistent"}:
        raise ValueError("cluster_selection_mode must be one of: maxclust, inconsistent.")
    k_target = max(1, min(int(n_clusters), n_train)) if n_clusters is not None else max(1, min(2, n_train))
    if n_train <= 1:
        labels_train = np.zeros((n_train,), dtype=np.int32)
    else:
        Z = linkage(X_train, method=str(linkage_method or "ward").lower(), metric="euclidean")
        if selection_mode == "maxclust":
            labels_train = (fcluster(Z, t=k_target, criterion="maxclust") - 1).astype(np.int32, copy=False)
        else:
            threshold = float(inconsistent_threshold if inconsistent_threshold is not None else 1.0)
            depth = max(1, int(inconsistent_depth))
            labels_train = (
                fcluster(Z, t=threshold, criterion="inconsistent", depth=depth) - 1
            ).astype(np.int32, copy=False)

    uniq = np.array(sorted(int(v) for v in np.unique(labels_train)), dtype=np.int32)
    if uniq.size == 0:
        uniq = np.array([0], dtype=np.int32)
        labels_train = np.zeros((n_train,), dtype=np.int32)
    remap = {int(old): int(new) for new, old in enumerate(uniq.tolist())}
    labels_train = np.asarray([remap[int(v)] for v in labels_train], dtype=np.int32)
    k = int(len(uniq))

    means = np.zeros((k, d), dtype=np.float64)
    if str(covariance_type or "full").lower() == "diag":
        covs = np.zeros((k, d), dtype=np.float64)
    else:
        covs = np.zeros((k, d, d), dtype=np.float64)
    weights = np.zeros((k,), dtype=np.float64)
    thresholds = np.full((k,), -np.inf, dtype=np.float64)
    reg = float(max(reg_covar, 1e-8))
    halo_q = float(np.clip(halo_percentile, 0.0, 100.0))

    cov_kind = str(covariance_type or "full").lower()
    for j in range(k):
        mask = labels_train == j
        Xj = X_train[mask]
        if Xj.shape[0] == 0:
            continue
        means[j] = np.mean(Xj, axis=0)
        weights[j] = float(Xj.shape[0]) / float(n_train)
        if cov_kind == "diag":
            if Xj.shape[0] <= 1:
                var = np.full((d,), reg, dtype=np.float64)
            else:
                var = np.var(Xj, axis=0, ddof=1)
                var = np.maximum(var, reg)
            covs[j] = var
        else:
            if Xj.shape[0] <= 1:
                cov = np.eye(d, dtype=np.float64) * reg
            else:
                cov = np.cov(Xj, rowvar=False, ddof=1)
                if cov.shape != (d, d):
                    cov = np.eye(d, dtype=np.float64) * reg
                cov = cov + (np.eye(d, dtype=np.float64) * reg)
            covs[j] = cov

    model = {
        "kind": "frozen_gmm",
        "feature_space": "sin_cos_3angles_v1",
        "covariance_type": cov_kind,
        "cluster_selection_mode": selection_mode,
        "inconsistent_threshold": float(inconsistent_threshold) if inconsistent_threshold is not None else None,
        "inconsistent_depth": int(max(1, int(inconsistent_depth))),
        "means": means.tolist(),
        "covariances": covs.tolist(),
        "weights": weights.tolist(),
        "thresholds_logpdf": thresholds.tolist(),
        "reg_covar": reg,
        "halo_percentile": halo_q,
        "linkage_method": str(linkage_method or "ward").lower(),
        "n_clusters": int(k),
    }

    train_assigned, _ = _predict_cluster_frozen_gmm(model, samples[sub_idx])
    train_features = _angles_to_circular_features(samples[sub_idx])
    train_logpdf = _gaussian_logpdf_matrix(
        train_features,
        np.asarray(model["means"], dtype=np.float64),
        np.asarray(model["covariances"], dtype=np.float64),
        covariance_type=cov_kind,
        reg_covar=reg,
    )
    thresholds_arr = np.full((k,), -np.inf, dtype=np.float64)
    for j in range(k):
        mask = train_assigned == j
        if not np.any(mask):
            continue
        vals = train_logpdf[mask, j]
        thresholds_arr[j] = float(np.percentile(vals, halo_q))
    model["thresholds_logpdf"] = thresholds_arr.tolist()

    labels_assigned, labels_halo = _predict_cluster_frozen_gmm(model, samples)
    diag = {
        "algorithm": "hierarchical_gmm",
        "n_clusters": int(k),
        "cluster_selection_mode": selection_mode,
        "inconsistent_threshold": float(inconsistent_threshold) if inconsistent_threshold is not None else None,
        "inconsistent_depth": int(max(1, int(inconsistent_depth))),
        "linkage_method": str(linkage_method or "ward").lower(),
        "covariance_type": cov_kind,
        "reg_covar": reg,
        "halo_percentile": halo_q,
        "subsampled": bool(sub_idx.size != n_frames),
        "subsample_size": int(sub_idx.size),
        "total_frames": int(n_frames),
    }
    return labels_halo, labels_assigned, int(k), diag, int(sub_idx.size), model


def _predict_labels_with_model(
    model: Any,
    samples: np.ndarray,
    *,
    density_maxk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels for one residue with ADP, frozen GMM, or legacy ADP pair models."""
    if model is None:
        n = int(samples.shape[0]) if samples.ndim > 0 else 0
        empty = np.full((n,), -1, dtype=np.int32)
        return empty, empty
    if isinstance(model, Data):
        return _predict_cluster_adp(model, samples, density_maxk=density_maxk)
    if isinstance(model, dict):
        kind = str(model.get("kind") or "").lower()
        if kind == "frozen_gmm":
            return _predict_cluster_frozen_gmm(model, samples)
        if kind == "adp_legacy_models":
            model_assigned = model.get("model_assigned")
            model_halo = model.get("model_halo")
            assigned, _ = _predict_cluster_adp(model_assigned, samples, density_maxk=density_maxk)
            _, halo = _predict_cluster_adp(model_halo, samples, density_maxk=density_maxk)
            return assigned, halo
    raise ValueError(f"Unsupported residue model type: {type(model)}")


def _cluster_residue_samples(
    samples: np.ndarray,
    *,
    density_maxk: int,
    density_z: float | str,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any], Data]:
    """Cluster angles with ADP density peaks."""
    if samples.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            0,
            {},
            Data(coordinates=np.zeros((1, 1))),
        )
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[1] < 3:
        raise ValueError("Residue samples must be (n_frames, >=3) shaped.")

    dp_data, labels_assigned, labels_halo, k_final, diagnostics = _fit_density_peaks(
        samples,
        density_maxk=density_maxk,
        density_z=density_z,
    )
    return labels_halo, labels_assigned, int(k_final), diagnostics, dp_data


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
    subsample_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any], int, Data]:
    """Fit ADP on a subsample if requested, then predict labels on all frames."""
    def _coerce_1d(labels: np.ndarray) -> np.ndarray:
        arr = np.asarray(labels, dtype=np.int32)
        if arr.ndim > 1:
            if arr.shape[1] == 0:
                return arr.reshape(-1)
            return arr[:, 0]
        return arr
    n_frames = samples.shape[0]
    if not max_cluster_frames or max_cluster_frames <= 0 or n_frames <= max_cluster_frames:
        labels_halo, labels_assigned, k_final, diag, dp_data = _cluster_residue_samples(
            samples,
            density_maxk=density_maxk,
            density_z=density_z,
        )
        labels_halo = _coerce_1d(labels_halo)
        labels_assigned = _coerce_1d(labels_assigned)
        diag["subsampled"] = False
        diag["subsample_size"] = int(n_frames)
        diag["total_frames"] = int(n_frames)
        return labels_halo, labels_assigned, k_final, diag, int(n_frames), dp_data

    subsample_indices = (
        _uniform_subsample_indices(n_frames, int(max_cluster_frames))
        if subsample_indices is None
        else subsample_indices
    )
    subsample_indices = np.asarray(subsample_indices, dtype=int)
    sub_samples = samples[subsample_indices]
    _, _, k_final, diag, dp_data = _cluster_residue_samples(
        sub_samples,
        density_maxk=density_maxk,
        density_z=density_z,
    )
    labels_assigned, labels_halo = _predict_cluster_adp(
        dp_data,
        samples,
        density_maxk=density_maxk,
    )
    labels_halo = _coerce_1d(labels_halo)
    labels_assigned = _coerce_1d(labels_assigned)
    diag["subsampled"] = True
    diag["subsample_size"] = int(subsample_indices.size)
    diag["total_frames"] = int(n_frames)
    return labels_halo, labels_assigned, k_final, diag, int(subsample_indices.size), dp_data


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
    )
    return col, labels_halo, labels_assigned, k


def _cluster_residue_worker_with_models(
    col: int,
    samples: np.ndarray,
    density_maxk: int,
    density_z: float | str,
    max_cluster_frames: Optional[int],
    subsample_indices: Optional[np.ndarray],
) -> Tuple[int, np.ndarray, np.ndarray, int, Data]:
    labels_halo, labels_assigned, k, _, _, dp_data = _cluster_with_subsample(
        samples,
        density_maxk=density_maxk,
        density_z=density_z,
        max_cluster_frames=max_cluster_frames,
        subsample_indices=subsample_indices,
    )
    return col, labels_halo, labels_assigned, k, dp_data


def _predict_residue_worker(
    res_idx: int,
    samples: np.ndarray,
    model: Any,
    density_maxk: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    labels_assigned, labels_halo = _predict_labels_with_model(
        model,
        samples,
        density_maxk=density_maxk,
    )
    return res_idx, labels_assigned, labels_halo


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
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
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

    assign_root = output_dir or (cluster_path.parent / "samples")
    assign_root.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
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

        sample_id = str(uuid.uuid4())
        sample_dir = assign_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / SAMPLE_NPZ_FILENAME
        labels_primary = labels_assigned if labels_assigned is not None else labels_halo
        keep = np.all(labels_primary >= 0, axis=1)
        if keep.size and not np.all(keep):
            labels_primary = labels_primary[keep]
            labels_halo = labels_halo[keep]
            frame_indices = frame_indices[keep]
        save_sample_npz(
            out_path,
            labels=labels_primary,
            labels_halo=labels_halo,
            frame_indices=frame_indices,
            frame_state_ids=np.full(frame_indices.shape[0], str(state_id), dtype=str),
        )
        assigned_state_paths[state_id] = str(out_path.relative_to(cluster_path.parent))
        system_dir = cluster_path.parent.parent.parent
        try:
            rel_system = str(out_path.relative_to(system_dir))
        except Exception:
            rel_system = str(out_path)
        sample_meta = {
            "sample_id": sample_id,
            "name": f"MD {getattr(state, 'name', state_id)}",
            "type": "md_eval",
            "method": "md_eval",
            "source": "clustering",
            "state_id": state_id,
            "created_at": datetime.utcnow().isoformat(),
            "path": rel_system,
            "paths": {"summary_npz": rel_system},
            "params": {},
        }
        (sample_dir / "sample_metadata.json").write_text(json.dumps(sample_meta, indent=2), encoding="utf-8")
        samples.append(
            {
                "sample_id": sample_id,
                "name": f"MD {getattr(state, 'name', state_id)}",
                "type": "md_eval",
                "method": "md_eval",
                "state_id": state_id,
                "path": str(out_path.relative_to(cluster_path.parent)),
                "created_at": datetime.utcnow().isoformat(),
                "summary": {"state_id": state_id},
            }
        )

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

        sample_id = str(uuid.uuid4())
        sample_dir = assign_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / SAMPLE_NPZ_FILENAME
        labels_primary = labels_assigned if labels_assigned is not None else labels_halo
        frame_state_ids_arr = frame_state_ids
        if frame_state_ids_arr.size != frame_indices.shape[0]:
            frame_state_ids_arr = np.full(frame_indices.shape[0], "", dtype=str)
        keep = np.all(labels_primary >= 0, axis=1)
        if keep.size and not np.all(keep):
            labels_primary = labels_primary[keep]
            labels_halo = labels_halo[keep]
            frame_indices = frame_indices[keep]
            frame_state_ids_arr = frame_state_ids_arr[keep] if frame_state_ids_arr.size == keep.shape[0] else frame_state_ids_arr
        save_sample_npz(
            out_path,
            labels=labels_primary,
            labels_halo=labels_halo,
            frame_indices=frame_indices,
            frame_state_ids=frame_state_ids_arr,
        )
        assigned_meta_paths[str(meta_id)] = str(out_path.relative_to(cluster_path.parent))
        system_dir = cluster_path.parent.parent.parent
        try:
            rel_system = str(out_path.relative_to(system_dir))
        except Exception:
            rel_system = str(out_path)
        sample_meta = {
            "sample_id": sample_id,
            "name": f"MD {meta_lookup.get(meta_id, {}).get('label') or meta_id}",
            "type": "md_eval",
            "method": "md_eval",
            "source": "clustering",
            "metastable_id": str(meta_id),
            "created_at": datetime.utcnow().isoformat(),
            "path": rel_system,
            "paths": {"summary_npz": rel_system},
            "params": {},
        }
        (sample_dir / "sample_metadata.json").write_text(json.dumps(sample_meta, indent=2), encoding="utf-8")
        meta_label = meta_lookup.get(meta_id, {}).get("label") if isinstance(meta_id, str) else None
        samples.append(
            {
                "sample_id": sample_id,
                "name": f"MD {meta_label or meta_id}",
                "type": "md_eval",
                "method": "md_eval",
                "metastable_id": str(meta_id),
                "path": str(out_path.relative_to(cluster_path.parent)),
                "created_at": datetime.utcnow().isoformat(),
                "summary": {"metastable_id": str(meta_id)},
            }
        )

    return {
        "assigned_state_paths": assigned_state_paths,
        "assigned_metastable_paths": assigned_meta_paths,
        "samples": samples,
    }


def evaluate_state_with_models(
    project_id: str,
    system_id: str,
    cluster_id: str,
    state_id: str,
    *,
    store: ProjectStore | None = None,
    sample_id: str | None = None,
) -> Dict[str, Any]:
    store = store or ProjectStore()
    system = store.get_system(project_id, system_id)
    state = system.states.get(state_id)
    if not state or not state.descriptor_file:
        raise ValueError(f"State '{state_id}' is missing descriptors.")

    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    cluster_path = cluster_dirs["cluster_dir"] / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found for cluster_id='{cluster_id}'.")
    data = np.load(cluster_path, allow_pickle=True)
    meta_raw = data.get("metadata_json")
    if meta_raw is None:
        raise ValueError("Cluster NPZ missing metadata_json.")
    try:
        meta_val = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
        meta = json.loads(str(meta_val))
    except Exception as exc:
        raise ValueError(f"Failed to parse cluster metadata_json: {exc}") from exc

    residue_keys = [str(k) for k in meta.get("residue_keys", [])]
    cluster_params = meta.get("cluster_params") or {}
    density_maxk = int(cluster_params.get("density_maxk", 100))
    if not residue_keys:
        raise ValueError("Cluster NPZ metadata is missing residue keys for evaluation.")
    residue_models = _load_residue_models_from_metadata(
        store=store,
        project_id=project_id,
        system_id=system_id,
        residue_keys=residue_keys,
        metadata=meta,
    )
    if not any(m is not None for m in residue_models):
        raise ValueError("Cluster NPZ is missing predictor models for evaluation.")

    descriptor_path = store.resolve_path(project_id, system_id, state.descriptor_file)
    feature_dict = load_descriptor_npz(descriptor_path)

    n_frames = None
    labels_halo = np.full((feature_dict[residue_keys[0]].shape[0], len(residue_keys)), -1, dtype=np.int32)
    labels_assigned = np.full_like(labels_halo, -1)
    cluster_counts = np.zeros(len(residue_keys), dtype=np.int32)

    for idx, key in enumerate(residue_keys):
        if key not in feature_dict:
            raise ValueError(f"Descriptor missing residue key '{key}'.")
        arr = feature_dict[key]
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Descriptor for '{key}' must be (n_frames, 1, >=3).")
        samples = np.asarray(arr[:, 0, :], dtype=float)
        if n_frames is None:
            n_frames = samples.shape[0]
        if samples.shape[0] != n_frames:
            raise ValueError("Descriptor frame counts are inconsistent across residues.")
        model = residue_models[idx] if idx < len(residue_models) else None
        if model is None:
            continue
        labels_assigned[:, idx], labels_halo[:, idx] = _predict_labels_with_model(
            model,
            samples,
            density_maxk=density_maxk,
        )
        if np.any(labels_assigned[:, idx] >= 0):
            cluster_counts[idx] = int(labels_assigned[:, idx].max()) + 1

    sample_id = str(sample_id) if sample_id else str(uuid.uuid4())
    sample_dir = cluster_dirs["samples_dir"] / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / SAMPLE_NPZ_FILENAME
    frame_indices = np.arange(labels_halo.shape[0], dtype=np.int64)
    keep = np.all(labels_assigned >= 0, axis=1)
    if keep.size and not np.all(keep):
        labels_assigned = labels_assigned[keep]
        labels_halo = labels_halo[keep]
        frame_indices = frame_indices[keep]
    save_sample_npz(
        out_path,
        labels=labels_assigned,
        labels_halo=labels_halo,
        frame_indices=frame_indices,
        frame_state_ids=np.full(frame_indices.shape[0], str(state_id), dtype=str),
    )
    try:
        rel = str(out_path.relative_to(cluster_dirs["system_dir"]))
    except Exception:
        rel = str(out_path)
    sample_entry = {
        "sample_id": sample_id,
        "name": f"MD {state.name or state_id}",
        "type": "md_eval",
        "method": "md_eval",
        "source": "clustering",
        "state_id": state_id,
        "created_at": datetime.utcnow().isoformat(),
        "path": rel,
        "paths": {"summary_npz": rel},
        "params": {},
    }
    # Sample metadata is persisted via ProjectStore.save_system(). We return the full entry so
    # both offline scripts and the webserver can append/update it in cluster metadata.
    return sample_entry


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


def _load_cluster_payload(cluster_path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    data = np.load(cluster_path, allow_pickle=True)
    payload: Dict[str, Any] = {key: data[key] for key in data.files}
    meta: Dict[str, Any] = {}
    meta_raw = payload.get("metadata_json")
    if meta_raw is not None:
        try:
            meta_val = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
            meta = json.loads(str(meta_val))
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return payload, meta


def _save_cluster_payload(cluster_path: Path, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(cluster_path, **payload)


def _resolve_patch_residue_indices(
    residue_keys: List[str],
    *,
    residue_indices: Optional[List[int]] = None,
    residue_key_subset: Optional[List[str]] = None,
) -> List[int]:
    idxs: List[int] = []
    if residue_indices:
        for ridx in residue_indices:
            ir = int(ridx)
            if ir < 0 or ir >= len(residue_keys):
                raise ValueError(f"Residue index out of range: {ir}")
            idxs.append(ir)
    if residue_key_subset:
        key_to_idx = {str(k): i for i, k in enumerate(residue_keys)}
        for key in residue_key_subset:
            s = str(key).strip()
            if not s:
                continue
            if s not in key_to_idx:
                raise ValueError(f"Residue key not found in cluster: {s}")
            idxs.append(key_to_idx[s])
    out = sorted(set(int(i) for i in idxs))
    if not out:
        raise ValueError("Select at least one residue to patch.")
    return out


def _build_source_to_target_row_map(
    *,
    source_state_ids: Sequence[Any],
    source_frame_indices: Sequence[Any],
    target_state_ids: Sequence[Any],
    target_frame_indices: Sequence[Any],
) -> np.ndarray:
    source_lookup: Dict[tuple[str, int], List[int]] = {}
    for i, (sid, fidx) in enumerate(zip(source_state_ids, source_frame_indices)):
        key = (str(sid), int(fidx))
        source_lookup.setdefault(key, []).append(i)
    source_ptr: Dict[tuple[str, int], int] = {k: 0 for k in source_lookup.keys()}
    mapped = np.full((len(target_state_ids),), -1, dtype=np.int64)
    for i, (sid, fidx) in enumerate(zip(target_state_ids, target_frame_indices)):
        key = (str(sid), int(fidx))
        bucket = source_lookup.get(key)
        if not bucket:
            continue
        ptr = source_ptr[key]
        if ptr >= len(bucket):
            continue
        mapped[i] = int(bucket[ptr])
        source_ptr[key] = ptr + 1
    return mapped


def _prefix_condition_payload(
    condition_payload: Dict[str, Any],
    predictions_meta: Dict[str, Any],
    extra_meta: Dict[str, Any],
    *,
    prefix: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    key_map: Dict[str, str] = {}
    prefixed_payload: Dict[str, Any] = {}
    for key, value in (condition_payload or {}).items():
        new_key = f"{prefix}{key}"
        key_map[key] = new_key
        prefixed_payload[new_key] = value

    prefixed_predictions: Dict[str, Any] = {}
    for pred_key, entry in (predictions_meta or {}).items():
        if not isinstance(entry, dict):
            continue
        new_entry: Dict[str, Any] = {}
        for k, v in entry.items():
            if isinstance(v, str) and v in key_map:
                new_entry[k] = key_map[v]
            else:
                new_entry[k] = v
        prefixed_predictions[str(pred_key)] = new_entry

    prefixed_extra = dict(extra_meta or {})
    halo_summary = prefixed_extra.get("halo_summary")
    if isinstance(halo_summary, dict):
        npz_keys = halo_summary.get("npz_keys")
        if isinstance(npz_keys, dict):
            halo_summary["npz_keys"] = {
                str(k): key_map.get(str(v), str(v))
                for k, v in npz_keys.items()
            }
        prefixed_extra["halo_summary"] = halo_summary

    return prefixed_payload, prefixed_predictions, prefixed_extra, key_map


def _load_residue_models_from_metadata(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    residue_keys: List[str],
    metadata: Dict[str, Any],
) -> List[Any]:
    n_res = len(residue_keys)
    models: List[Any] = [None] * n_res
    model_paths = metadata.get("model_paths") or []
    model_paths_halo = metadata.get("model_paths_halo") or []
    model_paths_assigned = metadata.get("model_paths_assigned") or []

    if isinstance(model_paths, list) and len(model_paths) == n_res:
        for i in range(n_res):
            rel = model_paths[i]
            if not rel:
                continue
            p = store.resolve_path(project_id, system_id, str(rel))
            if not p.exists():
                continue
            with open(p, "rb") as inp:
                models[i] = pickle.load(inp)
    elif (
        isinstance(model_paths_halo, list)
        and isinstance(model_paths_assigned, list)
        and len(model_paths_halo) == n_res
        and len(model_paths_assigned) == n_res
    ):
        for i in range(n_res):
            rel_h = model_paths_halo[i]
            rel_a = model_paths_assigned[i]
            if not rel_h or not rel_a:
                continue
            ph = store.resolve_path(project_id, system_id, str(rel_h))
            pa = store.resolve_path(project_id, system_id, str(rel_a))
            if not ph.exists() or not pa.exists():
                continue
            with open(ph, "rb") as inp:
                mh = pickle.load(inp)
            with open(pa, "rb") as inp:
                ma = pickle.load(inp)
            models[i] = {"kind": "adp_legacy_models", "model_halo": mh, "model_assigned": ma}

    overrides = metadata.get("residue_model_overrides") or {}
    if isinstance(overrides, dict):
        for k, spec in overrides.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if idx < 0 or idx >= n_res:
                continue
            if isinstance(spec, dict) and str(spec.get("kind") or "").lower() == "frozen_gmm":
                models[idx] = spec
    return models


def list_cluster_patches(
    project_id: str,
    system_id: str,
    cluster_id: str,
    *,
    store: ProjectStore | None = None,
) -> Dict[str, Any]:
    store = store or ProjectStore()
    cluster_path = store.ensure_cluster_directories(project_id, system_id, cluster_id)["cluster_dir"] / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found for cluster_id='{cluster_id}'.")
    _, meta = _load_cluster_payload(cluster_path)
    patches = meta.get("cluster_patches") or []
    out = []
    for item in patches:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "patch_id": str(item.get("patch_id") or ""),
                "name": item.get("name"),
                "status": item.get("status") or "preview",
                "created_at": item.get("created_at"),
                "residue_keys": (item.get("residues") or {}).get("keys") or [],
                "algorithm": item.get("algorithm") or "hierarchical_gmm",
            }
        )
    return {"cluster_id": cluster_id, "patches": out}


def create_cluster_residue_patch(
    project_id: str,
    system_id: str,
    cluster_id: str,
    *,
    residue_indices: Optional[List[int]] = None,
    residue_keys: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    cluster_selection_mode: str = "maxclust",
    inconsistent_threshold: Optional[float] = None,
    inconsistent_depth: int = 2,
    linkage_method: str = "ward",
    covariance_type: str = "full",
    reg_covar: float = 1e-5,
    halo_percentile: float = 5.0,
    max_cluster_frames: Optional[int] = None,
    patch_name: Optional[str] = None,
    predict_jobs: Optional[int] = None,
    store: ProjectStore | None = None,
) -> Dict[str, Any]:
    store = store or ProjectStore()
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    cluster_path = cluster_dirs["cluster_dir"] / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found for cluster_id='{cluster_id}'.")

    payload, meta = _load_cluster_payload(cluster_path)
    if "merged__labels" not in payload or "merged__labels_assigned" not in payload:
        raise ValueError("Cluster NPZ missing merged label arrays.")

    residue_key_list = [str(v) for v in (meta.get("residue_keys") or payload.get("residue_keys", []))]
    if not residue_key_list:
        raise ValueError("Cluster metadata missing residue_keys.")
    patch_indices = _resolve_patch_residue_indices(
        residue_key_list,
        residue_indices=residue_indices,
        residue_key_subset=residue_keys,
    )

    selected_ids = [str(v) for v in (meta.get("selected_state_ids") or meta.get("selected_metastable_ids") or [])]
    if not selected_ids:
        raise ValueError("Cluster metadata missing selected_state_ids.")
    collected = _collect_cluster_inputs(project_id, system_id, selected_ids)
    source_state_ids = np.asarray(collected["merged_frame_state_ids"], dtype=str)
    source_frame_idx = np.asarray(collected["merged_frame_indices"], dtype=np.int64)
    target_state_ids = np.asarray(payload.get("merged__frame_state_ids"), dtype=str)
    target_frame_idx = np.asarray(payload.get("merged__frame_indices"), dtype=np.int64)
    row_map = _build_source_to_target_row_map(
        source_state_ids=source_state_ids,
        source_frame_indices=source_frame_idx,
        target_state_ids=target_state_ids,
        target_frame_indices=target_frame_idx,
    )
    if np.any(row_map < 0):
        missing = int(np.count_nonzero(row_map < 0))
        raise ValueError(f"Could not align {missing} merged frames between source descriptors and cluster NPZ.")

    merged_halo = np.asarray(payload["merged__labels"], dtype=np.int32)
    merged_assigned = np.asarray(payload["merged__labels_assigned"], dtype=np.int32)
    merged_counts = np.asarray(payload.get("merged__cluster_counts"), dtype=np.int32).copy()
    if merged_halo.shape != merged_assigned.shape:
        raise ValueError("Merged halo/assigned arrays have incompatible shapes.")
    n_frames, n_residues = merged_halo.shape
    if n_residues != len(residue_key_list):
        raise ValueError("Residue key count does not match merged label shape.")

    out_halo = merged_halo.copy()
    out_assigned = merged_assigned.copy()
    cluster_params = meta.get("cluster_params") or {}
    if max_cluster_frames is None:
        max_cluster_frames = cluster_params.get("max_cluster_frames")
    linkage_method = str(linkage_method or "ward").lower()
    covariance_type = str(covariance_type or "full").lower()
    cluster_selection_mode = str(cluster_selection_mode or "maxclust").lower()
    if linkage_method not in {"ward", "complete", "average", "single"}:
        raise ValueError("linkage_method must be one of: ward, complete, average, single.")
    if covariance_type not in {"full", "diag"}:
        raise ValueError("covariance_type must be one of: full, diag.")
    if cluster_selection_mode not in {"maxclust", "inconsistent"}:
        raise ValueError("cluster_selection_mode must be one of: maxclust, inconsistent.")
    if cluster_selection_mode == "inconsistent":
        if inconsistent_threshold is None:
            raise ValueError("inconsistent_threshold is required when cluster_selection_mode='inconsistent'.")
        try:
            inconsistent_threshold = float(inconsistent_threshold)
        except Exception as exc:
            raise ValueError("inconsistent_threshold must be numeric.") from exc
        if not np.isfinite(float(inconsistent_threshold)):
            raise ValueError("inconsistent_threshold must be finite.")
    inconsistent_depth = max(1, int(inconsistent_depth))

    overrides_update: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    for ridx in patch_indices:
        source_samples = np.asarray(collected["merged_angles_per_residue"][ridx], dtype=float)
        if source_samples.shape[0] != int(source_state_ids.shape[0]):
            raise ValueError("Merged source sample length mismatch for residue patching.")
        samples = source_samples[row_map]
        if samples.shape[0] != n_frames:
            raise ValueError("Aligned sample length mismatch for residue patching.")
        k_default = int(merged_counts[ridx]) if ridx < merged_counts.shape[0] and int(merged_counts[ridx]) > 0 else 2
        k_target: Optional[int]
        if cluster_selection_mode == "maxclust":
            k_target = int(n_clusters) if n_clusters is not None else k_default
            k_target = max(1, k_target)
        else:
            k_target = None
        labels_halo_r, labels_assigned_r, k_final, diag, _, model_spec = _fit_hierarchical_frozen_gmm(
            samples,
            n_clusters=k_target,
            cluster_selection_mode=cluster_selection_mode,
            inconsistent_threshold=float(inconsistent_threshold) if inconsistent_threshold is not None else None,
            inconsistent_depth=inconsistent_depth,
            linkage_method=linkage_method,
            covariance_type=covariance_type,
            reg_covar=float(reg_covar),
            halo_percentile=float(halo_percentile),
            max_cluster_frames=int(max_cluster_frames) if max_cluster_frames else None,
        )
        if labels_halo_r.shape[0] != n_frames or labels_assigned_r.shape[0] != n_frames:
            raise ValueError("Patched labels have unexpected frame count.")
        out_halo[:, ridx] = labels_halo_r
        out_assigned[:, ridx] = labels_assigned_r
        if ridx < merged_counts.shape[0]:
            merged_counts[ridx] = int(k_final)
        overrides_update[str(int(ridx))] = model_spec
        diagnostics[residue_key_list[ridx]] = diag

    system_meta = store.get_system(project_id, system_id)
    state_labels, _ = _build_state_name_maps(system_meta)

    # Build preview predictions from merged patched labels directly.
    # This avoids re-predicting all residues for all states and keeps patch preview responsive.
    unique_state_ids = sorted({str(v) for v in target_state_ids.tolist()})
    state_frame_counts: Dict[str, int] = {}
    for sid in unique_state_ids:
        count = 0
        state_obj = (system_meta.states or {}).get(sid) if hasattr(system_meta, "states") else None
        desc_rel = getattr(state_obj, "descriptor_file", None)
        if isinstance(desc_rel, str) and desc_rel:
            try:
                desc_path = store.resolve_path(project_id, system_id, desc_rel)
                if desc_path.exists():
                    features = load_descriptor_npz(desc_path)
                    count = int(_infer_frame_count(features))
            except Exception:
                count = 0
        if count <= 0:
            mask = target_state_ids == sid
            if np.any(mask):
                count = int(np.max(target_frame_idx[mask]) + 1)
        state_frame_counts[sid] = max(0, int(count))

    condition_payload, predictions_meta, extra_meta = _build_state_predictions_from_merged(
        merged_labels_halo=out_halo,
        merged_labels_assigned=out_assigned,
        merged_frame_state_ids=[str(v) for v in target_state_ids.tolist()],
        merged_frame_indices=[int(v) for v in target_frame_idx.tolist()],
        state_frame_counts=state_frame_counts,
        state_labels=state_labels,
    )

    patch_id = str(uuid.uuid4())
    prefix = f"patch__{_slug(patch_id)}__"
    pref_payload, pref_predictions, pref_extra, _ = _prefix_condition_payload(
        condition_payload,
        predictions_meta,
        extra_meta,
        prefix=prefix,
    )
    payload.update(pref_payload)
    payload[f"{prefix}merged__labels"] = out_halo
    payload[f"{prefix}merged__labels_assigned"] = out_assigned
    payload[f"{prefix}merged__cluster_counts"] = merged_counts

    merged_keys = ((meta.get("merged") or {}).get("npz_keys") or {})
    patch_entry = {
        "patch_id": patch_id,
        "name": patch_name or f"patch_{patch_id[:8]}",
        "status": "preview",
        "created_at": datetime.utcnow().isoformat(),
        "algorithm": "hierarchical_gmm",
        "params": {
            "n_clusters": int(n_clusters) if n_clusters is not None else None,
            "cluster_selection_mode": cluster_selection_mode,
            "inconsistent_threshold": float(inconsistent_threshold) if inconsistent_threshold is not None else None,
            "inconsistent_depth": int(inconsistent_depth),
            "linkage_method": linkage_method,
            "covariance_type": covariance_type,
            "reg_covar": float(reg_covar),
            "halo_percentile": float(halo_percentile),
            "max_cluster_frames": int(max_cluster_frames) if max_cluster_frames else None,
        },
        "residues": {
            "indices": [int(i) for i in patch_indices],
            "keys": [residue_key_list[int(i)] for i in patch_indices],
        },
        "merged": {
            "npz_keys": {
                "labels": f"{prefix}merged__labels",
                "labels_halo": f"{prefix}merged__labels",
                "labels_assigned": f"{prefix}merged__labels_assigned",
                "cluster_counts": f"{prefix}merged__cluster_counts",
                "frame_state_ids": merged_keys.get("frame_state_ids", "merged__frame_state_ids"),
                "frame_indices": merged_keys.get("frame_indices", "merged__frame_indices"),
            }
        },
        "predictions": pref_predictions,
        "halo_summary": (pref_extra or {}).get("halo_summary") or {},
        "residue_model_overrides": overrides_update,
        "diagnostics": diagnostics,
    }

    patches = [p for p in (meta.get("cluster_patches") or []) if isinstance(p, dict)]
    patches.append(patch_entry)
    meta["cluster_patches"] = patches
    _save_cluster_payload(cluster_path, payload, meta)

    return {
        "cluster_id": cluster_id,
        "patch_id": patch_id,
        "patch": patch_entry,
    }


def discard_cluster_residue_patch(
    project_id: str,
    system_id: str,
    cluster_id: str,
    *,
    patch_id: str,
    store: ProjectStore | None = None,
) -> Dict[str, Any]:
    store = store or ProjectStore()
    cluster_path = store.ensure_cluster_directories(project_id, system_id, cluster_id)["cluster_dir"] / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found for cluster_id='{cluster_id}'.")
    payload, meta = _load_cluster_payload(cluster_path)
    patches = [p for p in (meta.get("cluster_patches") or []) if isinstance(p, dict)]
    keep: List[Dict[str, Any]] = []
    removed = None
    prefix = f"patch__{_slug(str(patch_id))}__"
    for p in patches:
        if str(p.get("patch_id")) == str(patch_id):
            removed = p
            continue
        keep.append(p)
    if removed is None:
        raise ValueError(f"Patch not found: {patch_id}")
    for key in [k for k in payload.keys() if str(k).startswith(prefix)]:
        payload.pop(key, None)
    meta["cluster_patches"] = keep
    _save_cluster_payload(cluster_path, payload, meta)
    return {"cluster_id": cluster_id, "patch_id": str(patch_id), "status": "discarded"}


def confirm_cluster_residue_patch(
    project_id: str,
    system_id: str,
    cluster_id: str,
    *,
    patch_id: str,
    recompute_assignments: bool = True,
    store: ProjectStore | None = None,
) -> Dict[str, Any]:
    store = store or ProjectStore()
    cluster_path = store.ensure_cluster_directories(project_id, system_id, cluster_id)["cluster_dir"] / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found for cluster_id='{cluster_id}'.")
    payload, meta = _load_cluster_payload(cluster_path)

    patches = [p for p in (meta.get("cluster_patches") or []) if isinstance(p, dict)]
    patch = next((p for p in patches if str(p.get("patch_id")) == str(patch_id)), None)
    if patch is None:
        raise ValueError(f"Patch not found: {patch_id}")
    merged_keys = ((patch.get("merged") or {}).get("npz_keys") or {})
    key_halo = merged_keys.get("labels_halo") or merged_keys.get("labels")
    key_assigned = merged_keys.get("labels_assigned")
    key_counts = merged_keys.get("cluster_counts")
    if not key_halo or key_halo not in payload:
        raise ValueError("Patch is missing merged halo labels.")
    if not key_assigned or key_assigned not in payload:
        raise ValueError("Patch is missing merged assigned labels.")
    if not key_counts or key_counts not in payload:
        raise ValueError("Patch is missing merged cluster counts.")

    payload["merged__labels"] = np.asarray(payload[key_halo], dtype=np.int32)
    payload["merged__labels_assigned"] = np.asarray(payload[key_assigned], dtype=np.int32)
    payload["merged__cluster_counts"] = np.asarray(payload[key_counts], dtype=np.int32)

    predictions_patch = patch.get("predictions") or {}
    prefix = f"patch__{_slug(str(patch_id))}__"
    predictions_new: Dict[str, Any] = {}
    keys_to_copy: Dict[str, str] = {}
    for pred_key, entry in predictions_patch.items():
        if not isinstance(entry, dict):
            continue
        new_entry: Dict[str, Any] = {}
        for k, v in entry.items():
            if isinstance(v, str) and v.startswith(prefix):
                target_key = v[len(prefix) :]
                keys_to_copy[v] = target_key
                new_entry[k] = target_key
            else:
                new_entry[k] = v
        predictions_new[str(pred_key)] = new_entry

    halo_summary_patch = patch.get("halo_summary") or {}
    halo_summary_new = {}
    if isinstance(halo_summary_patch, dict):
        halo_summary_new = dict(halo_summary_patch)
        npz_keys = halo_summary_new.get("npz_keys")
        if isinstance(npz_keys, dict):
            remapped = {}
            for k, v in npz_keys.items():
                if isinstance(v, str) and v.startswith(prefix):
                    target_key = v[len(prefix) :]
                    keys_to_copy[v] = target_key
                    remapped[str(k)] = target_key
                else:
                    remapped[str(k)] = v
            halo_summary_new["npz_keys"] = remapped

    for src_key, dst_key in keys_to_copy.items():
        if src_key in payload:
            payload[dst_key] = payload[src_key]

    meta["predictions"] = predictions_new
    if halo_summary_new:
        meta["halo_summary"] = halo_summary_new

    overrides = meta.get("residue_model_overrides") or {}
    if not isinstance(overrides, dict):
        overrides = {}
    patch_overrides = patch.get("residue_model_overrides") or {}
    if isinstance(patch_overrides, dict):
        for k, v in patch_overrides.items():
            overrides[str(k)] = v
    meta["residue_model_overrides"] = overrides

    now = datetime.utcnow().isoformat()
    history = [h for h in (meta.get("patch_history") or []) if isinstance(h, dict)]
    history.append(
        {
            "patch_id": str(patch_id),
            "name": patch.get("name"),
            "confirmed_at": now,
            "algorithm": patch.get("algorithm"),
            "residue_keys": (patch.get("residues") or {}).get("keys") or [],
        }
    )
    meta["patch_history"] = history
    meta["cluster_patches"] = [p for p in patches if str(p.get("patch_id")) != str(patch_id)]

    for key in [k for k in payload.keys() if str(k).startswith(prefix)]:
        payload.pop(key, None)

    _save_cluster_payload(cluster_path, payload, meta)

    assignments: Dict[str, Any] = {}
    if recompute_assignments:
        assignments = assign_cluster_labels_to_states(cluster_path, project_id, system_id)
        update_cluster_metadata_with_assignments(cluster_path, assignments)

    return {
        "cluster_id": cluster_id,
        "patch_id": str(patch_id),
        "status": "confirmed",
        "assignments": assignments,
    }


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
    residue_models: List[Any],
    density_maxk: int,
    state_labels: Dict[str, str],
    metastable_labels: Dict[str, str],
    analysis_mode: Optional[str],
    predict_jobs: int | None = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
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

    states_to_process: List[tuple[str, DescriptorState, Path]] = []
    for state_id, state in (system_meta.states or {}).items():
        if not state.descriptor_file:
            continue
        desc_path = store.resolve_path(project_id, system_id, state.descriptor_file)
        if not desc_path.exists():
            continue
        states_to_process.append((str(state_id), state, desc_path))

    total_pred_jobs = len(states_to_process) * n_residues
    completed_pred_jobs = 0
    if progress_callback and total_pred_jobs:
        progress_callback("Predicting labels...", 0, total_pred_jobs)

    use_processes = predict_jobs is not None and int(predict_jobs) > 1
    if use_processes:
        # Legacy dual-ADP models embed live Data objects; keep prediction in-process for stability.
        for mdl in residue_models:
            if isinstance(mdl, dict) and str(mdl.get("kind") or "").lower() == "adp_legacy_models":
                use_processes = False
                break

    for state_id, state, desc_path in states_to_process:
        features = load_descriptor_npz(desc_path)
        frame_count = _infer_frame_count(features)
        if frame_count <= 0:
            continue
        labels_halo = np.full((frame_count, n_residues), -1, dtype=np.int32)
        labels_assigned = np.full((frame_count, n_residues), -1, dtype=np.int32)
        if use_processes:
            with ProcessPoolExecutor(max_workers=int(predict_jobs)) as executor:
                futures: Dict[Any, int] = {}
                for res_idx, key in enumerate(residue_keys):
                    angles = _extract_angles_array(features, key)
                    if angles is None or angles.shape[0] != frame_count or residue_models[res_idx] is None:
                        completed_pred_jobs += 1
                        if progress_callback and total_pred_jobs:
                            progress_callback(
                                f"Predicting labels: {completed_pred_jobs}/{total_pred_jobs}",
                                completed_pred_jobs,
                                total_pred_jobs,
                            )
                        continue
                    futures[executor.submit(
                        _predict_residue_worker,
                        res_idx,
                        angles,
                        residue_models[res_idx],
                        density_maxk,
                    )] = res_idx
                for fut in as_completed(futures):
                    res_idx, labels_assigned_res, labels_halo_res = fut.result()
                    labels_assigned[:, res_idx] = labels_assigned_res
                    labels_halo[:, res_idx] = labels_halo_res
                    completed_pred_jobs += 1
                    if progress_callback and total_pred_jobs:
                        progress_callback(
                            f"Predicting labels: {completed_pred_jobs}/{total_pred_jobs}",
                            completed_pred_jobs,
                            total_pred_jobs,
                        )
        else:
            for res_idx, key in enumerate(residue_keys):
                angles = _extract_angles_array(features, key)
                if angles is None or angles.shape[0] != frame_count or residue_models[res_idx] is None:
                    if progress_callback and total_pred_jobs:
                        completed_pred_jobs += 1
                        progress_callback(
                            f"Predicting labels: {completed_pred_jobs}/{total_pred_jobs}",
                            completed_pred_jobs,
                            total_pred_jobs,
                        )
                    continue
                labels_assigned_res, labels_halo_res = _predict_labels_with_model(
                    residue_models[res_idx],
                    angles,
                    density_maxk=density_maxk,
                )
                labels_assigned[:, res_idx] = labels_assigned_res
                labels_halo[:, res_idx] = labels_halo_res
                if progress_callback and total_pred_jobs:
                    completed_pred_jobs += 1
                    progress_callback(
                        f"Predicting labels: {completed_pred_jobs}/{total_pred_jobs}",
                        completed_pred_jobs,
                        total_pred_jobs,
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
    n_jobs: int | None = None,
    persist_models: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
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
    if density_z is None:
        density_z_val: float | str = 2.0
    elif isinstance(density_z, str) and density_z.lower() == "auto":
        density_z_val = "auto"
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

    dp_models: List[Data | None] = [None] * len(residue_keys)

    use_processes = n_jobs is not None and int(n_jobs) > 1
    if use_processes:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
            futures = []
            for col, samples in enumerate(merged_angles_per_residue):
                sample_arr = np.asarray(samples, dtype=float)
                if sample_arr.shape[0] != merged_frame_count:
                    raise ValueError("Merged residue samples have inconsistent frame counts.")
                futures.append(
                    executor.submit(
                        _cluster_residue_worker_with_models,
                        col,
                        sample_arr,
                        density_maxk_val,
                        density_z_val,
                        max_cluster_frames_val,
                        merged_subsample_indices,
                    )
                )
            for fut in as_completed(futures):
                col, labels_halo, labels_assigned, k, dp_data = fut.result()
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
                dp_models[col] = dp_data
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
            if sample_arr.shape[0] != merged_frame_count:
                raise ValueError("Merged residue samples have inconsistent frame counts.")
            labels_halo, labels_assigned, k, _, _, dp_data = _cluster_with_subsample(
                sample_arr,
                density_maxk=density_maxk_val,
                density_z=density_z_val,
                max_cluster_frames=max_cluster_frames_val,
                subsample_indices=merged_subsample_indices,
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
            dp_models[col] = dp_data
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
        residue_models=dp_models,
        density_maxk=density_maxk_val,
        state_labels=state_labels,
        metastable_labels=metastable_labels,
        analysis_mode=getattr(system_meta, "analysis_mode", None),
        predict_jobs=n_jobs,
        progress_callback=progress_callback,
    )

    # Persist NPZ
    dirs = store.ensure_directories(project_id, system_id)
    cluster_dir = dirs["clusters_dir"]
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
        "merged__labels": merged_labels_halo,
        "merged__labels_assigned": merged_labels_assigned,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": np.array(merged_frame_state_ids),
        "merged__frame_metastable_ids": np.array(merged_frame_meta_ids),
        "merged__frame_indices": np.array(merged_frame_indices, dtype=np.int64),
    }
    payload.update(condition_payload)
    if persist_models:
        model_dir = out_path.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_paths: List[str | None] = []
        for idx, dp_model in enumerate(dp_models):
            if dp_model is None:
                model_paths.append(None)
                continue
            model_path = model_dir / f"res_{idx:04d}.pkl"
            with open(model_path, "wb") as outp:
                pickle.dump(dp_model, outp, pickle.HIGHEST_PROTOCOL)
            try:
                system_root = store.ensure_directories(project_id, system_id)["system_dir"]
                rel_path = str(model_path.relative_to(system_root))
            except Exception:
                rel_path = str(model_path)
            model_paths.append(rel_path)
        metadata["model_paths"] = model_paths
    payload["metadata_json"] = np.array(json.dumps(metadata))

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
    if density_z is None:
        density_z_val: float | str = 2.0
    elif isinstance(density_z, str) and density_z.lower() == "auto":
        density_z_val = "auto"
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
    cluster_id: Optional[str] = None,
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
        "cluster_id": cluster_id,
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
        density_z=params.get("density_z", 2.0),
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
    density_z_val = cluster_params.get("density_z", 2.0)
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

    dp_models: List[Data | None] = []

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
    system_meta = store.get_system(project_id, system_id)
    state_labels, metastable_labels = _build_state_name_maps(system_meta)
    metastable_kinds = _build_metastable_kind_map(system_meta)
    cluster_id = manifest.get("cluster_id")
    if cluster_id:
        out_path = store.ensure_cluster_directories(project_id, system_id, cluster_id)["cluster_dir"] / "cluster.npz"
    else:
        dirs = store.ensure_directories(project_id, system_id)
        cluster_dir = dirs["clusters_dir"]
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        suffix = "-".join(_slug(mid)[:24] for mid in selected_meta_ids) or "cluster"
        out_path = cluster_dir / f"{suffix}_clusters_{timestamp}.npz"

    frame_state_ids = np.load(work_dir / manifest["frame_state_ids_path"], allow_pickle=True)
    frame_meta_ids = np.load(work_dir / manifest["frame_meta_ids_path"], allow_pickle=True)
    frame_indices = np.load(work_dir / manifest["frame_indices_path"], allow_pickle=True)
    contact_sources = manifest.get("contact_sources") or []

    condition_payload, predictions_meta, extra_meta = _build_condition_predictions(
        project_id=project_id,
        system_id=system_id,
        residue_keys=residue_keys,
        residue_models=dp_models,
        density_maxk=density_maxk_val,
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
        "metastable_labels": metastable_labels,
        "metastable_kinds": metastable_kinds,
        "cluster_algorithm": "density_peaks",
        "cluster_params": cluster_params,
        "predictions": predictions_meta,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "random_state": cluster_params.get("random_state"),
        "contact_sources": contact_sources,
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
        "merged__labels": merged_labels_halo,
        "merged__labels_assigned": merged_labels_assigned,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": frame_state_ids,
        "merged__frame_metastable_ids": frame_meta_ids,
        "merged__frame_indices": frame_indices,
    }
    payload.update(condition_payload)
    model_dir = out_path.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_paths: List[str | None] = []
    for idx, dp_model in enumerate(dp_models):
        if dp_model is None:
            model_paths.append(None)
            continue
        model_path = model_dir / f"res_{idx:04d}.pkl"
        with open(model_path, "wb") as outp:
            pickle.dump(dp_model, outp, pickle.HIGHEST_PROTOCOL)
        try:
            system_root = store.ensure_directories(project_id, system_id)["system_dir"]
            rel_path = str(model_path.relative_to(system_root))
        except Exception:
            rel_path = str(model_path)
        model_paths.append(rel_path)
    metadata["model_paths"] = model_paths
    payload["metadata_json"] = np.array(json.dumps(metadata))

    np.savez_compressed(out_path, **payload)
    return out_path, metadata
