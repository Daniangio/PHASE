from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phase.io.data import load_npz
from phase.potts.metrics import (
    combined_distance,
    js_divergence,
    marginals,
    pairwise_joints_on_edges,
    per_residue_js,
)
from phase.potts.potts_model import PottsModel, load_potts_model, zero_sum_gauge_model
from phase.potts.sample_io import load_sample_npz
from phase.services.project_store import ProjectStore


ANALYSIS_METADATA_FILENAME = "analysis_metadata.json"


@dataclass(frozen=True)
class AnalysisPaths:
    analysis_id: str
    analysis_dir: Path
    npz_path: Path
    metadata_path: Path


def _utc_now() -> str:
    return datetime.utcnow().isoformat()


def _relativize(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _ensure_analysis_dir(cluster_dir: Path, kind: str) -> Path:
    root = cluster_dir / "analyses" / kind
    root.mkdir(parents=True, exist_ok=True)
    return root


def _compute_edge_js(
    X_a: np.ndarray,
    X_b: np.ndarray,
    *,
    K: Sequence[int],
    edges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    if not edges:
        return np.zeros((0,), dtype=float)
    P_a = pairwise_joints_on_edges(X_a, K, edges)
    P_b = pairwise_joints_on_edges(X_b, K, edges)
    out = np.zeros(len(edges), dtype=float)
    for idx, e in enumerate(edges):
        out[idx] = js_divergence(P_a[e].ravel(), P_b[e].ravel())
    return out


def _pairwise_joints_flat_on_edges(
    labels: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    *,
    max_k: int,
    chunk_edges: int = 512,
) -> np.ndarray:
    """
    Fast joint distribution estimator on many edges.

    Returns an array P with shape (E, max_k*max_k) where each row sums to 1 and corresponds
    to the flattened joint distribution over encoded pairs: code = a*max_k + b.

    This is used by the lambda-sweep analysis (validation_ladder4.MD) where we need edge-JS
    for many (lambda, reference) comparisons efficiently.
    """
    X = np.asarray(labels, dtype=int)
    if X.ndim != 2:
        raise ValueError("labels must be 2D (T,N).")
    edges_arr = np.asarray(list(edges), dtype=int)
    if edges_arr.size == 0:
        return np.zeros((0, int(max_k) * int(max_k)), dtype=float)

    T = int(X.shape[0])
    E = int(edges_arr.shape[0])
    max_k = int(max_k)
    if max_k < 1:
        raise ValueError("max_k must be >= 1.")
    V = int(max_k * max_k)
    out = np.zeros((E, V), dtype=float)
    if T <= 0:
        return out

    chunk_edges = max(1, int(chunk_edges))
    for start in range(0, E, chunk_edges):
        chunk = edges_arr[start : start + chunk_edges]
        r_idx = chunk[:, 0]
        s_idx = chunk[:, 1]
        # (T, Echunk)
        codes = X[:, r_idx] * max_k + X[:, s_idx]
        echunk = int(chunk.shape[0])
        offsets = (np.arange(echunk, dtype=np.int64) * V)[None, :]
        flat = (codes.astype(np.int64, copy=False) + offsets).ravel()
        counts = np.bincount(flat, minlength=echunk * V).astype(float, copy=False).reshape(echunk, V)
        out[start : start + echunk] = counts / float(T)
    return out


def _js_divergence_rows(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized JS divergence over rows.

    p, q: arrays of shape (M, V) where rows represent distributions.
    Returns: (M,) JS divergence for each row.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("Expected matching 2D arrays for p and q.")

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    q = q / np.sum(q, axis=1, keepdims=True)
    m = 0.5 * (p + q)
    # KL(p||m) + KL(q||m), with the same epsilon smoothing.
    kl_p = np.sum(p * np.log(p / m), axis=1)
    kl_q = np.sum(q * np.log(q / m), axis=1)
    return 0.5 * (kl_p + kl_q)


def compute_md_vs_sample_metrics(
    X_md: np.ndarray,
    X_sample: np.ndarray,
    *,
    K: Sequence[int],
    edges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    def _stat(arr: np.ndarray, fn) -> float | None:
        if arr is None or not getattr(arr, "size", 0):
            return None
        val = float(fn(arr))
        return val if np.isfinite(val) else None

    p_md = marginals(X_md, K)
    p_s = marginals(X_sample, K)
    node_js = per_residue_js(p_md, p_s)
    edge_js = _compute_edge_js(X_md, X_sample, K=K, edges=edges)
    combined = float(combined_distance(X_md, X_sample, K=K, edges=edges, w_marg=1.0, w_pair=1.0))
    payload: Dict[str, Any] = {
        "node_js": node_js,
        "edge_js": edge_js,
        "node_js_mean": _stat(node_js, np.mean),
        "node_js_median": _stat(node_js, np.median),
        "node_js_max": _stat(node_js, np.max),
        "edge_js_mean": _stat(edge_js, np.mean),
        "edge_js_median": _stat(edge_js, np.median),
        "edge_js_max": _stat(edge_js, np.max),
        "combined_distance": combined if np.isfinite(combined) else None,
    }
    return payload


def compute_sample_energies(model: PottsModel, X: np.ndarray) -> Dict[str, Any]:
    energies = model.energy_batch(X)
    if energies is None or energies.size == 0:
        return {
            "energies": np.asarray([], dtype=float),
            "energy_mean": None,
            "energy_median": None,
            "energy_min": None,
            "energy_max": None,
        }
    payload: Dict[str, Any] = {
        "energies": energies,
        "energy_mean": float(np.mean(energies)) if np.isfinite(np.mean(energies)) else None,
        "energy_median": float(np.median(energies)) if np.isfinite(np.median(energies)) else None,
        "energy_min": float(np.min(energies)) if np.isfinite(np.min(energies)) else None,
        "energy_max": float(np.max(energies)) if np.isfinite(np.max(energies)) else None,
    }
    return payload


def analyze_cluster_samples(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_ref: str | None = None,
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
) -> Dict[str, Any]:
    """
    Compute:
      - MD-vs-sample distribution metrics (node JS + edge JS) for all MD samples vs all potts_sampling samples
      - optional: per-sample energies under a selected model (if model_ref is provided)

    model_ref:
      - a model_id found in potts_models metadata, OR
      - a path to a model NPZ
    """
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]
    cluster_path = cluster_dir / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found: {cluster_path}")

    ds = load_npz(str(cluster_path), unassigned_policy="drop_frames", allow_missing_edges=True)
    K = ds.cluster_counts.tolist()
    cluster_edges = ds.edges

    samples = store.list_samples(project_id, system_id, cluster_id)
    md_samples = [s for s in samples if s.get("type") == "md_eval"]
    other_samples = [s for s in samples if s.get("type") != "md_eval"]

    # Resolve model if requested
    model: Optional[PottsModel] = None
    model_id = None
    model_name = None
    if model_ref:
        model_path = Path(model_ref)
        if not model_path.suffix:
            # maybe a model_id
            model_id = str(model_ref)
            models = store.list_potts_models(project_id, system_id, cluster_id)
            entry = next((m for m in models if m.get("model_id") == model_id), None)
            if not entry or not entry.get("path"):
                raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
            model_name = entry.get("name") or model_id
            model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
        else:
            if not model_path.is_absolute():
                model_path = store.resolve_path(project_id, system_id, str(model_path))
            model_name = model_path.stem
        if not model_path.exists():
            raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
        model = load_potts_model(str(model_path))

    # Edge-based metrics are model-dependent: use the Potts model's edges when available.
    # Cluster NPZ edges are often empty (clusters don't persist contacts), but Potts models do.
    edges_for_metrics: list[tuple[int, int]] = []
    if model is not None:
        edges_for_metrics = list(model.edges or [])
    else:
        edges_for_metrics = list(cluster_edges or [])
    # Normalize edge ordering (r < s) and sort for stable analysis artifacts.
    edges_for_metrics = sorted({(min(int(r), int(s)), max(int(r), int(s))) for r, s in edges_for_metrics if int(r) != int(s)})

    comparisons_root = _ensure_analysis_dir(cluster_dir, "md_vs_sample")
    energies_root = _ensure_analysis_dir(cluster_dir, "model_energy")

    written_comparisons: List[Dict[str, Any]] = []
    written_energies: List[Dict[str, Any]] = []

    def _load_labels(sample_entry: Dict[str, Any], *, md_mode: bool) -> np.ndarray:
        paths = sample_entry.get("paths") or {}
        rel = None
        if isinstance(paths, dict):
            rel = paths.get("summary_npz") or paths.get("path")
        rel = rel or sample_entry.get("path")
        if not rel:
            raise FileNotFoundError("Sample entry missing path.")
        npz_path = Path(str(rel))
        if not npz_path.is_absolute():
            resolved = store.resolve_path(project_id, system_id, str(rel))
            # Some legacy metadata stored paths relative to the cluster dir (e.g. "samples/<id>/...").
            # Prefer a working path if possible.
            if not resolved.exists():
                alt = cluster_dir / str(rel)
                npz_path = alt if alt.exists() else resolved
            else:
                npz_path = resolved
        try:
            sample_npz = load_sample_npz(npz_path)
        except Exception as exc:
            # Skip legacy/broken samples rather than failing the entire cluster analysis.
            print(f"[potts.analysis] warning: failed to load sample {sample_entry.get('sample_id')} ({npz_path}): {exc}")
            return np.zeros((0, 0), dtype=int)
        if md_mode and (md_label_mode or "assigned").lower() in {"halo", "labels_halo"} and sample_npz.labels_halo is not None:
            X = sample_npz.labels_halo
        else:
            X = sample_npz.labels
        if drop_invalid and sample_npz.invalid_mask is not None:
            keep = ~np.asarray(sample_npz.invalid_mask, dtype=bool)
            if keep.shape[0] == X.shape[0]:
                X = X[keep]
        return np.asarray(X, dtype=int)

    # Pairwise MD vs non-MD comparisons
    for md in md_samples:
        X_md = _load_labels(md, md_mode=True)
        if X_md.size == 0:
            continue
        for other in other_samples:
            X_s = _load_labels(other, md_mode=False)
            if X_s.size == 0:
                continue
            metrics = compute_md_vs_sample_metrics(X_md, X_s, K=K, edges=edges_for_metrics)
            analysis_id = str(uuid.uuid4())
            analysis_dir = comparisons_root / analysis_id
            analysis_dir.mkdir(parents=True, exist_ok=True)
            npz_path = analysis_dir / "analysis.npz"
            np.savez_compressed(
                npz_path,
                node_js=np.asarray(metrics["node_js"], dtype=float),
                edge_js=np.asarray(metrics["edge_js"], dtype=float),
                edges=np.asarray(edges_for_metrics, dtype=int),
            )
            meta = {
                "analysis_id": analysis_id,
                "analysis_type": "md_vs_sample",
                "created_at": _utc_now(),
                "project_id": project_id,
                "system_id": system_id,
                "cluster_id": cluster_id,
                "md_sample_id": md.get("sample_id"),
                "md_sample_name": md.get("name"),
                "sample_id": other.get("sample_id"),
                "sample_name": other.get("name"),
                "sample_type": other.get("type"),
                "sample_method": other.get("method"),
                "model_id": model_id,
                "model_name": model_name,
                "drop_invalid": bool(drop_invalid),
                "md_label_mode": md_label_mode,
                "paths": {
                    "analysis_npz": _relativize(npz_path, system_dir),
                },
                "summary": {
                    "node_js_mean": metrics["node_js_mean"],
                    "node_js_median": metrics["node_js_median"],
                    "node_js_max": metrics["node_js_max"],
                    "edge_js_mean": metrics["edge_js_mean"],
                    "edge_js_median": metrics["edge_js_median"],
                    "edge_js_max": metrics["edge_js_max"],
                    "combined_distance": metrics["combined_distance"],
                    "md_count": int(X_md.shape[0]),
                    "sample_count": int(X_s.shape[0]),
                },
            }
            meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            written_comparisons.append(meta)

    # Energies for all samples under selected model (if any)
    if model is not None:
        for sample in samples:
            X = _load_labels(sample, md_mode=(sample.get("type") == "md_eval"))
            if X.size == 0:
                continue
            payload = compute_sample_energies(model, X)
            analysis_id = str(uuid.uuid4())
            analysis_dir = energies_root / analysis_id
            analysis_dir.mkdir(parents=True, exist_ok=True)
            npz_path = analysis_dir / "analysis.npz"
            np.savez_compressed(npz_path, energies=np.asarray(payload["energies"], dtype=float))
            meta = {
                "analysis_id": analysis_id,
                "analysis_type": "model_energy",
                "created_at": _utc_now(),
                "project_id": project_id,
                "system_id": system_id,
                "cluster_id": cluster_id,
                "model_id": model_id,
                "model_name": model_name,
                "sample_id": sample.get("sample_id"),
                "sample_name": sample.get("name"),
                "sample_type": sample.get("type"),
                "sample_method": sample.get("method"),
                "drop_invalid": bool(drop_invalid),
                "md_label_mode": md_label_mode,
                "paths": {
                    "analysis_npz": _relativize(npz_path, system_dir),
                },
                "summary": {
                    "energy_mean": payload["energy_mean"],
                    "energy_median": payload["energy_median"],
                    "energy_min": payload["energy_min"],
                    "energy_max": payload["energy_max"],
                    "count": int(X.shape[0]),
                },
            }
            meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            written_energies.append(meta)

    return {
        "cluster_id": cluster_id,
        "md_samples": len(md_samples),
        "other_samples": len(other_samples),
        "comparisons_written": len(written_comparisons),
        "energies_written": len(written_energies),
        "model_id": model_id,
        "model_name": model_name,
    }


def compute_lambda_sweep_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str,
    model_b_ref: str,
    lambda_sample_ids: Sequence[str],
    lambdas: Sequence[float],
    ref_md_sample_ids: Sequence[str],
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    alpha: float = 0.5,
    edge_chunk: int = 512,
) -> dict[str, Any]:
    """
    Lambda-interpolation experiment analysis (validation_ladder4.MD).

    Given endpoint models A/B (λ=1 and λ=0) and a series of sampled ensembles from E_λ,
    compute:
      - ΔE(s) = E_A(s) - E_B(s): mean + IQR vs λ (order parameter)
      - Node/edge JS divergence vs 3 reference MD ensembles, as curves vs λ
      - Combined match curve D(λ) to the 3rd reference: α*JS_node_mean + (1-α)*JS_edge_mean

    Returns a dict of arrays ready to be persisted into analysis.npz plus metadata helpers.
    """
    if len(lambda_sample_ids) != len(lambdas):
        raise ValueError("lambda_sample_ids and lambdas must have the same length.")
    if len(ref_md_sample_ids) != 3:
        raise ValueError("ref_md_sample_ids must contain exactly 3 sample ids.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0,1].")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]
    cluster_path = cluster_dir / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found: {cluster_path}")

    ds = load_npz(str(cluster_path), unassigned_policy="drop_frames", allow_missing_edges=True)
    K = ds.cluster_counts.tolist()
    max_k = int(max(K)) if len(K) else 0

    samples = store.list_samples(project_id, system_id, cluster_id)

    def _resolve_sample_path(entry: dict[str, Any]) -> Path:
        paths = entry.get("paths") or {}
        rel = None
        if isinstance(paths, dict):
            rel = paths.get("summary_npz") or paths.get("path")
        rel = rel or entry.get("path")
        if not rel:
            raise FileNotFoundError("Sample entry missing path.")
        p = Path(str(rel))
        if not p.is_absolute():
            resolved = store.resolve_path(project_id, system_id, str(rel))
            if not resolved.exists():
                alt = cluster_dir / str(rel)
                p = alt if alt.exists() else resolved
            else:
                p = resolved
        return p

    def _load_labels(entry: dict[str, Any], *, md_mode: bool) -> np.ndarray:
        p = _resolve_sample_path(entry)
        s = load_sample_npz(p)
        X = s.labels
        if md_mode and (md_label_mode or "assigned").lower() in {"halo", "labels_halo"} and s.labels_halo is not None:
            X = s.labels_halo
        if drop_invalid and s.invalid_mask is not None:
            keep = ~np.asarray(s.invalid_mask, dtype=bool)
            if keep.shape[0] == X.shape[0]:
                X = X[keep]
        X = np.asarray(X, dtype=int)
        # Defensive: drop frames with unassigned labels (-1) to keep distributions well-defined.
        if X.size and np.any(X < 0):
            keep = np.all(X >= 0, axis=1)
            X = X[keep]
        return X

    def _resolve_model(ref: str) -> tuple[PottsModel, str | None, str | None, str]:
        model_id = None
        model_name = None
        model_path = Path(ref)
        if not model_path.suffix:
            model_id = str(ref)
            models = store.list_potts_models(project_id, system_id, cluster_id)
            entry = next((m for m in models if m.get("model_id") == model_id), None)
            if not entry or not entry.get("path"):
                raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
            model_name = entry.get("name") or model_id
            model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
        else:
            if not model_path.is_absolute():
                model_path = store.resolve_path(project_id, system_id, str(model_path))
            model_name = model_path.stem
        if not model_path.exists():
            raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
        return load_potts_model(str(model_path)), model_id, model_name, _relativize(model_path, system_dir)

    model_a, model_a_id, model_a_name, model_a_path = _resolve_model(model_a_ref)
    model_b, model_b_id, model_b_name, model_b_path = _resolve_model(model_b_ref)

    if len(model_a.h) != len(model_b.h):
        raise ValueError("Endpoint model sizes do not match.")

    # Always gauge-fix before parameter-based comparisons (validation_ladder4.MD pre-step).
    model_a = zero_sum_gauge_model(model_a)
    model_b = zero_sum_gauge_model(model_b)

    edges_a = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_a.edges or []) if int(r) != int(s)}
    edges_b = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or []) if int(r) != int(s)}
    edges = sorted(edges_a & edges_b)

    # Precompute reference ensemble distributions (marginals + edge joints).
    ref_entries: list[dict[str, Any]] = []
    ref_labels: list[np.ndarray] = []
    ref_marginals: list[list[np.ndarray]] = []
    ref_p2_flat: list[np.ndarray] = []
    ref_names: list[str] = []

    for sid in ref_md_sample_ids:
        entry = next((s for s in samples if s.get("sample_id") == sid), None)
        if not entry:
            raise FileNotFoundError(f"Reference MD sample not found: {sid}")
        ref_entries.append(entry)
        ref_names.append(str(entry.get("name") or sid))
        X_ref = _load_labels(entry, md_mode=True)
        if X_ref.ndim != 2 or X_ref.size == 0:
            raise ValueError(f"Reference MD sample is empty: {sid}")
        ref_labels.append(X_ref)
        ref_marginals.append(marginals(X_ref, K))
        if edges and max_k > 0:
            ref_p2_flat.append(_pairwise_joints_flat_on_edges(X_ref, edges, max_k=max_k, chunk_edges=edge_chunk))
        else:
            ref_p2_flat.append(np.zeros((0, max(1, max_k) * max(1, max_k)), dtype=float))

    # Sort by lambda ascending (keep sample_id association)
    order = np.argsort(np.asarray(lambdas, dtype=float))
    lambdas_sorted = [float(lambdas[i]) for i in order.tolist()]
    sample_ids_sorted = [str(lambda_sample_ids[i]) for i in order.tolist()]

    n_lambda = len(sample_ids_sorted)
    n_ref = 3

    node_js_mean = np.full((n_ref, n_lambda), np.nan, dtype=float)
    edge_js_mean = np.full((n_ref, n_lambda), np.nan, dtype=float)
    combined = np.full((n_ref, n_lambda), np.nan, dtype=float)

    deltaE_mean = np.full((n_lambda,), np.nan, dtype=float)
    deltaE_q25 = np.full((n_lambda,), np.nan, dtype=float)
    deltaE_q75 = np.full((n_lambda,), np.nan, dtype=float)

    sample_names: list[str] = [""] * n_lambda

    def _stat(v: np.ndarray, fn) -> float:
        if v.size == 0:
            return float("nan")
        return float(fn(v))

    for j, sid in enumerate(sample_ids_sorted):
        entry = next((s for s in samples if s.get("sample_id") == sid), None)
        if not entry:
            raise FileNotFoundError(f"Lambda sample not found: {sid}")
        sample_names[j] = str(entry.get("name") or sid)
        X_s = _load_labels(entry, md_mode=False)
        if X_s.ndim != 2 or X_s.size == 0:
            continue
        if X_s.shape[1] != len(K):
            raise ValueError(f"Lambda sample labels do not match cluster size: {sid}")

        p_s = marginals(X_s, K)
        if edges and max_k > 0:
            p2_s = _pairwise_joints_flat_on_edges(X_s, edges, max_k=max_k, chunk_edges=edge_chunk)
        else:
            p2_s = np.zeros((0, max(1, max_k) * max(1, max_k)), dtype=float)

        # ΔE order parameter under endpoint models.
        dE = model_a.energy_batch(X_s) - model_b.energy_batch(X_s)
        deltaE_mean[j] = _stat(dE, np.mean)
        deltaE_q25[j] = _stat(dE, lambda arr: np.quantile(arr, 0.25))
        deltaE_q75[j] = _stat(dE, lambda arr: np.quantile(arr, 0.75))

        for i in range(n_ref):
            js_nodes = per_residue_js(ref_marginals[i], p_s)
            node_js_mean[i, j] = float(np.mean(js_nodes)) if js_nodes.size else np.nan
            if edges:
                js_edges = _js_divergence_rows(ref_p2_flat[i], p2_s)
                edge_js_mean[i, j] = float(np.mean(js_edges)) if js_edges.size else 0.0
            else:
                edge_js_mean[i, j] = 0.0
            combined[i, j] = alpha * node_js_mean[i, j] + (1.0 - alpha) * edge_js_mean[i, j]

    match_ref_index = 2  # the "third" reference is the comparison ensemble
    match_curve = combined[match_ref_index]
    finite_mask = np.isfinite(match_curve)
    if finite_mask.any():
        best_idx = int(np.nanargmin(match_curve))
        best_lambda = float(lambdas_sorted[best_idx])
        best_value = float(match_curve[best_idx])
    else:
        best_idx = -1
        best_lambda = float("nan")
        best_value = float("nan")

    return {
        "model_a_id": model_a_id,
        "model_a_name": model_a_name,
        "model_a_path": model_a_path,
        "model_b_id": model_b_id,
        "model_b_name": model_b_name,
        "model_b_path": model_b_path,
        "ref_md_sample_ids": list(ref_md_sample_ids),
        "ref_md_sample_names": ref_names,
        "md_label_mode": md_label_mode,
        "drop_invalid": bool(drop_invalid),
        "alpha": float(alpha),
        "edges": np.asarray(edges, dtype=int),
        "lambdas": np.asarray(lambdas_sorted, dtype=float),
        "sample_ids": sample_ids_sorted,
        "sample_names": sample_names,
        "node_js_mean": np.asarray(node_js_mean, dtype=float),
        "edge_js_mean": np.asarray(edge_js_mean, dtype=float),
        "combined_distance": np.asarray(combined, dtype=float),
        "deltaE_mean": np.asarray(deltaE_mean, dtype=float),
        "deltaE_q25": np.asarray(deltaE_q25, dtype=float),
        "deltaE_q75": np.asarray(deltaE_q75, dtype=float),
        "match_ref_index": int(match_ref_index),
        "lambda_star_index": int(best_idx),
        "lambda_star": float(best_lambda),
        "match_min": float(best_value),
    }


def compute_md_delta_preference(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    md_sample_id: str,
    model_a_ref: str,
    model_b_ref: str,
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    include_potts_overlay: bool = False,
) -> dict[str, Any]:
    """
    Point (4) diagnostic from validation_ladder2.MD.

    Given one MD sample X (cluster labels per residue), and two Potts models A/B (typically delta patch models),
    compute per-frame and per-residue preferences:

      ΔE(t) = E_A(s_t) - E_B(s_t)
      δ_i(t) = (h^A_i(s_{t,i}) - h^B_i(s_{t,i}))
      δ_{ij}(t) = (J^A_{ij}(s_{t,i}, s_{t,j}) - J^B_{ij}(s_{t,i}, s_{t,j}))

    Returns arrays (means) suitable for visualization:
      - delta_energy: (T,)
      - delta_residue_mean/std: (N,)
      - delta_edge_mean: (E,)
      - edges: (E,2)
    """
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    samples = store.list_samples(project_id, system_id, cluster_id)
    md_entry = next((s for s in samples if s.get("sample_id") == md_sample_id), None)
    if not md_entry:
        raise FileNotFoundError(f"MD sample_id not found on this cluster: {md_sample_id}")

    def _resolve_sample_path(entry: dict[str, Any]) -> Path:
        paths = entry.get("paths") or {}
        rel = None
        if isinstance(paths, dict):
            rel = paths.get("summary_npz") or paths.get("path")
        rel = rel or entry.get("path")
        if not rel:
            raise FileNotFoundError("Sample entry missing path.")
        p = Path(str(rel))
        if not p.is_absolute():
            resolved = store.resolve_path(project_id, system_id, str(rel))
            if not resolved.exists():
                alt = cluster_dir / str(rel)
                p = alt if alt.exists() else resolved
            else:
                p = resolved
        return p

    md_npz_path = _resolve_sample_path(md_entry)
    sample_npz = load_sample_npz(md_npz_path)
    X = sample_npz.labels
    if (md_label_mode or "assigned").lower() in {"halo", "labels_halo"} and sample_npz.labels_halo is not None:
        X = sample_npz.labels_halo
    if drop_invalid and sample_npz.invalid_mask is not None:
        keep = ~np.asarray(sample_npz.invalid_mask, dtype=bool)
        if keep.shape[0] == X.shape[0]:
            X = X[keep]
    X = np.asarray(X, dtype=int)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("MD sample labels are empty.")

    def _resolve_model(ref: str) -> tuple[PottsModel, str | None, str | None, str]:
        model_id = None
        model_name = None
        model_path = Path(ref)
        if not model_path.suffix:
            model_id = str(ref)
            models = store.list_potts_models(project_id, system_id, cluster_id)
            entry = next((m for m in models if m.get("model_id") == model_id), None)
            if not entry or not entry.get("path"):
                raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
            model_name = entry.get("name") or model_id
            model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
        else:
            if not model_path.is_absolute():
                model_path = store.resolve_path(project_id, system_id, str(model_path))
            model_name = model_path.stem
        if not model_path.exists():
            raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
        return load_potts_model(str(model_path)), model_id, model_name, _relativize(model_path, system_dir)

    model_a, model_a_id, model_a_name, model_a_path = _resolve_model(model_a_ref)
    model_b, model_b_id, model_b_name, model_b_path = _resolve_model(model_b_ref)

    if len(model_a.h) != len(model_b.h):
        raise ValueError("Model sizes do not match.")

    # Gauge-fix before decomposing parameters: otherwise large-looking Δh/ΔJ can be pure gauge artifacts.
    model_a = zero_sum_gauge_model(model_a)
    model_b = zero_sum_gauge_model(model_b)

    # Edge set: prefer model A, require availability in model B as well.
    edges = sorted({(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_a.edges or []) if int(r) != int(s)})
    missing = [e for e in edges if e not in {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or [])}]
    if missing:
        # fall back to intersection to stay robust to partial models
        edges_b = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or []) if int(r) != int(s)}
        edges = [e for e in edges if e in edges_b]

    T, N = X.shape
    if N != len(model_a.h):
        raise ValueError("Sample labels do not match model size.")

    dh_list: list[np.ndarray] = []
    for i in range(N):
        dh_list.append(np.asarray(model_a.h[i], dtype=float) - np.asarray(model_b.h[i], dtype=float))
    dJ: dict[tuple[int, int], np.ndarray] = {}
    for (r, s) in edges:
        dJ[(r, s)] = np.asarray(model_a.coupling(r, s), dtype=float) - np.asarray(model_b.coupling(r, s), dtype=float)
    diff_model = PottsModel(h=dh_list, J=dJ, edges=list(edges))

    # Per-residue contributions (store to compute mean/std cheaply).
    delta_res = np.zeros((T, N), dtype=float)
    for i in range(N):
        delta_res[:, i] = dh_list[i][X[:, i]]
    delta_res_mean = np.mean(delta_res, axis=0)
    delta_res_std = np.std(delta_res, axis=0)

    # Per-edge mean and per-frame delta energy (avoid storing T*E).
    delta_energy = diff_model.energy_batch(X)
    edge_sum = np.zeros((len(edges),), dtype=float)
    for idx, (r, s) in enumerate(edges):
        vals = dJ[(r, s)][X[:, r], X[:, s]]
        edge_sum[idx] = float(np.sum(vals))
    delta_edge_mean = edge_sum / float(T)

    delta_energy_potts_a = np.zeros((0,), dtype=float)
    delta_energy_potts_b = np.zeros((0,), dtype=float)
    potts_sample_ids_a: list[str] = []
    potts_sample_ids_b: list[str] = []
    if include_potts_overlay and model_a_id and model_b_id:
        sample_entries = store.list_samples(project_id, system_id, cluster_id)
        potts_samples = [s for s in sample_entries if (s.get("type") or "") == "potts_sampling"]

        def _entry_model_ids(entry: dict[str, Any]) -> list[str]:
            ids: list[str] = []
            raw = entry.get("model_ids")
            if isinstance(raw, list):
                ids = [str(v) for v in raw if v]
            else:
                mid = entry.get("model_id")
                if mid:
                    ids = [str(mid)]
            return ids

        def _load_sample_labels(entry: dict[str, Any]) -> np.ndarray:
            p = _resolve_sample_path(entry)
            s = load_sample_npz(p)
            Xs = np.asarray(s.labels, dtype=int)
            if drop_invalid and s.invalid_mask is not None:
                keep = ~np.asarray(s.invalid_mask, dtype=bool)
                if keep.shape[0] == Xs.shape[0]:
                    Xs = Xs[keep]
            return Xs

        potts_a = []
        potts_b = []
        for entry in potts_samples:
            ids = _entry_model_ids(entry)
            sid = entry.get("sample_id")
            if not sid:
                continue
            if model_a_id in ids:
                potts_a.append(entry)
            if model_b_id in ids:
                potts_b.append(entry)

        def _concat_delta_energy(entries: list[dict[str, Any]], sink: list[str]) -> np.ndarray:
            chunks: list[np.ndarray] = []
            for entry in entries:
                try:
                    Xs = _load_sample_labels(entry)
                except Exception:
                    continue
                if Xs.ndim != 2 or Xs.size == 0:
                    continue
                if Xs.shape[1] != N:
                    continue
                chunks.append(diff_model.energy_batch(Xs))
                sink.append(str(entry.get("sample_id")))
            if not chunks:
                return np.zeros((0,), dtype=float)
            return np.concatenate(chunks, axis=0)

        delta_energy_potts_a = _concat_delta_energy(potts_a, potts_sample_ids_a)
        delta_energy_potts_b = _concat_delta_energy(potts_b, potts_sample_ids_b)

    return {
        "md_sample_id": md_sample_id,
        "md_sample_name": md_entry.get("name"),
        "model_a_id": model_a_id,
        "model_a_name": model_a_name,
        "model_a_path": model_a_path,
        "model_b_id": model_b_id,
        "model_b_name": model_b_name,
        "model_b_path": model_b_path,
        "md_label_mode": md_label_mode,
        "drop_invalid": bool(drop_invalid),
        "delta_energy": np.asarray(delta_energy, dtype=float),
        "delta_residue_mean": np.asarray(delta_res_mean, dtype=float),
        "delta_residue_std": np.asarray(delta_res_std, dtype=float),
        "edges": np.asarray(edges, dtype=int),
        "delta_edge_mean": np.asarray(delta_edge_mean, dtype=float),
        "delta_energy_potts_a": np.asarray(delta_energy_potts_a, dtype=float),
        "delta_energy_potts_b": np.asarray(delta_energy_potts_b, dtype=float),
        "potts_sample_ids_a": potts_sample_ids_a,
        "potts_sample_ids_b": potts_sample_ids_b,
    }


def compute_delta_transition_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    active_md_sample_id: str,
    inactive_md_sample_id: str,
    pas_md_sample_id: str,
    model_a_ref: str,
    model_b_ref: str,
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    band_fraction: float = 0.1,
    top_k_residues: int = 20,
    top_k_edges: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Implements the "TS-like" operational analysis from validation_ladder3.MD.

    Inputs
    ------
    - Three MD-eval samples (ensemble 1 / ensemble 2 / ensemble 3): labels per frame.
    - Two Potts models A/B (typically E_A vs E_I, or equivalently delta-active vs delta-inactive).

    Outputs
    -------
    - delta_energy_{active,inactive,pas}: raw ΔE per frame (E_A - E_B)
    - z_{1,2,3}: robust-normalized coordinate
    - tau: band threshold such that P_train(|z| <= tau) ~= band_fraction
    - p_train, p_3, enrichment: enrichment = log((p_3+eps)/(p_train+eps))
    - D_residue: per-residue discriminative power on fields (mean_active δ_i - mean_inactive δ_i)
    - top_residue_indices: top-K indices by |D|
    - q_residue: commitment probabilities Pr(δ_i < 0) across ensembles {1, 2, 3, TS-band}
    - D_edge: per-edge discriminative power on couplings (mean_1 δ_ij - mean_2 δ_ij)
    - top_edge_indices: top-K edge indices by |D_edge|
    - q_edge: commitment probabilities Pr(δ_ij < 0) for top edges across ensembles {1, 2, 3, TS-band}
    """
    if not (0 < float(band_fraction) < 1):
        raise ValueError("band_fraction must be in (0,1).")
    top_k_residues = int(top_k_residues)
    if top_k_residues < 1:
        raise ValueError("top_k_residues must be >= 1.")
    top_k_edges = int(top_k_edges)
    if top_k_edges < 1:
        raise ValueError("top_k_edges must be >= 1.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    samples = store.list_samples(project_id, system_id, cluster_id)
    active_entry = next((s for s in samples if s.get("sample_id") == active_md_sample_id), None)
    inactive_entry = next((s for s in samples if s.get("sample_id") == inactive_md_sample_id), None)
    pas_entry = next((s for s in samples if s.get("sample_id") == pas_md_sample_id), None)
    if not active_entry:
        raise FileNotFoundError(f"Ensemble 1 sample_id not found on this cluster: {active_md_sample_id}")
    if not inactive_entry:
        raise FileNotFoundError(f"Ensemble 2 sample_id not found on this cluster: {inactive_md_sample_id}")
    if not pas_entry:
        raise FileNotFoundError(f"Ensemble 3 sample_id not found on this cluster: {pas_md_sample_id}")

    def _resolve_sample_path(entry: dict[str, Any]) -> Path:
        paths = entry.get("paths") or {}
        rel = None
        if isinstance(paths, dict):
            rel = paths.get("summary_npz") or paths.get("path")
        rel = rel or entry.get("path")
        if not rel:
            raise FileNotFoundError("Sample entry missing path.")
        p = Path(str(rel))
        if not p.is_absolute():
            resolved = store.resolve_path(project_id, system_id, str(rel))
            if not resolved.exists():
                alt = cluster_dir / str(rel)
                p = alt if alt.exists() else resolved
            else:
                p = resolved
        return p

    def _load_labels(entry: dict[str, Any]) -> np.ndarray:
        p = _resolve_sample_path(entry)
        s = load_sample_npz(p)
        X = s.labels
        if (md_label_mode or "assigned").lower() in {"halo", "labels_halo"} and s.labels_halo is not None:
            X = s.labels_halo
        if drop_invalid and s.invalid_mask is not None:
            keep = ~np.asarray(s.invalid_mask, dtype=bool)
            if keep.shape[0] == X.shape[0]:
                X = X[keep]
        return np.asarray(X, dtype=int)

    X_active = _load_labels(active_entry)
    X_inactive = _load_labels(inactive_entry)
    X_pas = _load_labels(pas_entry)
    if X_active.ndim != 2 or X_active.size == 0:
        raise ValueError("Ensemble 1 labels are empty.")
    if X_inactive.ndim != 2 or X_inactive.size == 0:
        raise ValueError("Ensemble 2 labels are empty.")
    if X_pas.ndim != 2 or X_pas.size == 0:
        raise ValueError("Ensemble 3 labels are empty.")
    if X_active.shape[1] != X_inactive.shape[1] or X_active.shape[1] != X_pas.shape[1]:
        raise ValueError("All ensembles must have the same number of residues.")

    def _resolve_model(ref: str) -> tuple[PottsModel, str | None, str | None, str]:
        model_id = None
        model_name = None
        model_path = Path(ref)
        if not model_path.suffix:
            model_id = str(ref)
            models = store.list_potts_models(project_id, system_id, cluster_id)
            entry = next((m for m in models if m.get("model_id") == model_id), None)
            if not entry or not entry.get("path"):
                raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
            model_name = entry.get("name") or model_id
            model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
        else:
            if not model_path.is_absolute():
                model_path = store.resolve_path(project_id, system_id, str(model_path))
            model_name = model_path.stem
        if not model_path.exists():
            raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
        return load_potts_model(str(model_path)), model_id, model_name, _relativize(model_path, system_dir)

    model_a, model_a_id, model_a_name, model_a_path = _resolve_model(model_a_ref)
    model_b, model_b_id, model_b_name, model_b_path = _resolve_model(model_b_ref)
    if len(model_a.h) != len(model_b.h):
        raise ValueError("Model sizes do not match.")

    model_a = zero_sum_gauge_model(model_a)
    model_b = zero_sum_gauge_model(model_b)

    N = X_active.shape[1]
    if N != len(model_a.h):
        raise ValueError("Labels do not match model size.")

    edges_a = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_a.edges or []) if int(r) != int(s)}
    edges_b = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or []) if int(r) != int(s)}
    edges = sorted(edges_a & edges_b)

    dh_list: list[np.ndarray] = [
        np.asarray(model_a.h[i], dtype=float) - np.asarray(model_b.h[i], dtype=float) for i in range(N)
    ]
    dJ: dict[tuple[int, int], np.ndarray] = {}
    for (r, s) in edges:
        dJ[(r, s)] = np.asarray(model_a.coupling(r, s), dtype=float) - np.asarray(model_b.coupling(r, s), dtype=float)
    diff_model = PottsModel(h=dh_list, J=dJ, edges=list(edges))

    delta_energy_active = diff_model.energy_batch(X_active)
    delta_energy_inactive = diff_model.energy_batch(X_inactive)
    delta_energy_pas = diff_model.energy_batch(X_pas)

    delta_train = np.concatenate([delta_energy_active, delta_energy_inactive], axis=0)
    median_train = float(np.median(delta_train))
    mad_train = float(np.median(np.abs(delta_train - median_train)))
    if not np.isfinite(mad_train) or mad_train <= 1e-12:
        mad_train = float(np.std(delta_train))
    if not np.isfinite(mad_train) or mad_train <= 1e-12:
        mad_train = 1.0

    z_active = (delta_energy_active - median_train) / mad_train
    z_inactive = (delta_energy_inactive - median_train) / mad_train
    z_pas = (delta_energy_pas - median_train) / mad_train
    z_train = np.concatenate([z_active, z_inactive], axis=0)

    abs_z_train = np.abs(z_train)
    tau = float(np.quantile(abs_z_train, float(band_fraction)))
    in_band_train = abs_z_train <= tau
    p_train = float(np.mean(in_band_train)) if in_band_train.size else 0.0
    in_band_pas = np.abs(z_pas) <= tau
    p_pas = float(np.mean(in_band_pas)) if in_band_pas.size else 0.0
    eps = 1e-12
    enrichment = float(np.log((p_pas + eps) / (p_train + eps)))

    def _field_means(X: np.ndarray) -> np.ndarray:
        means = np.zeros((N,), dtype=float)
        for i in range(N):
            means[i] = float(np.mean(dh_list[i][X[:, i]]))
        return means

    mean_active = _field_means(X_active)
    mean_inactive = _field_means(X_inactive)
    D_residue = mean_active - mean_inactive

    top_k = min(int(top_k_residues), int(N))
    top_indices = np.argsort(np.abs(D_residue))[::-1][:top_k].astype(int)

    # Edge discriminative power on training: D_ij = mean_1[δ_ij] - mean_2[δ_ij]
    D_edge = np.zeros((len(edges),), dtype=float)
    if edges:
        for idx, (r, s) in enumerate(edges):
            vals1 = dJ[(r, s)][X_active[:, r], X_active[:, s]]
            vals2 = dJ[(r, s)][X_inactive[:, r], X_inactive[:, s]]
            D_edge[idx] = float(np.mean(vals1) - np.mean(vals2))

    top_k_e = min(int(top_k_edges), int(len(edges)))
    top_edge_indices = (
        np.argsort(np.abs(D_edge))[::-1][:top_k_e].astype(int) if top_k_e > 0 else np.zeros((0,), dtype=int)
    )

    X_train = np.concatenate([X_active, X_inactive], axis=0)
    # Keep naming generic: the UI can map these to selected sample names.
    ensemble_labels = ["Ensemble 1", "Ensemble 2", "Ensemble 3", "TS-band"]
    q = np.zeros((len(ensemble_labels), top_k), dtype=float)
    q_edge = np.zeros((len(ensemble_labels), top_k_e), dtype=float)

    rng = np.random.default_rng(int(seed))
    _ = rng  # reserved for potential bootstrapping later

    for col, idx in enumerate(top_indices.tolist()):
        vals_active = dh_list[idx][X_active[:, idx]]
        vals_inactive = dh_list[idx][X_inactive[:, idx]]
        vals_pas = dh_list[idx][X_pas[:, idx]]
        vals_ts = dh_list[idx][X_train[:, idx]][in_band_train]
        q[0, col] = float(np.mean(vals_active < 0)) if vals_active.size else np.nan
        q[1, col] = float(np.mean(vals_inactive < 0)) if vals_inactive.size else np.nan
        q[2, col] = float(np.mean(vals_pas < 0)) if vals_pas.size else np.nan
        q[3, col] = float(np.mean(vals_ts < 0)) if vals_ts.size else np.nan

    if top_k_e > 0 and edges:
        for col, eidx in enumerate(top_edge_indices.tolist()):
            r, s = edges[int(eidx)]
            vals1 = dJ[(r, s)][X_active[:, r], X_active[:, s]]
            vals2 = dJ[(r, s)][X_inactive[:, r], X_inactive[:, s]]
            vals3 = dJ[(r, s)][X_pas[:, r], X_pas[:, s]]
            vals_ts = dJ[(r, s)][X_train[:, r], X_train[:, s]][in_band_train]
            q_edge[0, col] = float(np.mean(vals1 < 0)) if vals1.size else np.nan
            q_edge[1, col] = float(np.mean(vals2 < 0)) if vals2.size else np.nan
            q_edge[2, col] = float(np.mean(vals3 < 0)) if vals3.size else np.nan
            q_edge[3, col] = float(np.mean(vals_ts < 0)) if vals_ts.size else np.nan

    return {
        "active_md_sample_id": active_md_sample_id,
        "active_md_sample_name": active_entry.get("name"),
        "inactive_md_sample_id": inactive_md_sample_id,
        "inactive_md_sample_name": inactive_entry.get("name"),
        "pas_md_sample_id": pas_md_sample_id,
        "pas_md_sample_name": pas_entry.get("name"),
        "model_a_id": model_a_id,
        "model_a_name": model_a_name,
        "model_a_path": model_a_path,
        "model_b_id": model_b_id,
        "model_b_name": model_b_name,
        "model_b_path": model_b_path,
        "md_label_mode": md_label_mode,
        "drop_invalid": bool(drop_invalid),
        "band_fraction": float(band_fraction),
        "top_k_residues": int(top_k),
        "top_k_edges": int(top_k_e),
        "edges": np.asarray(edges, dtype=int),
        "delta_energy_active": np.asarray(delta_energy_active, dtype=float),
        "delta_energy_inactive": np.asarray(delta_energy_inactive, dtype=float),
        "delta_energy_pas": np.asarray(delta_energy_pas, dtype=float),
        "z_active": np.asarray(z_active, dtype=float),
        "z_inactive": np.asarray(z_inactive, dtype=float),
        "z_pas": np.asarray(z_pas, dtype=float),
        "median_train": float(median_train),
        "mad_train": float(mad_train),
        "tau": float(tau),
        "p_train": float(p_train),
        "p_pas": float(p_pas),
        "enrichment": float(enrichment),
        "D_residue": np.asarray(D_residue, dtype=float),
        "top_residue_indices": np.asarray(top_indices, dtype=int),
        "D_edge": np.asarray(D_edge, dtype=float),
        "top_edge_indices": np.asarray(top_edge_indices, dtype=int),
        "q_residue": np.asarray(q, dtype=float),
        "q_edge": np.asarray(q_edge, dtype=float),
        "ensemble_labels": ensemble_labels,
    }
