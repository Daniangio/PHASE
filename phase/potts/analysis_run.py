from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
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
from phase.potts.sampling import gibbs_sample_potts
from phase.services.project_store import ProjectStore


ANALYSIS_METADATA_FILENAME = "analysis_metadata.json"


def _convert_nan_to_none(obj: Any):
    """
    JSON helper: recursively replace NaN/inf and numpy scalar types with JSON-friendly values.
    """
    if isinstance(obj, dict):
        return {k: _convert_nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_nan_to_none(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return _convert_nan_to_none(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


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


def _default_residue_selection(key: str) -> str:
    match = re.search(r"(?:res[_-]?)(\d+)$", key, flags=re.IGNORECASE)
    if match:
        return f"resid {match.group(1)}"
    if key.isdigit():
        return f"resid {key}"
    return key


def _extract_residue_positions(
    pdb_path: Path,
    residue_keys: Sequence[str],
    residue_mapping: dict[str, str],
    contact_mode: str,
) -> list[np.ndarray | None]:
    import MDAnalysis as mda

    positions: list[np.ndarray | None] = []
    u = mda.Universe(str(pdb_path))
    for key in residue_keys:
        sel = residue_mapping.get(key) or _default_residue_selection(key)
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


def _compute_contact_edges_from_pdbs(
    pdb_paths: Sequence[Path],
    residue_keys: Sequence[str],
    residue_mapping: dict[str, str],
    cutoff: float,
    contact_mode: str,
) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for pdb_path in pdb_paths:
        if not pdb_path.exists():
            continue
        positions = _extract_residue_positions(pdb_path, residue_keys, residue_mapping, contact_mode)
        valid_indices = [i for i, pos in enumerate(positions) if pos is not None]
        if len(valid_indices) < 2:
            continue
        coords = np.stack([positions[i] for i in valid_indices], axis=0)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        for a_idx, i in enumerate(valid_indices):
            for b_idx in range(a_idx + 1, len(valid_indices)):
                j = valid_indices[b_idx]
                if dist[a_idx, b_idx] < cutoff:
                    edges.add((min(i, j), max(i, j)))
    return sorted(edges)


_GIBBS_RELAX_MODEL_CACHE: dict[str, PottsModel] = {}


def _gibbs_relax_worker(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Worker used by the Gibbs-relaxation analysis.

    Each job runs one Gibbs trajectory from a provided starting frame and returns
    summary arrays needed for aggregation (without returning the full trajectory).
    """
    model_path = str(payload["model_path"])
    model = _GIBBS_RELAX_MODEL_CACHE.get(model_path)
    if model is None:
        model = load_potts_model(model_path)
        _GIBBS_RELAX_MODEL_CACHE[model_path] = model

    x0 = np.asarray(payload["x0"], dtype=np.int32).ravel()
    n_sweeps = int(payload["n_sweeps"])
    beta = float(payload["beta"])
    seed = int(payload["seed"])

    traj = gibbs_sample_potts(
        model,
        beta=beta,
        n_samples=n_sweeps,
        burn_in=0,
        thinning=1,
        seed=seed,
        x0=x0,
        progress=False,
    )
    if traj.ndim != 2:
        raise ValueError("Gibbs trajectory must be 2D.")

    diff = traj != x0[None, :]
    any_flip = np.any(diff, axis=0)
    first_flip = np.argmax(diff, axis=0).astype(np.int32) + 1
    first_flip[~any_flip] = np.int32(n_sweeps + 1)

    flip_counts = diff.astype(np.uint16, copy=False)
    energy_trace = np.asarray(model.energy_batch(traj), dtype=np.float32)
    return {
        "first_flip": first_flip.astype(np.int32, copy=False),
        "flip_counts": flip_counts,
        "energy_trace": energy_trace,
    }


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


def run_gibbs_relaxation_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    start_sample_id: str,
    model_ref: str,
    beta: float = 1.0,
    n_start_frames: int = 100,
    gibbs_sweeps: int = 1000,
    seed: int = 0,
    start_label_mode: str = "assigned",
    drop_invalid: bool = True,
    n_workers: int | None = None,
    progress_callback: Optional[callable] = None,
) -> dict[str, Any]:
    """
    Relaxation experiment:
      - pick random starting frames from one MD sample
      - run Gibbs trajectories under a selected Potts Hamiltonian
      - aggregate per-residue first-flip statistics + percentile ranks for coloring
    """
    mode = (start_label_mode or "assigned").strip().lower()
    if mode not in {"assigned", "halo"}:
        raise ValueError("start_label_mode must be 'assigned' or 'halo'.")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be > 0.")
    n_start_frames = int(n_start_frames)
    gibbs_sweeps = int(gibbs_sweeps)
    seed = int(seed)
    if n_start_frames < 1:
        raise ValueError("n_start_frames must be >= 1.")
    if gibbs_sweeps < 1:
        raise ValueError("gibbs_sweeps must be >= 1.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    # Resolve model (id or path)
    model_id = None
    model_name = None
    model_path = Path(str(model_ref))
    if not model_path.suffix:
        model_id = str(model_ref)
        models = store.list_potts_models(project_id, system_id, cluster_id)
        entry = next((m for m in models if m.get("model_id") == model_id), None)
        if not entry or not entry.get("path"):
            raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
        model_name = str(entry.get("name") or model_id)
        model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
    else:
        if not model_path.is_absolute():
            model_path = store.resolve_path(project_id, system_id, str(model_path))
        model_name = model_path.stem
    if not model_path.exists():
        raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")

    model = load_potts_model(str(model_path))
    N = int(len(model.h))
    if N <= 0:
        raise ValueError("Model has no residues.")
    K_list = [int(k) for k in model.K_list()]

    # Resolve starting sample
    samples = store.list_samples(project_id, system_id, cluster_id)
    sample_entry = next((s for s in samples if str(s.get("sample_id")) == str(start_sample_id)), None)
    if not sample_entry:
        raise FileNotFoundError(f"Sample not found on this cluster: {start_sample_id}")

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

    sample_npz = load_sample_npz(_resolve_sample_path(sample_entry))
    X = sample_npz.labels
    if mode in {"halo", "labels_halo"} and sample_npz.labels_halo is not None:
        X = sample_npz.labels_halo
    X = np.asarray(X, dtype=np.int32)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("Starting sample contains no labels.")
    if int(X.shape[1]) != N:
        raise ValueError(f"Starting sample has N={X.shape[1]}, model expects N={N}.")

    frame_indices = (
        np.asarray(sample_npz.frame_indices, dtype=np.int64)
        if sample_npz.frame_indices is not None and sample_npz.frame_indices.shape[0] == X.shape[0]
        else np.arange(X.shape[0], dtype=np.int64)
    )
    frame_state_ids = (
        np.asarray(sample_npz.frame_state_ids, dtype=str)
        if sample_npz.frame_state_ids is not None and sample_npz.frame_state_ids.shape[0] == X.shape[0]
        else np.full((X.shape[0],), "", dtype=str)
    )

    if drop_invalid and sample_npz.invalid_mask is not None:
        keep = ~np.asarray(sample_npz.invalid_mask, dtype=bool).ravel()
        if keep.shape[0] == X.shape[0]:
            X = X[keep]
            frame_indices = frame_indices[keep]
            frame_state_ids = frame_state_ids[keep]

    # Keep only frames with fully valid labels for this model.
    valid = np.all(X >= 0, axis=1)
    for i, k in enumerate(K_list):
        valid &= X[:, i] < int(k)
    if not np.any(valid):
        raise ValueError("No valid starting frames after filtering (invalid/out-of-range labels).")
    X = X[valid]
    frame_indices = frame_indices[valid]
    frame_state_ids = frame_state_ids[valid]

    n_select = min(n_start_frames, int(X.shape[0]))
    rng = np.random.default_rng(seed)
    selected_local = np.asarray(rng.choice(X.shape[0], size=n_select, replace=False), dtype=np.int64)
    selected_starts = np.asarray(X[selected_local], dtype=np.int32)
    selected_frame_indices = np.asarray(frame_indices[selected_local], dtype=np.int64)
    selected_frame_state_ids = np.asarray(frame_state_ids[selected_local], dtype=str)

    # Optional residue labels from cluster metadata.
    residue_keys: list[str] = []
    cluster_npz_path = cluster_dir / "cluster.npz"
    if cluster_npz_path.exists():
        try:
            with np.load(cluster_npz_path, allow_pickle=True) as cnpz:
                if "metadata_json" in cnpz:
                    meta = json.loads(cnpz["metadata_json"].item())
                    if isinstance(meta, dict):
                        residue_keys = [str(v) for v in (meta.get("residue_keys") or [])]
        except Exception:
            residue_keys = []
    if len(residue_keys) != N:
        residue_keys = [f"res_{i}" for i in range(N)]

    # Run independent Gibbs relaxations (parallel across starting frames).
    if n_workers is None or int(n_workers) <= 0:
        workers = os.cpu_count() or 1
    else:
        workers = int(n_workers)
    workers = max(1, min(workers, n_select))

    first_flip = np.zeros((n_select, N), dtype=np.int32)
    flip_counts_sum = np.zeros((gibbs_sweeps, N), dtype=np.uint32)
    energy_traces = np.zeros((n_select, gibbs_sweeps), dtype=np.float32)

    if progress_callback:
        progress_callback("Running Gibbs relaxations...", 0, n_select)

    def _payload(row: int) -> dict[str, Any]:
        return {
            "model_path": str(model_path),
            "x0": selected_starts[row],
            "n_sweeps": int(gibbs_sweeps),
            "beta": float(beta),
            "seed": int(seed) + row,
        }

    if workers <= 1:
        for row in range(n_select):
            out = _gibbs_relax_worker(_payload(row))
            first_flip[row] = np.asarray(out["first_flip"], dtype=np.int32)
            flip_counts_sum += np.asarray(out["flip_counts"], dtype=np.uint32)
            energy_traces[row] = np.asarray(out["energy_trace"], dtype=np.float32)
            if progress_callback:
                progress_callback("Running Gibbs relaxations...", row + 1, n_select)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_gibbs_relax_worker, _payload(row)): row for row in range(n_select)}
            done = 0
            for fut in as_completed(futures):
                row = futures[fut]
                out = fut.result()
                first_flip[row] = np.asarray(out["first_flip"], dtype=np.int32)
                flip_counts_sum += np.asarray(out["flip_counts"], dtype=np.uint32)
                energy_traces[row] = np.asarray(out["energy_trace"], dtype=np.float32)
                done += 1
                if progress_callback:
                    progress_callback("Running Gibbs relaxations...", done, n_select)

    mean_first = np.mean(first_flip, axis=0).astype(np.float32)
    median_first = np.median(first_flip, axis=0).astype(np.float32)
    q25_first = np.quantile(first_flip, 0.25, axis=0).astype(np.float32)
    q75_first = np.quantile(first_flip, 0.75, axis=0).astype(np.float32)

    # Rank-based percentile coloring:
    #   fast percentile = 1.0 for early flippers (small mean first-flip time),
    #   0.0 for late flippers.
    order = np.argsort(mean_first, kind="mergesort")
    pct_fast = np.zeros((N,), dtype=np.float32)
    if N == 1:
        pct_fast[0] = 1.0
    else:
        pct_fast[order] = 1.0 - (np.arange(N, dtype=np.float32) / np.float32(N - 1))
    pct_slow = (1.0 - pct_fast).astype(np.float32)

    flip_prob_time = (flip_counts_sum.astype(np.float32) / float(n_select)).astype(np.float32)
    mean_flip_fraction_by_step = np.mean(flip_prob_time, axis=1).astype(np.float32)
    ever_flip_rate = np.mean(first_flip <= int(gibbs_sweeps), axis=0).astype(np.float32)
    early_cutoff = max(1, int(round(0.25 * float(gibbs_sweeps))))
    early_flip_rate = np.mean(first_flip <= int(early_cutoff), axis=0).astype(np.float32)

    energy_mean = np.mean(energy_traces, axis=0).astype(np.float32)
    energy_std = np.std(energy_traces, axis=0).astype(np.float32)

    top_k = min(20, N)
    top_fast_idx = np.argsort(mean_first)[:top_k].astype(np.int32)
    top_slow_idx = np.argsort(mean_first)[::-1][:top_k].astype(np.int32)

    analysis_id = str(uuid.uuid4())
    out_root = _ensure_analysis_dir(cluster_dir, "gibbs_relaxation")
    analysis_dir = out_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME

    np.savez_compressed(
        npz_path,
        residue_keys=np.asarray(residue_keys, dtype=str),
        start_frame_indices=np.asarray(selected_frame_indices, dtype=np.int64),
        start_frame_state_ids=np.asarray(selected_frame_state_ids, dtype=str),
        first_flip_steps=np.asarray(first_flip, dtype=np.int32),
        mean_first_flip_steps=np.asarray(mean_first, dtype=np.float32),
        median_first_flip_steps=np.asarray(median_first, dtype=np.float32),
        q25_first_flip_steps=np.asarray(q25_first, dtype=np.float32),
        q75_first_flip_steps=np.asarray(q75_first, dtype=np.float32),
        flip_percentile_fast=np.asarray(pct_fast, dtype=np.float32),
        flip_percentile_slow=np.asarray(pct_slow, dtype=np.float32),
        ever_flip_rate=np.asarray(ever_flip_rate, dtype=np.float32),
        early_flip_rate=np.asarray(early_flip_rate, dtype=np.float32),
        flip_prob_time=np.asarray(flip_prob_time, dtype=np.float32),
        mean_flip_fraction_by_step=np.asarray(mean_flip_fraction_by_step, dtype=np.float32),
        energy_traces=np.asarray(energy_traces, dtype=np.float32),
        energy_mean=np.asarray(energy_mean, dtype=np.float32),
        energy_std=np.asarray(energy_std, dtype=np.float32),
        top_fast_indices=np.asarray(top_fast_idx, dtype=np.int32),
        top_slow_indices=np.asarray(top_slow_idx, dtype=np.int32),
        beta=np.asarray([beta], dtype=np.float32),
        gibbs_sweeps=np.asarray([gibbs_sweeps], dtype=np.int32),
        n_start_frames=np.asarray([n_select], dtype=np.int32),
    )

    now = _utc_now()
    meta = {
        "analysis_id": analysis_id,
        "analysis_type": "gibbs_relaxation",
        "created_at": now,
        "updated_at": now,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "start_sample_id": str(start_sample_id),
        "start_sample_name": sample_entry.get("name"),
        "start_sample_type": sample_entry.get("type"),
        "model_id": model_id,
        "model_name": model_name,
        "model_path": _relativize(model_path, system_dir),
        "start_label_mode": mode,
        "drop_invalid": bool(drop_invalid),
        "beta": float(beta),
        "n_start_frames_requested": int(n_start_frames),
        "n_start_frames_used": int(n_select),
        "gibbs_sweeps": int(gibbs_sweeps),
        "seed": int(seed),
        "workers": int(workers),
        "paths": {"analysis_npz": _relativize(npz_path, system_dir)},
        "summary": {
            "n_residues": int(N),
            "mean_first_flip_min": float(np.min(mean_first)) if mean_first.size else None,
            "mean_first_flip_median": float(np.median(mean_first)) if mean_first.size else None,
            "mean_first_flip_max": float(np.max(mean_first)) if mean_first.size else None,
            "early_cutoff_step": int(early_cutoff),
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    return {
        "metadata": _convert_nan_to_none(meta),
        "analysis_npz": str(npz_path),
        "analysis_dir": str(analysis_dir),
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
    dh = np.stack([np.asarray(x, dtype=float).ravel() for x in dh_list], axis=0)
    if dh.shape != (N, K):
        raise ValueError(f"Unexpected dh shape: {dh.shape}, expected {(N, K)}")
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


def upsert_delta_commitment_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str,
    model_b_ref: str,
    sample_ids: Sequence[str],
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    top_k_residues: int = 20,
    top_k_edges: int = 30,
    ranking_method: str = "param_l2",
    energy_bins: int = 80,
) -> dict[str, Any]:
    """
    Incremental A–B commitment store.

    Creates (or updates) a single analysis directory for a fixed (A,B,params) key and stores:
      - Discriminative power (once per analysis key): D_residue, D_edge, top indices, edge list.
      - Per-sample commitment: q_residue, q_edge (rows = samples).
      - Per-sample ΔE histograms on the diff model (E_A - E_B), with a shared binning across samples.

    Notes
    -----
    - We do NOT attempt to be backwards compatible with older delta_transition artifacts.
    - For simplicity and robustness, each call recomputes all stored samples (existing ∪ requested)
      and overwrites the analysis.npz.
    """
    md_label_mode = (md_label_mode or "assigned").strip().lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise ValueError("md_label_mode must be 'assigned' or 'halo'.")
    top_k_residues = int(top_k_residues)
    top_k_edges = int(top_k_edges)
    if top_k_residues < 1:
        raise ValueError("top_k_residues must be >= 1.")
    if top_k_edges < 1:
        raise ValueError("top_k_edges must be >= 1.")
    ranking_method = (ranking_method or "param_l2").strip().lower()
    if ranking_method not in {"param_l2"}:
        raise ValueError("ranking_method must be 'param_l2'.")
    energy_bins = int(energy_bins)
    if energy_bins < 5:
        raise ValueError("energy_bins must be >= 5.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    # Resolve model refs (id or path). We store cluster-relative paths for portability.
    def _resolve_model(ref: str) -> tuple[PottsModel, str | None, str, str]:
        model_id = None
        model_name = None
        model_path = Path(str(ref))
        if not model_path.suffix:
            model_id = str(ref)
            models = store.list_potts_models(project_id, system_id, cluster_id)
            entry = next((m for m in models if m.get("model_id") == model_id), None)
            if not entry or not entry.get("path"):
                raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
            model_name = str(entry.get("name") or model_id)
            model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
        else:
            if not model_path.is_absolute():
                model_path = store.resolve_path(project_id, system_id, str(model_path))
            model_name = model_path.stem
        if not model_path.exists():
            raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
        return load_potts_model(str(model_path)), model_id, str(model_name), _relativize(model_path, system_dir)

    model_a, model_a_id, model_a_name, model_a_path = _resolve_model(model_a_ref)
    model_b, model_b_id, model_b_name, model_b_path = _resolve_model(model_b_ref)
    if model_a_id and model_b_id and model_a_id == model_b_id:
        raise ValueError("Select two different models.")
    if len(model_a.h) != len(model_b.h):
        raise ValueError("Model sizes do not match.")

    # Enforce same gauge before comparing parameters.
    model_a = zero_sum_gauge_model(model_a)
    model_b = zero_sum_gauge_model(model_b)

    N = int(len(model_a.h))
    if N <= 0:
        raise ValueError("Invalid Potts model size.")

    # Variable alphabet sizes per residue are supported (K_i can differ).
    K_list = [int(k) for k in model_a.K_list()]
    if len(K_list) != N:
        raise ValueError("Invalid K_list length.")
    K_max = int(max(K_list)) if K_list else 0
    if K_max <= 0:
        raise ValueError("Invalid Potts model alphabet size.")

    edges_a = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_a.edges or []) if int(r) != int(s)}
    edges_b = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or []) if int(r) != int(s)}
    edges = sorted(edges_a & edges_b)

    dh_list: list[np.ndarray] = []
    for i in range(N):
        a = np.asarray(model_a.h[i], dtype=float).ravel()
        b = np.asarray(model_b.h[i], dtype=float).ravel()
        if a.shape != b.shape:
            raise ValueError(f"Model alphabets do not match at residue {i}: {a.shape} vs {b.shape}")
        dh_list.append(a - b)

    # Padded Δh table for visualization/calibration (variable K_i supported).
    # dh[i, :K_i] is defined; dh[i, K_i:] is zero-padding (use K_list to know valid range).
    dh = np.zeros((N, K_max), dtype=np.float32)
    for i in range(N):
        Ki = int(dh_list[i].shape[0])
        if Ki > 0:
            dh[i, :Ki] = np.asarray(dh_list[i], dtype=np.float32)
    dJ: dict[tuple[int, int], np.ndarray] = {}
    for (r, s) in edges:
        dJ[(r, s)] = np.asarray(model_a.coupling(r, s), dtype=float) - np.asarray(model_b.coupling(r, s), dtype=float)
    diff_model = PottsModel(h=dh_list, J=dJ, edges=list(edges))

    # Discriminative power (parameter-only).
    D_residue = np.zeros((N,), dtype=float)
    for i in range(N):
        D_residue[i] = float(np.linalg.norm(np.asarray(dh_list[i], dtype=float).ravel(), ord=2))
    D_edge = np.zeros((len(edges),), dtype=float)
    for idx, (r, s) in enumerate(edges):
        D_edge[idx] = float(np.linalg.norm(np.asarray(dJ[(r, s)], dtype=float).ravel(), ord=2))

    top_k_r = min(top_k_residues, N)
    top_k_e = min(top_k_edges, len(edges))
    top_residue_indices = np.argsort(D_residue)[::-1][:top_k_r].astype(int)
    top_edge_indices = np.argsort(D_edge)[::-1][:top_k_e].astype(int) if top_k_e > 0 else np.zeros((0,), dtype=int)

    # Locate analysis directory for this (A,B,params) key.
    key = json.dumps(
        {
            "analysis_type": "delta_commitment",
            "model_a_id": model_a_id or model_a_path,
            "model_b_id": model_b_id or model_b_path,
            "md_label_mode": md_label_mode,
            "drop_invalid": bool(drop_invalid),
            "ranking_method": ranking_method,
        },
        sort_keys=True,
    )
    analysis_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    analyses_root = _ensure_analysis_dir(cluster_dir, "delta_commitment")
    analysis_dir = analyses_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME

    # Determine which samples to store: existing ∪ requested.
    existing_sample_ids: list[str] = []
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "sample_ids" in data:
                    existing_sample_ids = [str(x) for x in np.asarray(data["sample_ids"], dtype=str).tolist()]
        except Exception:
            existing_sample_ids = []

    requested = [str(s).strip() for s in sample_ids if str(s).strip()]
    # Keep deterministic ordering: existing first, then new in request order.
    seen = set()
    merged: list[str] = []
    for sid in existing_sample_ids + requested:
        if not sid or sid in seen:
            continue
        seen.add(sid)
        merged.append(sid)
    if not merged:
        raise ValueError("No samples selected.")

    samples = store.list_samples(project_id, system_id, cluster_id)
    sample_by_id: dict[str, dict[str, Any]] = {str(s.get("sample_id")): s for s in samples if s.get("sample_id")}

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
        if md_label_mode in {"halo", "labels_halo"} and s.labels_halo is not None:
            X = s.labels_halo
        if drop_invalid and s.invalid_mask is not None:
            keep = ~np.asarray(s.invalid_mask, dtype=bool)
            if keep.shape[0] == X.shape[0]:
                X = X[keep]
        return np.asarray(X, dtype=int)

    sample_labels: list[str] = []
    sample_types: list[str] = []
    # Store commitment for ALL residues (filtering is a visualization concern).
    q_residue_all = np.zeros((len(merged), N), dtype=float)
    # Per-sample per-residue marginals (for alternative visualizations/calibrations).
    # Shape: (S, N, K_max), with zero-padding for missing states (variable K_i supported).
    p_node = np.zeros((len(merged), N, K_max), dtype=np.float32)
    q_edge = np.zeros((len(merged), top_k_e), dtype=float)
    delta_energy_all: list[np.ndarray] = []
    energy_mean = np.zeros((len(merged),), dtype=float)
    energy_std = np.zeros((len(merged),), dtype=float)

    for row, sid in enumerate(merged):
        entry = sample_by_id.get(sid)
        if not entry:
            raise FileNotFoundError(f"Sample not found on this cluster: {sid}")
        sample_labels.append(str(entry.get("name") or sid))
        sample_types.append(str(entry.get("type") or "sample"))
        X = _load_labels(entry)
        if X.ndim != 2 or X.size == 0:
            raise ValueError(f"Sample labels are empty: {sid}")
        if int(X.shape[1]) != N:
            raise ValueError(f"Sample labels do not match model size for {sid}: got N={X.shape[1]}, expected {N}")
        # Validate label range per residue (variable K_i supported).
        # Note: this analysis assumes assigned labels are in [0, K_i-1].
        if np.min(X) < 0:
            raise ValueError(
                f"Sample contains negative labels for {sid}. "
                "Use md_label_mode='assigned' or remap unassigned labels before analysis."
            )
        for i in range(N):
            Ki = int(K_list[i])
            if Ki <= 0:
                continue
            col = X[:, i]
            mx = int(np.max(col)) if col.size else -1
            if mx >= Ki:
                raise ValueError(
                    f"Sample labels out of range for {sid} at residue {i}: max={mx}, expected in [0,{Ki-1}]"
                )

        n_frames = int(X.shape[0])
        # Node marginals + commitment on all residues: q_i = Pr(dh_i(X_i) < 0)
        # We compute from marginals so that downstream visualizations can reuse p_i(a).
        for i in range(N):
            Ki = int(K_list[i])
            counts = np.bincount(np.asarray(X[:, i], dtype=int), minlength=Ki).astype(np.float32, copy=False)
            if n_frames > 0:
                p = counts / float(n_frames)
            else:
                p = np.zeros((Ki,), dtype=np.float32)
            p_node[row, i, :Ki] = p
            mask = (np.asarray(dh_list[i], dtype=float) < 0).astype(np.float32, copy=False)
            q_residue_all[row, i] = float(np.sum(p * mask)) if p.size else np.nan

        # Commitment on top edges: Pr(dJ_ij(X_i,X_j) < 0)
        if top_k_e > 0 and edges:
            for col, eidx in enumerate(top_edge_indices.tolist()):
                r, s = edges[int(eidx)]
                vals = dJ[(r, s)][X[:, r], X[:, s]]
                q_edge[row, col] = float(np.mean(vals < 0)) if vals.size else np.nan

        de = np.asarray(diff_model.energy_batch(X), dtype=float)
        delta_energy_all.append(de)
        energy_mean[row] = float(np.mean(de)) if de.size else np.nan
        energy_std[row] = float(np.std(de)) if de.size else np.nan

    # Shared energy binning across all samples in this analysis.
    de_concat = np.concatenate(delta_energy_all, axis=0) if delta_energy_all else np.zeros((0,), dtype=float)
    if de_concat.size == 0:
        bins = np.linspace(-1.0, 1.0, energy_bins + 1, dtype=float)
    else:
        lo = float(np.min(de_concat))
        hi = float(np.max(de_concat))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        pad = 1e-6 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, energy_bins + 1, dtype=float)

    energy_hist = np.zeros((len(merged), energy_bins), dtype=float)
    for row, de in enumerate(delta_energy_all):
        h, _ = np.histogram(np.asarray(de, dtype=float), bins=bins, density=True)
        energy_hist[row] = np.asarray(h, dtype=float)

    # Persist NPZ (single file per analysis key).
    np.savez_compressed(
        npz_path,
        edges=np.asarray(edges, dtype=int),
        D_residue=np.asarray(D_residue, dtype=float),
        D_edge=np.asarray(D_edge, dtype=float),
        top_residue_indices=np.asarray(top_residue_indices, dtype=int),
        top_edge_indices=np.asarray(top_edge_indices, dtype=int),
        sample_ids=np.asarray(merged, dtype=str),
        sample_labels=np.asarray(sample_labels, dtype=str),
        sample_types=np.asarray(sample_types, dtype=str),
        K_list=np.asarray(K_list, dtype=int),
        dh=np.asarray(dh, dtype=np.float32),
        p_node=np.asarray(p_node, dtype=np.float32),
        q_residue_all=np.asarray(q_residue_all, dtype=float),
        q_edge=np.asarray(q_edge, dtype=float),
        energy_bins=np.asarray(bins, dtype=float),
        energy_hist=np.asarray(energy_hist, dtype=float),
        energy_mean=np.asarray(energy_mean, dtype=float),
        energy_std=np.asarray(energy_std, dtype=float),
    )

    now = _utc_now()
    created_at = now
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = str(old.get("created_at") or created_at)
        except Exception:
            created_at = now

    meta = {
        "analysis_id": analysis_id,
        "analysis_type": "delta_commitment",
        "created_at": created_at,
        "updated_at": now,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "model_a_id": model_a_id,
        "model_a_name": model_a_name,
        "model_a_path": model_a_path,
        "model_b_id": model_b_id,
        "model_b_name": model_b_name,
        "model_b_path": model_b_path,
        "md_label_mode": md_label_mode,
        "drop_invalid": bool(drop_invalid),
        "top_k_residues": int(top_k_r),
        "top_k_edges": int(top_k_e),
        "ranking_method": ranking_method,
        "energy_bins": int(energy_bins),
        "paths": {"analysis_npz": str(npz_path.relative_to(system_dir))},
        "summary": {
            "n_residues": int(N),
            "n_edges": int(len(edges)),
            "n_samples": int(len(merged)),
            "sample_ids": merged,
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")

    return {"metadata": _convert_nan_to_none(meta), "analysis_npz": str(npz_path), "analysis_dir": str(analysis_dir)}


def upsert_delta_js_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str | None = None,
    model_b_ref: str | None = None,
    sample_ids: Sequence[str],
    reference_sample_ids_a: Sequence[str] | None = None,
    reference_sample_ids_b: Sequence[str] | None = None,
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    top_k_residues: int = 20,
    top_k_edges: int = 30,
    ranking_method: str = "js_ab",
    node_edge_alpha: float = 0.5,
    edge_mode: str | None = None,
    contact_state_ids: Sequence[str] | None = None,
    contact_pdbs: Sequence[str] | None = None,
    contact_cutoff: float = 10.0,
    contact_atom_mode: str = "CA",
) -> dict[str, Any]:
    """
    Incremental JS A-vs-B-vs-Other store.

    For each selected sample:
      - compute per-residue JS distances to A and B references
      - compute per-edge JS distances to A and B references (on top edges)
      - store weighted node/edge aggregate distances for trajectory-level scoring

    Potts models are optional.

    Edge definition:
      - with model A/B: use the intersection of Potts edges and allow automatic reference
        inference from model state_ids.
      - without models: require edge_mode in {'cluster','all_vs_all','contact'} and explicit
        reference_sample_ids_a/b.
    """
    md_label_mode = (md_label_mode or "assigned").strip().lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise ValueError("md_label_mode must be 'assigned' or 'halo'.")
    top_k_residues = int(top_k_residues)
    top_k_edges = int(top_k_edges)
    if top_k_residues < 1:
        raise ValueError("top_k_residues must be >= 1.")
    if top_k_edges < 1:
        raise ValueError("top_k_edges must be >= 1.")
    ranking_method = (ranking_method or "js_ab").strip().lower()
    if ranking_method not in {"js_ab"}:
        raise ValueError("ranking_method must be 'js_ab'.")
    node_edge_alpha = float(node_edge_alpha)
    if not np.isfinite(node_edge_alpha) or node_edge_alpha < 0.0 or node_edge_alpha > 1.0:
        raise ValueError("node_edge_alpha must be in [0,1].")

    edge_mode = (edge_mode or "").strip().lower()
    if edge_mode and edge_mode not in {"cluster", "all_vs_all", "contact"}:
        raise ValueError("edge_mode must be one of: cluster, all_vs_all, contact.")
    contact_cutoff = float(contact_cutoff)
    if not np.isfinite(contact_cutoff) or contact_cutoff <= 0:
        raise ValueError("contact_cutoff must be > 0.")
    contact_atom_mode = str(contact_atom_mode or "CA").strip().upper()
    if contact_atom_mode not in {"CA", "CM"}:
        raise ValueError("contact_atom_mode must be 'CA' or 'CM'.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    cluster_npz_path = cluster_dir / "cluster.npz"
    if not cluster_npz_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found: {cluster_npz_path}")

    def _load_cluster_topology(path: Path) -> tuple[int, list[int], list[tuple[int, int]], list[str]]:
        with np.load(path, allow_pickle=True) as data:
            if "residue_keys" in data:
                residue_keys_raw = np.asarray(data["residue_keys"], dtype=str)
                residue_keys_local = [str(x) for x in residue_keys_raw.tolist()]
            else:
                residue_keys_local = []
            if "cluster_counts" in data:
                cc = np.asarray(data["cluster_counts"], dtype=int)
            elif "merged__cluster_counts" in data:
                cc = np.asarray(data["merged__cluster_counts"], dtype=int)
            else:
                raise KeyError("cluster_counts / merged__cluster_counts not found in cluster NPZ.")

            raw_edges: np.ndarray
            if "contact_edge_index" in data:
                edge_idx = np.asarray(data["contact_edge_index"], dtype=int)
                if edge_idx.ndim == 2 and edge_idx.shape[0] == 2:
                    raw_edges = edge_idx.T
                else:
                    raw_edges = np.zeros((0, 2), dtype=int)
            elif "edges" in data:
                edge_arr = np.asarray(data["edges"], dtype=int)
                if edge_arr.ndim == 2 and edge_arr.shape[1] >= 2:
                    raw_edges = edge_arr[:, :2]
                elif edge_arr.ndim == 2 and edge_arr.shape[0] == 2:
                    raw_edges = edge_arr.T
                else:
                    raw_edges = np.zeros((0, 2), dtype=int)
            else:
                raw_edges = np.zeros((0, 2), dtype=int)

        N_local = int(cc.shape[0])
        if N_local <= 0:
            raise ValueError("Invalid cluster topology (zero residues).")
        if len(residue_keys_local) != N_local:
            residue_keys_local = [f"res_{i}" for i in range(N_local)]
        K_local = [int(x) for x in cc.tolist()]
        if any(k <= 0 for k in K_local):
            raise ValueError("Invalid cluster_counts in cluster NPZ.")

        edge_set: set[tuple[int, int]] = set()
        if raw_edges.size:
            for pair in np.asarray(raw_edges, dtype=int):
                if pair.shape[0] < 2:
                    continue
                r = int(pair[0])
                s = int(pair[1])
                if r == s:
                    continue
                if r < 0 or s < 0 or r >= N_local or s >= N_local:
                    continue
                if r > s:
                    r, s = s, r
                edge_set.add((r, s))
        return N_local, K_local, sorted(edge_set), residue_keys_local

    cluster_N, cluster_K_list, cluster_edges, residue_keys = _load_cluster_topology(cluster_npz_path)

    models_meta = store.list_potts_models(project_id, system_id, cluster_id)
    model_by_id: dict[str, dict[str, Any]] = {
        str(m.get("model_id")): m for m in models_meta if isinstance(m, dict) and m.get("model_id")
    }

    def _model_state_ids(model_entry: dict[str, Any] | None) -> list[str]:
        if not isinstance(model_entry, dict):
            return []
        params = model_entry.get("params")
        if not isinstance(params, dict):
            return []
        raw = params.get("state_ids")
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for sid in raw:
            s = str(sid or "").strip()
            if s:
                out.append(s)
        return out

    def _resolve_model(ref: str) -> tuple[PottsModel, str | None, str, str, dict[str, Any] | None]:
        model_id = None
        model_name = None
        model_path = Path(str(ref))
        model_entry = None
        if not model_path.suffix:
            model_id = str(ref)
            model_entry = model_by_id.get(model_id)
            if not model_entry or not model_entry.get("path"):
                raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
            model_name = str(model_entry.get("name") or model_id)
            model_path = store.resolve_path(project_id, system_id, str(model_entry.get("path")))
        else:
            if not model_path.is_absolute():
                model_path = store.resolve_path(project_id, system_id, str(model_path))
            model_name = model_path.stem
        if not model_path.exists():
            raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
        return (
            load_potts_model(str(model_path)),
            model_id,
            str(model_name),
            _relativize(model_path, system_dir),
            model_entry,
        )

    model_a_ref = str(model_a_ref or "").strip()
    model_b_ref = str(model_b_ref or "").strip()
    use_models = bool(model_a_ref or model_b_ref)
    if use_models and (not model_a_ref or not model_b_ref):
        raise ValueError("Provide both model_a_ref and model_b_ref, or neither.")

    model_a_id: str | None = None
    model_b_id: str | None = None
    model_a_name: str | None = None
    model_b_name: str | None = None
    model_a_path: str | None = None
    model_b_path: str | None = None
    model_a_entry: dict[str, Any] | None = None
    model_b_entry: dict[str, Any] | None = None
    edge_source = "cluster"

    if use_models:
        model_a, model_a_id, model_a_name, model_a_path, model_a_entry = _resolve_model(model_a_ref)
        model_b, model_b_id, model_b_name, model_b_path, model_b_entry = _resolve_model(model_b_ref)
        if model_a_id and model_b_id and model_a_id == model_b_id:
            raise ValueError("Select two different models.")
        if len(model_a.h) != len(model_b.h):
            raise ValueError("Model sizes do not match.")

        model_a = zero_sum_gauge_model(model_a)
        model_b = zero_sum_gauge_model(model_b)
        N = int(len(model_a.h))
        if N <= 0:
            raise ValueError("Invalid Potts model size.")
        K_list = [int(k) for k in model_a.K_list()]
        K_list_b = [int(k) for k in model_b.K_list()]
        if len(K_list) != N or len(K_list_b) != N:
            raise ValueError("Invalid model K_list length.")
        if K_list != K_list_b:
            raise ValueError("Model alphabet sizes do not match.")

        if cluster_N != N:
            raise ValueError(
                f"Model size mismatch with cluster topology: model N={N}, cluster N={cluster_N}."
            )
        if cluster_K_list and K_list != cluster_K_list:
            raise ValueError("Model alphabet sizes do not match cluster_counts.")

        edges_a = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_a.edges or []) if int(r) != int(s)}
        edges_b = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or []) if int(r) != int(s)}
        edges = sorted(edges_a & edges_b)
        edge_source = "potts_intersection"
    else:
        if not edge_mode:
            raise ValueError("edge_mode is required when Potts models are not provided.")
        N = int(cluster_N)
        K_list = list(cluster_K_list)
        if edge_mode == "cluster":
            edges = list(cluster_edges)
            edge_source = "cluster"
        elif edge_mode == "all_vs_all":
            edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
            edge_source = "all_vs_all"
        else:
            system_meta = store.get_system(project_id, system_id)
            state_map = system_meta.states or {}

            raw_state_ids = [str(s or "").strip() for s in (contact_state_ids or []) if str(s or "").strip()]
            raw_pdbs = [str(p or "").strip() for p in (contact_pdbs or []) if str(p or "").strip()]

            resolved_pdbs: list[Path] = []
            seen_pdb: set[str] = set()

            for sid in raw_state_ids:
                state = state_map.get(sid)
                pdb_rel = state.pdb_file if state and getattr(state, "pdb_file", None) else ""
                if not pdb_rel:
                    continue
                p = Path(str(pdb_rel))
                if not p.is_absolute():
                    p = store.resolve_path(project_id, system_id, str(pdb_rel))
                key = str(p.resolve()) if p.exists() else str(p)
                if key in seen_pdb:
                    continue
                seen_pdb.add(key)
                resolved_pdbs.append(p)

            for raw in raw_pdbs:
                p = Path(raw)
                if not p.is_absolute():
                    p = store.resolve_path(project_id, system_id, raw)
                key = str(p.resolve()) if p.exists() else str(p)
                if key in seen_pdb:
                    continue
                seen_pdb.add(key)
                resolved_pdbs.append(p)

            resolved_pdbs = [p for p in resolved_pdbs if p.exists()]
            if not resolved_pdbs:
                raise ValueError(
                    "edge_mode=contact requires at least one valid PDB from contact_state_ids or contact_pdbs."
                )
            edges = _compute_contact_edges_from_pdbs(
                resolved_pdbs,
                residue_keys,
                {},
                float(contact_cutoff),
                str(contact_atom_mode).upper(),
            )
            edge_source = "contact"

    K_max = int(max(K_list)) if K_list else 0
    if K_max <= 0:
        raise ValueError("Invalid alphabet size.")

    samples = store.list_samples(project_id, system_id, cluster_id)
    sample_by_id: dict[str, dict[str, Any]] = {
        str(s.get("sample_id")): s for s in samples if isinstance(s, dict) and s.get("sample_id")
    }

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
        if md_label_mode in {"halo", "labels_halo"} and s.labels_halo is not None:
            X = s.labels_halo
        if drop_invalid and s.invalid_mask is not None:
            keep = ~np.asarray(s.invalid_mask, dtype=bool)
            if keep.shape[0] == X.shape[0]:
                X = X[keep]
        return np.asarray(X, dtype=int)

    def _resolve_reference_ids(
        provided_ids: Sequence[str] | None,
        model_entry: dict[str, Any] | None,
        side_label: str,
        *,
        allow_infer_from_model: bool,
    ) -> list[str]:
        if provided_ids:
            out: list[str] = []
            seen: set[str] = set()
            for sid in provided_ids:
                s = str(sid or "").strip()
                if not s or s in seen:
                    continue
                if s not in sample_by_id:
                    raise FileNotFoundError(f"Reference sample not found ({side_label}): {s}")
                out.append(s)
                seen.add(s)
            if not out:
                raise ValueError(f"No valid reference samples selected for side {side_label}.")
            return out

        if not allow_infer_from_model:
            raise ValueError(
                f"reference_sample_ids_{side_label.lower()} is required when Potts models are not provided."
            )

        state_ids = _model_state_ids(model_entry)
        if not state_ids:
            raise ValueError(
                f"Could not infer reference samples for side {side_label}: model has no state_ids. "
                f"Provide reference_sample_ids_{side_label.lower()} explicitly."
            )
        refs: list[str] = []
        for sid, entry in sample_by_id.items():
            if str(entry.get("type") or "") != "md_eval":
                continue
            state_id = str(entry.get("state_id") or "").strip()
            if state_ids and state_id not in state_ids:
                continue
            refs.append(sid)
        if refs:
            return refs
        raise ValueError(
            f"Could not infer reference samples for side {side_label}. "
            f"Provide reference_sample_ids_{side_label.lower()} explicitly."
        )

    ref_ids_a = _resolve_reference_ids(
        reference_sample_ids_a,
        model_a_entry,
        "A",
        allow_infer_from_model=use_models,
    )
    ref_ids_b = _resolve_reference_ids(
        reference_sample_ids_b,
        model_b_entry,
        "B",
        allow_infer_from_model=use_models,
    )

    def _analysis_key() -> str:
        payload = {
            "analysis_type": "delta_js",
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "model_a_path": model_a_path,
            "model_b_path": model_b_path,
            "edge_source": edge_source,
            "edge_mode": edge_mode,
            "contact_state_ids": list(map(str, sorted({str(s).strip() for s in (contact_state_ids or []) if str(s).strip()}))),
            "contact_pdbs": list(map(str, sorted({str(p).strip() for p in (contact_pdbs or []) if str(p).strip()}))),
            "contact_cutoff": float(contact_cutoff),
            "contact_atom_mode": str(contact_atom_mode).upper(),
            "md_label_mode": md_label_mode,
            "drop_invalid": bool(drop_invalid),
            "ranking_method": ranking_method,
            "node_edge_alpha": float(node_edge_alpha),
            "ref_a": list(map(str, sorted(ref_ids_a))),
            "ref_b": list(map(str, sorted(ref_ids_b))),
        }
        return json.dumps(payload, sort_keys=True)

    analysis_id = str(uuid.uuid5(uuid.NAMESPACE_URL, _analysis_key()))
    analyses_root = _ensure_analysis_dir(cluster_dir, "delta_js")
    analysis_dir = analyses_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME

    existing_sample_ids: list[str] = []
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "sample_ids" in data:
                    existing_sample_ids = [str(x) for x in np.asarray(data["sample_ids"], dtype=str).tolist()]
        except Exception:
            existing_sample_ids = []

    requested = [str(s).strip() for s in sample_ids if str(s).strip()]
    seen: set[str] = set()
    merged: list[str] = []
    for sid in existing_sample_ids + requested:
        if not sid or sid in seen:
            continue
        seen.add(sid)
        merged.append(sid)
    if not merged:
        raise ValueError("No samples selected.")

    def _aggregate_refs(ref_ids: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        node_counts = [np.zeros((int(K_list[i]),), dtype=float) for i in range(N)]
        edge_counts = [np.zeros((int(K_list[r]), int(K_list[s])), dtype=float) for (r, s) in edges]
        total = 0
        for sid in ref_ids:
            entry = sample_by_id.get(sid)
            if not entry:
                continue
            X = _load_labels(entry)
            if X.ndim != 2 or X.size == 0:
                continue
            if int(X.shape[1]) != N:
                raise ValueError(f"Reference sample size mismatch ({sid}): got N={X.shape[1]}, expected {N}")
            if np.min(X) < 0:
                raise ValueError(
                    f"Reference sample contains negative labels ({sid}). "
                    "Use md_label_mode='assigned' or remap unassigned labels first."
                )
            for i in range(N):
                Ki = int(K_list[i])
                if Ki <= 0:
                    continue
                col = X[:, i]
                mx = int(np.max(col)) if col.size else -1
                if mx >= Ki:
                    raise ValueError(
                        f"Reference sample labels out of range for {sid} at residue {i}: max={mx}, expected in [0,{Ki-1}]"
                    )
            T = int(X.shape[0])
            total += T
            for i in range(N):
                node_counts[i] += np.bincount(np.asarray(X[:, i], dtype=int), minlength=int(K_list[i])).astype(float)
            if edges:
                P = pairwise_joints_on_edges(X, K_list, edges)
                for eidx, e in enumerate(edges):
                    edge_counts[eidx] += np.asarray(P[e], dtype=float) * float(T)
        if total <= 0:
            raise ValueError("Reference samples are empty after filtering.")
        p_node = [c / max(1.0, float(np.sum(c))) for c in node_counts]
        p_edge = [c / max(1.0, float(np.sum(c))) for c in edge_counts]
        return p_node, p_edge

    p_node_a, p_edge_a = _aggregate_refs(ref_ids_a)
    p_node_b, p_edge_b = _aggregate_refs(ref_ids_b)

    D_residue = np.zeros((N,), dtype=float)
    for i in range(N):
        D_residue[i] = float(js_divergence(np.asarray(p_node_a[i], dtype=float), np.asarray(p_node_b[i], dtype=float)))
    D_edge = np.zeros((len(edges),), dtype=float)
    for eidx, _ in enumerate(edges):
        D_edge[eidx] = float(js_divergence(np.asarray(p_edge_a[eidx], dtype=float).ravel(), np.asarray(p_edge_b[eidx], dtype=float).ravel()))

    top_k_r = min(top_k_residues, N)
    top_k_e = min(top_k_edges, len(edges))
    top_residue_indices = np.argsort(D_residue)[::-1][:top_k_r].astype(int)
    top_edge_indices = np.argsort(D_edge)[::-1][:top_k_e].astype(int) if top_k_e > 0 else np.zeros((0,), dtype=int)
    top_edges = [edges[int(eidx)] for eidx in top_edge_indices.tolist()]

    def _weighted_mean(vals: np.ndarray, weights: np.ndarray) -> float:
        v = np.asarray(vals, dtype=float)
        w = np.asarray(weights, dtype=float)
        good = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(good):
            return float(np.nan)
        ws = float(np.sum(w[good]))
        if ws <= 0:
            return float(np.nan)
        return float(np.sum(v[good] * w[good]) / ws)

    sample_labels: list[str] = []
    sample_types: list[str] = []
    js_node_a = np.zeros((len(merged), N), dtype=float)
    js_node_b = np.zeros((len(merged), N), dtype=float)
    js_edge_a = np.zeros((len(merged), top_k_e), dtype=float)
    js_edge_b = np.zeros((len(merged), top_k_e), dtype=float)
    js_node_weighted_a = np.zeros((len(merged),), dtype=float)
    js_node_weighted_b = np.zeros((len(merged),), dtype=float)
    js_edge_weighted_a = np.zeros((len(merged),), dtype=float)
    js_edge_weighted_b = np.zeros((len(merged),), dtype=float)
    js_mixed_a = np.zeros((len(merged),), dtype=float)
    js_mixed_b = np.zeros((len(merged),), dtype=float)

    top_edge_weights = np.asarray([float(D_edge[int(eidx)]) for eidx in top_edge_indices.tolist()], dtype=float)
    for row, sid in enumerate(merged):
        entry = sample_by_id.get(sid)
        if not entry:
            raise FileNotFoundError(f"Sample not found on this cluster: {sid}")
        sample_labels.append(str(entry.get("name") or sid))
        sample_types.append(str(entry.get("type") or "sample"))
        X = _load_labels(entry)
        if X.ndim != 2 or X.size == 0:
            raise ValueError(f"Sample labels are empty: {sid}")
        if int(X.shape[1]) != N:
            raise ValueError(f"Sample labels do not match model size for {sid}: got N={X.shape[1]}, expected {N}")
        if np.min(X) < 0:
            raise ValueError(
                f"Sample contains negative labels for {sid}. "
                "Use md_label_mode='assigned' or remap unassigned labels before analysis."
            )
        for i in range(N):
            Ki = int(K_list[i])
            if Ki <= 0:
                continue
            col = X[:, i]
            mx = int(np.max(col)) if col.size else -1
            if mx >= Ki:
                raise ValueError(
                    f"Sample labels out of range for {sid} at residue {i}: max={mx}, expected in [0,{Ki-1}]"
                )

        p_s = marginals(X, K_list)
        for i in range(N):
            js_node_a[row, i] = float(js_divergence(np.asarray(p_s[i], dtype=float), np.asarray(p_node_a[i], dtype=float)))
            js_node_b[row, i] = float(js_divergence(np.asarray(p_s[i], dtype=float), np.asarray(p_node_b[i], dtype=float)))
        js_node_weighted_a[row] = _weighted_mean(js_node_a[row], D_residue)
        js_node_weighted_b[row] = _weighted_mean(js_node_b[row], D_residue)

        if top_k_e > 0:
            p2_s_top = pairwise_joints_on_edges(X, K_list, top_edges)
            for col, e in enumerate(top_edges):
                eidx = int(top_edge_indices[col])
                js_edge_a[row, col] = float(
                    js_divergence(np.asarray(p2_s_top[e], dtype=float).ravel(), np.asarray(p_edge_a[eidx], dtype=float).ravel())
                )
                js_edge_b[row, col] = float(
                    js_divergence(np.asarray(p2_s_top[e], dtype=float).ravel(), np.asarray(p_edge_b[eidx], dtype=float).ravel())
                )
            js_edge_weighted_a[row] = _weighted_mean(js_edge_a[row], top_edge_weights)
            js_edge_weighted_b[row] = _weighted_mean(js_edge_b[row], top_edge_weights)
        else:
            js_edge_weighted_a[row] = js_node_weighted_a[row]
            js_edge_weighted_b[row] = js_node_weighted_b[row]

        a_node = float(js_node_weighted_a[row])
        b_node = float(js_node_weighted_b[row])
        a_edge = float(js_edge_weighted_a[row])
        b_edge = float(js_edge_weighted_b[row])
        js_mixed_a[row] = (1.0 - node_edge_alpha) * a_node + node_edge_alpha * a_edge
        js_mixed_b[row] = (1.0 - node_edge_alpha) * b_node + node_edge_alpha * b_edge

    p_node_ref_a_padded = np.zeros((N, K_max), dtype=float)
    p_node_ref_b_padded = np.zeros((N, K_max), dtype=float)
    for i in range(N):
        Ki = int(K_list[i])
        p_node_ref_a_padded[i, :Ki] = np.asarray(p_node_a[i], dtype=float)
        p_node_ref_b_padded[i, :Ki] = np.asarray(p_node_b[i], dtype=float)

    np.savez_compressed(
        npz_path,
        edges=np.asarray(edges, dtype=int),
        D_residue=np.asarray(D_residue, dtype=float),
        D_edge=np.asarray(D_edge, dtype=float),
        top_residue_indices=np.asarray(top_residue_indices, dtype=int),
        top_edge_indices=np.asarray(top_edge_indices, dtype=int),
        sample_ids=np.asarray(merged, dtype=str),
        sample_labels=np.asarray(sample_labels, dtype=str),
        sample_types=np.asarray(sample_types, dtype=str),
        K_list=np.asarray(K_list, dtype=int),
        ref_sample_ids_a=np.asarray(ref_ids_a, dtype=str),
        ref_sample_ids_b=np.asarray(ref_ids_b, dtype=str),
        p_node_ref_a=np.asarray(p_node_ref_a_padded, dtype=float),
        p_node_ref_b=np.asarray(p_node_ref_b_padded, dtype=float),
        js_node_a=np.asarray(js_node_a, dtype=float),
        js_node_b=np.asarray(js_node_b, dtype=float),
        js_edge_a=np.asarray(js_edge_a, dtype=float),
        js_edge_b=np.asarray(js_edge_b, dtype=float),
        js_node_weighted_a=np.asarray(js_node_weighted_a, dtype=float),
        js_node_weighted_b=np.asarray(js_node_weighted_b, dtype=float),
        js_edge_weighted_a=np.asarray(js_edge_weighted_a, dtype=float),
        js_edge_weighted_b=np.asarray(js_edge_weighted_b, dtype=float),
        js_mixed_a=np.asarray(js_mixed_a, dtype=float),
        js_mixed_b=np.asarray(js_mixed_b, dtype=float),
        node_edge_alpha=np.asarray([float(node_edge_alpha)], dtype=float),
    )

    now = _utc_now()
    created_at = now
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = str(old.get("created_at") or created_at)
        except Exception:
            created_at = now

    meta = {
        "analysis_id": analysis_id,
        "analysis_type": "delta_js",
        "created_at": created_at,
        "updated_at": now,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "model_a_id": model_a_id,
        "model_a_name": model_a_name,
        "model_a_path": model_a_path,
        "model_b_id": model_b_id,
        "model_b_name": model_b_name,
        "model_b_path": model_b_path,
        "edge_source": edge_source,
        "edge_mode": edge_mode or edge_source,
        "contact_state_ids": [str(s).strip() for s in (contact_state_ids or []) if str(s).strip()],
        "contact_pdbs": [str(p).strip() for p in (contact_pdbs or []) if str(p).strip()],
        "contact_cutoff": float(contact_cutoff) if str(edge_mode).lower() == "contact" else None,
        "contact_atom_mode": str(contact_atom_mode).upper() if str(edge_mode).lower() == "contact" else None,
        "md_label_mode": md_label_mode,
        "drop_invalid": bool(drop_invalid),
        "top_k_residues": int(top_k_r),
        "top_k_edges": int(top_k_e),
        "ranking_method": ranking_method,
        "node_edge_alpha": float(node_edge_alpha),
        "reference_sample_ids_a": list(ref_ids_a),
        "reference_sample_ids_b": list(ref_ids_b),
        "paths": {"analysis_npz": str(npz_path.relative_to(system_dir))},
        "summary": {
            "n_residues": int(N),
            "n_edges": int(len(edges)),
            "n_samples": int(len(merged)),
            "sample_ids": merged,
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")

    return {"metadata": _convert_nan_to_none(meta), "analysis_npz": str(npz_path), "analysis_dir": str(analysis_dir)}
