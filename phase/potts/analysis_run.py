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
from phase.potts.potts_model import PottsModel, load_potts_model
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

    # Per-residue contributions (store to compute mean/std cheaply).
    delta_res = np.zeros((T, N), dtype=float)
    for i in range(N):
        dh = np.asarray(model_a.h[i], dtype=float) - np.asarray(model_b.h[i], dtype=float)
        delta_res[:, i] = dh[X[:, i]]
    delta_res_mean = np.mean(delta_res, axis=0)
    delta_res_std = np.std(delta_res, axis=0)

    # Per-edge mean and per-frame delta energy (avoid storing T*E).
    delta_energy = np.sum(delta_res, axis=1)
    edge_sum = np.zeros((len(edges),), dtype=float)
    for idx, (r, s) in enumerate(edges):
        dJ = np.asarray(model_a.coupling(r, s), dtype=float) - np.asarray(model_b.coupling(r, s), dtype=float)
        vals = dJ[X[:, r], X[:, s]]
        edge_sum[idx] = float(np.sum(vals))
        delta_energy = delta_energy + vals
    delta_edge_mean = edge_sum / float(T)

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
    }
