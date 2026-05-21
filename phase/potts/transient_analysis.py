from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from phase.potts.sample_io import load_sample_npz
from phase.services.project_store import ProjectStore

ProgressCallback = Optional[Callable[[str, int, int], None]]
ANALYSIS_METADATA_FILENAME = "analysis_metadata.json"


def _utc_now() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat()


def _convert_nan_to_none(obj: Any):
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


def _ensure_analysis_dir(cluster_dir: Path, kind: str) -> Path:
    root = cluster_dir / "analyses" / kind
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_edges(data: np.lib.npyio.NpzFile, n_residues: int) -> list[tuple[int, int]]:
    raw = None
    if "contact_edge_index" in data:
        arr = np.asarray(data["contact_edge_index"], dtype=int)
        if arr.ndim == 2 and arr.shape[0] == 2:
            raw = arr.T
    if raw is None and "edges" in data:
        arr = np.asarray(data["edges"], dtype=int)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            raw = arr[:, :2]
        elif arr.ndim == 2 and arr.shape[0] == 2:
            raw = arr.T
    if raw is None:
        return []
    out: set[tuple[int, int]] = set()
    for r, s in np.asarray(raw, dtype=int):
        r = int(r); s = int(s)
        if r == s or r < 0 or s < 0 or r >= n_residues or s >= n_residues:
            continue
        if r > s:
            r, s = s, r
        out.add((r, s))
    return sorted(out)


def _cluster_topology(cluster_npz_path: Path) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    with np.load(cluster_npz_path, allow_pickle=True) as data:
        if "cluster_counts" in data:
            counts = np.asarray(data["cluster_counts"], dtype=int)
        elif "merged__cluster_counts" in data:
            counts = np.asarray(data["merged__cluster_counts"], dtype=int)
        else:
            raise KeyError("cluster_counts / merged__cluster_counts not found in cluster NPZ.")
        n = int(counts.shape[0])
        if "residue_keys" in data:
            labels = [str(x) for x in np.asarray(data["residue_keys"], dtype=str).tolist()]
        else:
            labels = [f"res_{i}" for i in range(n)]
        edges = _parse_edges(data, n)
    if len(labels) != n:
        labels = [f"res_{i}" for i in range(n)]
    k_list = [int(x) for x in counts.tolist()]
    if any(k <= 0 for k in k_list):
        raise ValueError("Invalid cluster_counts in cluster NPZ.")
    return labels, k_list, edges


def _resolve_sample_path(store: ProjectStore, project_id: str, system_id: str, cluster_dir: Path, entry: dict[str, Any]) -> Path:
    paths = entry.get("paths") or {}
    rel = paths.get("summary_npz") if isinstance(paths, dict) else None
    rel = rel or (paths.get("path") if isinstance(paths, dict) else None) or entry.get("path")
    if not rel:
        raise FileNotFoundError("Sample entry missing path.")
    p = Path(str(rel))
    if p.is_absolute():
        return p
    resolved = store.resolve_path(project_id, system_id, str(rel))
    if resolved.exists():
        return resolved
    alt = cluster_dir / str(rel)
    return alt if alt.exists() else resolved


def _load_labels(path: Path, *, md_label_mode: str, drop_invalid: bool, sample_type: str) -> np.ndarray:
    sample = load_sample_npz(path)
    X = sample.labels
    if str(sample_type or "") == "md_eval" and md_label_mode in {"halo", "labels_halo"} and sample.labels_halo is not None:
        X = sample.labels_halo
    if drop_invalid and sample.invalid_mask is not None:
        keep = ~np.asarray(sample.invalid_mask, dtype=bool)
        if keep.shape[0] == X.shape[0]:
            X = X[keep]
    X = np.asarray(X, dtype=np.int32)
    if X.ndim != 2:
        raise ValueError(f"Sample labels must be 2D: {path}")
    return X


def _episodes(mask: np.ndarray) -> tuple[int, float, int]:
    b = np.asarray(mask, dtype=bool).ravel()
    if b.size == 0 or not np.any(b):
        return 0, 0.0, 0
    padded = np.concatenate([[False], b, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    lengths = stops - starts
    if lengths.size == 0:
        return 0, 0.0, 0
    return int(lengths.size), float(np.mean(lengths)), int(np.max(lengths))


def _safe_log2_ratio(p: float, bg: float, eps: float) -> float:
    return float(np.log2((float(p) + eps) / (float(bg) + eps)))


def _safe_pmi(pij: float, pi: float, pj: float, eps: float) -> float:
    return float(np.log((float(pij) + eps) / (float(pi) * float(pj) + eps)))


def run_transient_state_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_ids: Sequence[str],
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    p_min: float = 0.005,
    p_max: float = 0.05,
    enrichment_min: float = 1.0,
    epsilon: float = 1e-9,
    top_k_nodes: int = 500,
    include_edges: bool = True,
    edge_mode: str = "cluster",
    edge_p_min: Optional[float] = None,
    edge_p_max: Optional[float] = None,
    edge_enrichment_min: Optional[float] = None,
    delta_pmi_min: Optional[float] = None,
    top_k_edges: int = 1000,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    md_label_mode = str(md_label_mode or "assigned").strip().lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise ValueError("md_label_mode must be 'assigned' or 'halo'.")
    edge_mode = str(edge_mode or "cluster").strip().lower()
    if edge_mode not in {"cluster", "all_vs_all"}:
        raise ValueError("edge_mode must be 'cluster' or 'all_vs_all'.")
    p_min = float(p_min); p_max = float(p_max); enrichment_min = float(enrichment_min); epsilon = float(epsilon)
    edge_p_min = float(p_min if edge_p_min is None else edge_p_min)
    edge_p_max = float(p_max if edge_p_max is None else edge_p_max)
    edge_enrichment_min = float(enrichment_min if edge_enrichment_min is None else edge_enrichment_min)
    delta_pmi_min = float(-np.inf if delta_pmi_min is None else delta_pmi_min)
    top_k_nodes = int(top_k_nodes); top_k_edges = int(top_k_edges)
    if not (0.0 <= p_min <= p_max <= 1.0):
        raise ValueError("Require 0 <= p_min <= p_max <= 1.")
    if not (0.0 <= edge_p_min <= edge_p_max <= 1.0):
        raise ValueError("Require 0 <= edge_p_min <= edge_p_max <= 1.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    selected_ids = [str(s).strip() for s in sample_ids if str(s).strip()]
    if len(selected_ids) < 2:
        raise ValueError("Select at least two samples for leave-one-out transient analysis.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]
    cluster_npz_path = cluster_dir / "cluster.npz"
    residue_labels, k_list, cluster_edges = _cluster_topology(cluster_npz_path)
    n_res = len(k_list)

    samples_meta = store.list_samples(project_id, system_id, cluster_id)
    sample_by_id = {str(s.get("sample_id")): s for s in samples_meta if isinstance(s, dict) and s.get("sample_id")}
    labels: list[np.ndarray] = []
    sample_labels: list[str] = []
    sample_types: list[str] = []
    sample_states: list[str] = []
    for idx, sid in enumerate(selected_ids):
        entry = sample_by_id.get(sid)
        if not entry:
            raise FileNotFoundError(f"Sample not found on this cluster: {sid}")
        typ = str(entry.get("type") or "sample")
        X = _load_labels(_resolve_sample_path(store, project_id, system_id, cluster_dir, entry), md_label_mode=md_label_mode, drop_invalid=drop_invalid, sample_type=typ)
        if X.shape[1] != n_res:
            raise ValueError(f"Sample {sid} has N={X.shape[1]} labels, expected {n_res}.")
        if np.any(X < 0):
            raise ValueError(f"Sample {sid} contains negative labels; use assigned labels or remap unassigned frames.")
        for i, k in enumerate(k_list):
            if X[:, i].size and int(np.max(X[:, i])) >= int(k):
                raise ValueError(f"Sample {sid} has out-of-range labels at residue {i}.")
        labels.append(X)
        sample_labels.append(str(entry.get("name") or sid))
        sample_types.append(typ)
        sample_states.append(str(entry.get("state_id") or ""))
        if progress_callback:
            progress_callback("Loaded samples", idx + 1, len(selected_ids))

    n_samples = len(labels)
    frame_counts = np.asarray([int(x.shape[0]) for x in labels], dtype=np.int64)
    if np.any(frame_counts <= 0):
        raise ValueError("All selected samples must contain at least one frame.")
    total_frames = int(np.sum(frame_counts))

    node_counts = [[np.bincount(X[:, i], minlength=int(k_list[i])).astype(np.int64) for i in range(n_res)] for X in labels]
    node_total = [np.sum([node_counts[m][i] for m in range(n_samples)], axis=0).astype(np.int64) for i in range(n_res)]

    node_rows: list[dict[str, Any]] = []
    for m, X in enumerate(labels):
        bg_frames = max(1, total_frames - int(frame_counts[m]))
        for i in range(n_res):
            counts = node_counts[m][i]
            total = node_total[i]
            for k, count in enumerate(counts.tolist()):
                if count <= 0:
                    continue
                p = float(count) / float(frame_counts[m])
                if p < p_min or p > p_max:
                    continue
                bg = float(int(total[k]) - int(count)) / float(bg_frames)
                log2_enr = _safe_log2_ratio(p, bg, epsilon)
                if log2_enr <= enrichment_min:
                    continue
                episodes, mean_dwell, max_dwell = _episodes(X[:, i] == int(k))
                score = float(log2_enr * np.sqrt(float(count)))
                node_rows.append({
                    "sample_index": m, "residue_index": i, "cluster": int(k), "occupancy": p,
                    "background": bg, "log2_enrichment": log2_enr, "count": int(count),
                    "episodes": episodes, "mean_dwell": mean_dwell, "max_dwell": max_dwell, "score": score,
                })
        if progress_callback:
            progress_callback("Computed node transient states", m + 1, n_samples)
    node_rows.sort(key=lambda r: (float(r["score"]), float(r["log2_enrichment"])), reverse=True)
    if top_k_nodes > 0:
        node_rows = node_rows[:top_k_nodes]

    if edge_mode == "all_vs_all":
        edges = [(i, j) for i in range(n_res) for j in range(i + 1, n_res)]
    else:
        edges = list(cluster_edges)
    edge_rows: list[dict[str, Any]] = []
    if include_edges and edges:
        for epos, (i, j) in enumerate(edges):
            Ki = int(k_list[i]); Kj = int(k_list[j])
            per_counts: list[np.ndarray] = []
            for X in labels:
                joint = np.bincount((X[:, i].astype(np.int64) * Kj + X[:, j].astype(np.int64)), minlength=Ki * Kj).reshape(Ki, Kj)
                per_counts.append(joint.astype(np.int64))
            total_joint = np.sum(per_counts, axis=0).astype(np.int64)
            for m, X in enumerate(labels):
                bg_frames = max(1, total_frames - int(frame_counts[m]))
                counts = per_counts[m]
                nz = np.argwhere(counts > 0)
                for k, l in nz.tolist():
                    count = int(counts[k, l])
                    p = float(count) / float(frame_counts[m])
                    if p < edge_p_min or p > edge_p_max:
                        continue
                    bg = float(int(total_joint[k, l]) - count) / float(bg_frames)
                    log2_enr = _safe_log2_ratio(p, bg, epsilon)
                    if log2_enr <= edge_enrichment_min:
                        continue
                    pi = float(node_counts[m][i][k]) / float(frame_counts[m])
                    pj = float(node_counts[m][j][l]) / float(frame_counts[m])
                    bg_pi = float(int(node_total[i][k]) - int(node_counts[m][i][k])) / float(bg_frames)
                    bg_pj = float(int(node_total[j][l]) - int(node_counts[m][j][l])) / float(bg_frames)
                    pmi = _safe_pmi(p, pi, pj, epsilon)
                    bg_pmi = _safe_pmi(bg, bg_pi, bg_pj, epsilon)
                    delta_pmi = float(pmi - bg_pmi)
                    if delta_pmi <= delta_pmi_min:
                        continue
                    episodes, mean_dwell, max_dwell = _episodes((X[:, i] == int(k)) & (X[:, j] == int(l)))
                    score = float(log2_enr * np.sqrt(float(count)) * max(0.1, 1.0 + max(0.0, delta_pmi)))
                    edge_rows.append({
                        "sample_index": m, "edge_index": epos, "residue_i": i, "residue_j": j,
                        "cluster_i": int(k), "cluster_j": int(l), "occupancy": p, "background": bg,
                        "log2_enrichment": log2_enr, "count": count, "episodes": episodes,
                        "mean_dwell": mean_dwell, "max_dwell": max_dwell, "pmi": pmi,
                        "background_pmi": bg_pmi, "delta_pmi": delta_pmi, "score": score,
                    })
            if progress_callback and ((epos + 1) % max(1, len(edges) // 50) == 0 or epos + 1 == len(edges)):
                progress_callback("Computed edge transient states", epos + 1, len(edges))
    edge_rows.sort(key=lambda r: (float(r["score"]), float(r.get("delta_pmi", 0.0))), reverse=True)
    if top_k_edges > 0:
        edge_rows = edge_rows[:top_k_edges]

    key_payload = {
        "analysis_type": "transient_states",
        "sample_ids": selected_ids,
        "md_label_mode": md_label_mode,
        "drop_invalid": bool(drop_invalid),
        "p_min": p_min, "p_max": p_max, "enrichment_min": enrichment_min,
        "edge_mode": edge_mode, "edge_p_min": edge_p_min, "edge_p_max": edge_p_max,
        "edge_enrichment_min": edge_enrichment_min, "delta_pmi_min": None if not np.isfinite(delta_pmi_min) else delta_pmi_min,
    }
    analysis_id = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(key_payload, sort_keys=True)))
    root = _ensure_analysis_dir(cluster_dir, "transient_states")
    analysis_dir = root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME

    def arr(rows: list[dict[str, Any]], key: str, dtype: Any):
        return np.asarray([r.get(key, "") for r in rows], dtype=dtype)

    node_sample_idx = arr(node_rows, "sample_index", int)
    edge_sample_idx = arr(edge_rows, "sample_index", int)
    np.savez_compressed(
        npz_path,
        sample_ids=np.asarray(selected_ids, dtype=str),
        sample_labels=np.asarray(sample_labels, dtype=str),
        sample_types=np.asarray(sample_types, dtype=str),
        sample_state_ids=np.asarray(sample_states, dtype=str),
        frame_counts=np.asarray(frame_counts, dtype=np.int64),
        residue_labels=np.asarray(residue_labels, dtype=str),
        K_list=np.asarray(k_list, dtype=np.int32),
        edges=np.asarray(edges, dtype=np.int32),
        node_sample_index=node_sample_idx,
        node_sample_id=np.asarray([selected_ids[int(i)] for i in node_sample_idx], dtype=str),
        node_sample_label=np.asarray([sample_labels[int(i)] for i in node_sample_idx], dtype=str),
        node_residue_index=arr(node_rows, "residue_index", int),
        node_residue_label=np.asarray([residue_labels[int(r["residue_index"])] for r in node_rows], dtype=str),
        node_cluster=arr(node_rows, "cluster", int),
        node_occupancy=arr(node_rows, "occupancy", float),
        node_background=arr(node_rows, "background", float),
        node_log2_enrichment=arr(node_rows, "log2_enrichment", float),
        node_count=arr(node_rows, "count", int),
        node_episodes=arr(node_rows, "episodes", int),
        node_mean_dwell=arr(node_rows, "mean_dwell", float),
        node_max_dwell=arr(node_rows, "max_dwell", int),
        node_score=arr(node_rows, "score", float),
        edge_sample_index=edge_sample_idx,
        edge_sample_id=np.asarray([selected_ids[int(i)] for i in edge_sample_idx], dtype=str),
        edge_sample_label=np.asarray([sample_labels[int(i)] for i in edge_sample_idx], dtype=str),
        edge_index=arr(edge_rows, "edge_index", int),
        edge_residue_i=arr(edge_rows, "residue_i", int),
        edge_residue_j=arr(edge_rows, "residue_j", int),
        edge_label=np.asarray([f"{residue_labels[int(r['residue_i'])]}-{residue_labels[int(r['residue_j'])]}" for r in edge_rows], dtype=str),
        edge_cluster_i=arr(edge_rows, "cluster_i", int),
        edge_cluster_j=arr(edge_rows, "cluster_j", int),
        edge_occupancy=arr(edge_rows, "occupancy", float),
        edge_background=arr(edge_rows, "background", float),
        edge_log2_enrichment=arr(edge_rows, "log2_enrichment", float),
        edge_count=arr(edge_rows, "count", int),
        edge_episodes=arr(edge_rows, "episodes", int),
        edge_mean_dwell=arr(edge_rows, "mean_dwell", float),
        edge_max_dwell=arr(edge_rows, "max_dwell", int),
        edge_pmi=arr(edge_rows, "pmi", float),
        edge_background_pmi=arr(edge_rows, "background_pmi", float),
        edge_delta_pmi=arr(edge_rows, "delta_pmi", float),
        edge_score=arr(edge_rows, "score", float),
    )

    now = _utc_now()
    created_at = now
    if meta_path.exists():
        try:
            created_at = str(json.loads(meta_path.read_text(encoding="utf-8")).get("created_at") or now)
        except Exception:
            pass
    meta = {
        "analysis_id": analysis_id,
        "analysis_type": "transient_states",
        "created_at": created_at,
        "updated_at": now,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "sample_ids": selected_ids,
        "sample_labels": sample_labels,
        "md_label_mode": md_label_mode,
        "drop_invalid": bool(drop_invalid),
        "params": {
            "p_min": p_min,
            "p_max": p_max,
            "enrichment_min": enrichment_min,
            "epsilon": epsilon,
            "top_k_nodes": top_k_nodes,
            "include_edges": bool(include_edges),
            "edge_mode": edge_mode,
            "edge_p_min": edge_p_min,
            "edge_p_max": edge_p_max,
            "edge_enrichment_min": edge_enrichment_min,
            "delta_pmi_min": None if not np.isfinite(delta_pmi_min) else delta_pmi_min,
            "top_k_edges": top_k_edges,
        },
        "summary": {
            "n_samples": n_samples,
            "n_residues": n_res,
            "n_edges_considered": int(len(edges)),
            "n_node_hits": int(len(node_rows)),
            "n_edge_hits": int(len(edge_rows)),
        },
        "paths": {"analysis_npz": str(npz_path.relative_to(system_dir))},
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    return {"metadata": _convert_nan_to_none(meta), "analysis_npz": str(npz_path), "analysis_dir": str(analysis_dir)}
