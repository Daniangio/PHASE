from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
import shutil
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def _purge_matching_sampling_analyses(
    analysis_root: Path,
    *,
    model_id: str | None,
    model_name: str | None,
    md_label_mode: str,
    drop_invalid: bool,
) -> None:
    if not analysis_root.exists():
        return
    target_model_id = str(model_id or "").strip()
    target_model_name = str(model_name or "").strip()
    target_mode = str(md_label_mode or "assigned").strip().lower()
    target_drop_invalid = bool(drop_invalid)
    for analysis_dir in analysis_root.iterdir():
        if not analysis_dir.is_dir():
            continue
        meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta_mode = str(meta.get("md_label_mode") or "assigned").strip().lower()
        meta_drop_invalid = bool(meta.get("drop_invalid"))
        meta_model_id = str(meta.get("model_id") or "").strip()
        meta_model_name = str(meta.get("model_name") or "").strip()
        if meta_mode != target_mode or meta_drop_invalid != target_drop_invalid:
            continue
        if meta_model_id != target_model_id or meta_model_name != target_model_name:
            continue
        shutil.rmtree(analysis_dir, ignore_errors=True)


def _resolve_model_for_analysis(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_ref: str,
) -> tuple[PottsModel, str | None, str | None, Path]:
    model_path = Path(model_ref)
    model_id = None
    model_name = None
    if not model_path.suffix:
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
    return model, model_id, model_name, model_path


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


_LIGAND_COMPLETION_MODEL_CACHE: dict[str, PottsModel] = {}


def _load_gauged_model_from_path(path: str) -> PottsModel:
    key = str(Path(path).resolve())
    model = _LIGAND_COMPLETION_MODEL_CACHE.get(key)
    if model is None:
        model = zero_sum_gauge_model(load_potts_model(key))
        _LIGAND_COMPLETION_MODEL_CACHE[key] = model
    return model


def _sa_metropolis_trajectory(
    model: PottsModel,
    *,
    x0: np.ndarray,
    n_steps: int,
    beta_hot: float,
    beta_cold: float,
    seed: int,
    schedule: str = "geom",
) -> np.ndarray:
    """
    Lightweight SA trajectory on Potts labels with single-site Metropolis updates.
    """
    x = np.asarray(x0, dtype=np.int32).copy()
    n_steps = max(1, int(n_steps))
    beta_hot = float(beta_hot)
    beta_cold = float(beta_cold)
    if not np.isfinite(beta_hot) or not np.isfinite(beta_cold) or beta_hot <= 0 or beta_cold <= 0:
        raise ValueError("SA betas must be finite and > 0.")
    if beta_hot > beta_cold:
        beta_hot, beta_cold = beta_cold, beta_hot

    sched = str(schedule or "geom").strip().lower()
    if n_steps == 1:
        betas = np.asarray([beta_cold], dtype=float)
    elif sched == "lin":
        betas = np.linspace(beta_hot, beta_cold, num=n_steps, dtype=float)
    else:
        # Geometric spacing is the default (request: 0.8 -> 50).
        betas = np.geomspace(beta_hot, beta_cold, num=n_steps).astype(float)

    rng = np.random.default_rng(int(seed))
    neigh = model.neighbors()
    K_list = model.K_list()
    N = len(K_list)
    traj = np.zeros((n_steps, N), dtype=np.int32)

    for t, beta in enumerate(betas.tolist()):
        for r in range(N):
            Kr = int(K_list[r])
            if Kr <= 1:
                continue
            old = int(x[r])
            new = int(rng.integers(0, Kr - 1))
            if new >= old:
                new += 1

            dE = float(model.h[r][new] - model.h[r][old])
            for s in neigh[r]:
                xs = int(x[s])
                Jrs = model.coupling(r, s)
                dE += float(Jrs[new, xs] - Jrs[old, xs])

            if dE <= 0.0 or float(rng.random()) < float(np.exp(-beta * dE)):
                x[r] = np.int32(new)
        traj[t] = x
    return traj


def _build_adjusted_model_with_penalty(
    base_model: PottsModel,
    *,
    constrained_indices: Sequence[int],
    constrained_weights: Sequence[float],
    penalty_phi: Sequence[np.ndarray],
    lam: float,
) -> PottsModel:
    """
    Build E_cond(s) = E_base(s) + lam * sum_i w_i * phi_i(s_i) by adding to local fields.
    """
    h_adj = [np.asarray(hr, dtype=float).copy() for hr in base_model.h]
    if float(lam) != 0.0:
        for i, w, phi in zip(constrained_indices, constrained_weights, penalty_phi):
            idx = int(i)
            h_adj[idx] = h_adj[idx] + float(lam) * float(w) * np.asarray(phi, dtype=float)
    return PottsModel(h=h_adj, J=base_model.J, edges=base_model.edges)


def _subsample_tail_states(
    traj: np.ndarray,
    *,
    tail_steps: int,
    n_samples: int,
) -> np.ndarray:
    X = np.asarray(traj, dtype=np.int32)
    if X.ndim != 2 or X.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    tail_n = max(1, min(int(tail_steps), int(X.shape[0])))
    tail = X[-tail_n:]
    n_out = int(n_samples)
    if n_out <= 0:
        return tail
    if n_out == tail_n:
        return tail
    idx = np.linspace(0, tail_n - 1, num=n_out, dtype=np.int64)
    return np.asarray(tail[idx], dtype=np.int32)


def _normalized_auc(lambda_values: np.ndarray, success_values: np.ndarray) -> float:
    x = np.asarray(lambda_values, dtype=float).ravel()
    y = np.asarray(success_values, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size == 1:
        return float(y[0]) if np.isfinite(y[0]) else float("nan")
    area = float(np.trapz(y, x))
    span = float(x[-1] - x[0])
    if not np.isfinite(span) or span <= 0:
        return float(np.mean(y))
    return area / span


def _completion_cost_from_curve(
    lambda_values: np.ndarray,
    success_values: np.ndarray,
    *,
    target_success: float,
    unreached_value: float,
) -> float:
    x = np.asarray(lambda_values, dtype=float).ravel()
    y = np.asarray(success_values, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    target = float(target_success)
    idx = np.where(y >= target)[0]
    if idx.size == 0:
        return float(unreached_value)
    k = int(idx[0])
    if k <= 0:
        return float(x[0])
    x0, x1 = float(x[k - 1]), float(x[k])
    y0, y1 = float(y[k - 1]), float(y[k])
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) <= 1e-12:
        return float(x1)
    frac = (target - y0) / (y1 - y0)
    frac = max(0.0, min(1.0, float(frac)))
    return float(x0 + frac * (x1 - x0))


def _mean_js_to_reference(
    p_batch: Sequence[np.ndarray],
    p_ref_padded: np.ndarray,
    K_list: Sequence[int],
) -> float:
    vals: list[float] = []
    for i, p_i in enumerate(p_batch):
        Ki = int(K_list[i])
        p_ref = np.asarray(p_ref_padded[i, :Ki], dtype=float)
        p_cur = np.asarray(p_i, dtype=float)
        vals.append(float(js_divergence(p_cur, p_ref)))
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=float)))


def _weighted_mean_with_weights(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return float("nan")
    ws = float(np.sum(w[good]))
    if ws <= 0.0:
        return float("nan")
    return float(np.sum(v[good] * w[good]) / ws)


def _normalize_js_filter_rules(rules_raw: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    if not isinstance(rules_raw, list):
        return out
    for item in rules_raw:
        if not isinstance(item, dict):
            continue
        try:
            a_min = float(item.get("aMin", 0.0))
            a_max = float(item.get("aMax", 1.0))
            b_min = float(item.get("bMin", 0.0))
            b_max = float(item.get("bMax", 1.0))
        except Exception:
            continue
        if not np.isfinite(a_min):
            a_min = 0.0
        if not np.isfinite(a_max):
            a_max = 1.0
        if not np.isfinite(b_min):
            b_min = 0.0
        if not np.isfinite(b_max):
            b_max = 1.0
        if a_max < a_min:
            a_min, a_max = a_max, a_min
        if b_max < b_min:
            b_min, b_max = b_max, b_min
        out.append({"aMin": a_min, "aMax": a_max, "bMin": b_min, "bMax": b_max})
    return out


def _passes_js_filter_rules(d_a: float, d_b: float, rules: Sequence[dict[str, float]]) -> bool:
    if not np.isfinite(d_a) or not np.isfinite(d_b):
        return False
    if not rules:
        return True
    for rule in rules:
        if (
            d_a >= float(rule.get("aMin", 0.0))
            and d_a <= float(rule.get("aMax", 1.0))
            and d_b >= float(rule.get("bMin", 0.0))
            and d_b <= float(rule.get("bMax", 1.0))
        ):
            return True
    return False


def _delta_js_row_node_edge_values(
    *,
    row: int,
    N: int,
    js_node_a: np.ndarray,
    js_node_b: np.ndarray,
    js_edge_a: np.ndarray,
    js_edge_b: np.ndarray,
    edges_all: np.ndarray,
    top_edge_indices: np.ndarray,
    D_edge: np.ndarray,
    edge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if row < 0 or row >= js_node_a.shape[0] or js_node_a.shape[1] != N or js_node_b.shape != js_node_a.shape:
        raise ValueError("delta_js row or js_node dimensions are invalid.")
    d_a = np.asarray(js_node_a[row], dtype=float).copy()
    d_b = np.asarray(js_node_b[row], dtype=float).copy()
    alpha = float(max(0.0, min(1.0, edge_alpha)))
    if (
        alpha <= 0.0
        or edges_all.size == 0
        or top_edge_indices.size == 0
        or js_edge_a.ndim != 2
        or js_edge_b.ndim != 2
        or js_edge_a.shape != js_edge_b.shape
        or row >= js_edge_a.shape[0]
    ):
        return d_a, d_b
    sum_w = np.zeros((N,), dtype=float)
    sum_a = np.zeros((N,), dtype=float)
    sum_b = np.zeros((N,), dtype=float)
    n_cols = min(int(top_edge_indices.size), int(js_edge_a.shape[1]))
    for col in range(n_cols):
        eidx = int(top_edge_indices[col])
        if eidx < 0 or eidx >= edges_all.shape[0] or eidx >= D_edge.shape[0]:
            continue
        r, s = int(edges_all[eidx, 0]), int(edges_all[eidx, 1])
        if r < 0 or s < 0 or r >= N or s >= N:
            continue
        v_a = float(js_edge_a[row, col])
        v_b = float(js_edge_b[row, col])
        if not np.isfinite(v_a) or not np.isfinite(v_b):
            continue
        wr = float(D_edge[eidx])
        w = max(0.0, wr) if np.isfinite(wr) else 0.0
        if w <= 0.0:
            w = 1.0
        sum_w[r] += w
        sum_w[s] += w
        sum_a[r] += w * v_a
        sum_a[s] += w * v_a
        sum_b[r] += w * v_b
        sum_b[s] += w * v_b
    edge_a = np.where(sum_w > 0.0, sum_a / np.maximum(sum_w, 1e-12), d_a)
    edge_b = np.where(sum_w > 0.0, sum_b / np.maximum(sum_w, 1e-12), d_b)
    d_a = (1.0 - alpha) * d_a + alpha * edge_a
    d_b = (1.0 - alpha) * d_b + alpha * edge_b
    return d_a, d_b


def _compute_delta_js_edge_weighted_scores(
    *,
    tail: np.ndarray,
    p_tail: Sequence[np.ndarray],
    p_ref_a_padded: np.ndarray,
    p_ref_b_padded: np.ndarray,
    K_list: Sequence[int],
    residue_indices: Sequence[int],
    residue_weights: Sequence[float],
    edges: Sequence[Tuple[int, int]],
    edge_weights: Sequence[float],
    ref_edge_a: Sequence[np.ndarray],
    ref_edge_b: Sequence[np.ndarray],
    node_edge_alpha: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute Delta-JS style weighted node/edge scores on a tail sample.

    Returns:
        (mixed_a, mixed_b, node_a, node_b, edge_a, edge_b)
    """
    residues = [int(i) for i in residue_indices]
    if not residues:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    r_weights = np.asarray(residue_weights, dtype=float).ravel()
    if r_weights.shape[0] != len(residues):
        raise ValueError("residue_weights size mismatch in delta-js success evaluation.")

    node_js_a = np.zeros((len(residues),), dtype=float)
    node_js_b = np.zeros((len(residues),), dtype=float)
    for ridx, i in enumerate(residues):
        Ki = int(K_list[i])
        p_ref_a = np.asarray(p_ref_a_padded[i, :Ki], dtype=float)
        p_ref_b = np.asarray(p_ref_b_padded[i, :Ki], dtype=float)
        p_cur = np.asarray(p_tail[i], dtype=float)
        node_js_a[ridx] = float(js_divergence(p_cur, p_ref_a))
        node_js_b[ridx] = float(js_divergence(p_cur, p_ref_b))
    node_score_a = _weighted_mean_with_weights(node_js_a, r_weights)
    node_score_b = _weighted_mean_with_weights(node_js_b, r_weights)

    edge_list = [(int(r), int(s)) for (r, s) in edges]
    if (
        not edge_list
        or len(ref_edge_a) != len(edge_list)
        or len(ref_edge_b) != len(edge_list)
        or len(edge_weights) != len(edge_list)
    ):
        edge_score_a = node_score_a
        edge_score_b = node_score_b
        alpha_eff = 0.0
    else:
        e_weights = np.asarray(edge_weights, dtype=float).ravel()
        p2_tail = pairwise_joints_on_edges(np.asarray(tail, dtype=np.int32), K_list, edge_list)
        edge_js_a = np.zeros((len(edge_list),), dtype=float)
        edge_js_b = np.zeros((len(edge_list),), dtype=float)
        for eidx, e in enumerate(edge_list):
            p_tail_e = np.asarray(p2_tail[e], dtype=float).ravel()
            p_ref_e_a = np.asarray(ref_edge_a[eidx], dtype=float).ravel()
            p_ref_e_b = np.asarray(ref_edge_b[eidx], dtype=float).ravel()
            edge_js_a[eidx] = float(js_divergence(p_tail_e, p_ref_e_a))
            edge_js_b[eidx] = float(js_divergence(p_tail_e, p_ref_e_b))
        edge_score_a = _weighted_mean_with_weights(edge_js_a, e_weights)
        edge_score_b = _weighted_mean_with_weights(edge_js_b, e_weights)
        alpha_eff = float(max(0.0, min(1.0, node_edge_alpha)))

    mixed_a = (1.0 - alpha_eff) * float(node_score_a) + alpha_eff * float(edge_score_a)
    mixed_b = (1.0 - alpha_eff) * float(node_score_b) + alpha_eff * float(edge_score_b)
    return (
        float(mixed_a),
        float(mixed_b),
        float(node_score_a),
        float(node_score_b),
        float(edge_score_a),
        float(edge_score_b),
    )


def _single_frame_ligand_completion_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Worker for one starting frame in ligand-conditional completion analysis.
    """
    model_a = _load_gauged_model_from_path(str(payload["model_a_path"]))
    model_b = _load_gauged_model_from_path(str(payload["model_b_path"]))
    x0 = np.asarray(payload["x0"], dtype=np.int32).ravel()
    K_list = [int(v) for v in payload["K_list"]]
    lambdas = np.asarray(payload["lambda_values"], dtype=float).ravel()
    constrained_indices = [int(i) for i in payload["constrained_indices"]]
    constrained_weights = [float(w) for w in payload["constrained_weights"]]
    penalty_phi = [np.asarray(v, dtype=float).ravel() for v in payload["penalty_phi"]]

    sampler = str(payload.get("sampler") or "sa").strip().lower()
    n_steps = int(payload["n_steps"])
    tail_steps = int(payload["tail_steps"])
    n_samples = int(payload["n_samples_per_frame"])
    gibbs_beta = float(payload.get("gibbs_beta", 1.0))
    sa_beta_hot = float(payload.get("sa_beta_hot", 0.8))
    sa_beta_cold = float(payload.get("sa_beta_cold", 50.0))
    sa_schedule = str(payload.get("sa_schedule") or "geom")
    deltae_margin = float(payload.get("deltae_margin", 0.0))
    success_metric_mode = str(payload.get("success_metric_mode") or "deltae").strip().lower()
    js_success_threshold = float(payload.get("js_success_threshold", 0.15))
    js_success_margin = float(payload.get("js_success_margin", 0.02))
    p_ref_a = np.asarray(payload["p_ref_a"], dtype=float)
    p_ref_b = np.asarray(payload["p_ref_b"], dtype=float)
    delta_js_eval_spec = payload.get("delta_js_eval_spec") or {}
    rng_seed = int(payload["seed"])

    if sampler not in {"sa", "gibbs"}:
        raise ValueError("sampler must be 'sa' or 'gibbs'.")
    if success_metric_mode not in {"deltae", "delta_js_edge"}:
        raise ValueError("success_metric_mode must be one of: deltae, delta_js_edge.")

    djs_residue_indices = np.asarray(delta_js_eval_spec.get("residue_indices", []), dtype=np.int32).ravel()
    djs_residue_weights = np.asarray(delta_js_eval_spec.get("residue_weights", []), dtype=float).ravel()
    djs_edges_raw = np.asarray(delta_js_eval_spec.get("edges", []), dtype=np.int32)
    djs_edge_weights = np.asarray(delta_js_eval_spec.get("edge_weights", []), dtype=float).ravel()
    djs_ref_edge_a = [np.asarray(v, dtype=float) for v in (delta_js_eval_spec.get("ref_edge_a") or [])]
    djs_ref_edge_b = [np.asarray(v, dtype=float) for v in (delta_js_eval_spec.get("ref_edge_b") or [])]
    djs_alpha = float(delta_js_eval_spec.get("node_edge_alpha", 0.5))

    if success_metric_mode == "delta_js_edge":
        if djs_residue_indices.size == 0:
            raise ValueError("delta_js_edge success mode requires at least one discriminative residue.")
        if djs_residue_weights.size != djs_residue_indices.size:
            raise ValueError("delta_js_edge residue weights mismatch.")
        if djs_edges_raw.size == 0:
            djs_edges: list[Tuple[int, int]] = []
        else:
            djs_edges_raw = np.asarray(djs_edges_raw, dtype=np.int32).reshape(-1, 2)
            djs_edges = [(int(r), int(s)) for (r, s) in djs_edges_raw.tolist()]
        if djs_edges and djs_edge_weights.size != len(djs_edges):
            raise ValueError("delta_js_edge edge weights mismatch.")
        if djs_edges and (len(djs_ref_edge_a) != len(djs_edges) or len(djs_ref_edge_b) != len(djs_edges)):
            raise ValueError("delta_js_edge edge reference distributions mismatch.")
    else:
        djs_edges = []

    L = int(lambdas.shape[0])
    success_a = np.zeros((L,), dtype=np.float32)
    success_b = np.zeros((L,), dtype=np.float32)
    js_a_under_a = np.zeros((L,), dtype=np.float32)
    js_b_under_a = np.zeros((L,), dtype=np.float32)
    js_a_under_b = np.zeros((L,), dtype=np.float32)
    js_b_under_b = np.zeros((L,), dtype=np.float32)
    novelty_under_a = np.zeros((L,), dtype=np.float32)
    novelty_under_b = np.zeros((L,), dtype=np.float32)
    deltae_mean_under_a = np.zeros((L,), dtype=np.float32)
    deltae_mean_under_b = np.zeros((L,), dtype=np.float32)
    success_js_eval_under_a = np.zeros((L,), dtype=np.float32)
    success_js_eval_under_b = np.zeros((L,), dtype=np.float32)
    success_js_eval_node_under_a = np.zeros((L,), dtype=np.float32)
    success_js_eval_node_under_b = np.zeros((L,), dtype=np.float32)
    success_js_eval_edge_under_a = np.zeros((L,), dtype=np.float32)
    success_js_eval_edge_under_b = np.zeros((L,), dtype=np.float32)

    for li, lam in enumerate(lambdas.tolist()):
        for endpoint in (0, 1):
            base_model = model_a if endpoint == 0 else model_b
            cond_model = _build_adjusted_model_with_penalty(
                base_model,
                constrained_indices=constrained_indices,
                constrained_weights=constrained_weights,
                penalty_phi=penalty_phi,
                lam=float(lam),
            )
            local_seed = int(rng_seed + li * 17 + endpoint * 10007)
            if sampler == "gibbs":
                traj = gibbs_sample_potts(
                    cond_model,
                    beta=float(gibbs_beta),
                    n_samples=int(n_steps),
                    burn_in=0,
                    thinning=1,
                    seed=local_seed,
                    x0=x0,
                    progress=False,
                )
            else:
                traj = _sa_metropolis_trajectory(
                    cond_model,
                    x0=x0,
                    n_steps=int(n_steps),
                    beta_hot=float(sa_beta_hot),
                    beta_cold=float(sa_beta_cold),
                    seed=local_seed,
                    schedule=sa_schedule,
                )

            tail = _subsample_tail_states(traj, tail_steps=int(tail_steps), n_samples=int(n_samples))
            if tail.ndim != 2 or tail.shape[0] == 0:
                continue

            dE = model_b.energy_batch(tail) - model_a.energy_batch(tail)
            p_tail = marginals(tail, K_list)
            js_a = _mean_js_to_reference(p_tail, p_ref_a, K_list)
            js_b = _mean_js_to_reference(p_tail, p_ref_b, K_list)
            novelty = float(min(js_a, js_b))
            djs_mixed_a = float("nan")
            djs_mixed_b = float("nan")
            djs_node_a = float("nan")
            djs_node_b = float("nan")
            djs_edge_a = float("nan")
            djs_edge_b = float("nan")
            if success_metric_mode == "delta_js_edge":
                (
                    djs_mixed_a,
                    djs_mixed_b,
                    djs_node_a,
                    djs_node_b,
                    djs_edge_a,
                    djs_edge_b,
                ) = _compute_delta_js_edge_weighted_scores(
                    tail=tail,
                    p_tail=p_tail,
                    p_ref_a_padded=p_ref_a,
                    p_ref_b_padded=p_ref_b,
                    K_list=K_list,
                    residue_indices=djs_residue_indices.tolist(),
                    residue_weights=djs_residue_weights.tolist(),
                    edges=djs_edges,
                    edge_weights=djs_edge_weights.tolist(),
                    ref_edge_a=djs_ref_edge_a,
                    ref_edge_b=djs_ref_edge_b,
                    node_edge_alpha=djs_alpha,
                )

            if endpoint == 0:
                if success_metric_mode == "deltae":
                    success_a[li] = np.float32(np.mean(dE > float(deltae_margin)))
                else:
                    ok = (
                        np.isfinite(djs_mixed_a)
                        and np.isfinite(djs_mixed_b)
                        and (float(djs_mixed_a) <= float(js_success_threshold))
                        and (float(djs_mixed_a) + float(js_success_margin) <= float(djs_mixed_b))
                    )
                    success_a[li] = np.float32(1.0 if ok else 0.0)
                js_a_under_a[li] = np.float32(js_a)
                js_b_under_a[li] = np.float32(js_b)
                novelty_under_a[li] = np.float32(novelty)
                deltae_mean_under_a[li] = np.float32(np.mean(dE))
                success_js_eval_under_a[li] = np.float32(djs_mixed_a)
                success_js_eval_node_under_a[li] = np.float32(djs_node_a)
                success_js_eval_edge_under_a[li] = np.float32(djs_edge_a)
            else:
                if success_metric_mode == "deltae":
                    success_b[li] = np.float32(np.mean(dE < -float(deltae_margin)))
                else:
                    ok = (
                        np.isfinite(djs_mixed_a)
                        and np.isfinite(djs_mixed_b)
                        and (float(djs_mixed_b) <= float(js_success_threshold))
                        and (float(djs_mixed_b) + float(js_success_margin) <= float(djs_mixed_a))
                    )
                    success_b[li] = np.float32(1.0 if ok else 0.0)
                js_a_under_b[li] = np.float32(js_a)
                js_b_under_b[li] = np.float32(js_b)
                novelty_under_b[li] = np.float32(novelty)
                deltae_mean_under_b[li] = np.float32(np.mean(dE))
                success_js_eval_under_b[li] = np.float32(djs_mixed_b)
                success_js_eval_node_under_b[li] = np.float32(djs_node_b)
                success_js_eval_edge_under_b[li] = np.float32(djs_edge_b)

    raw_deltae = float(model_b.energy(x0) - model_a.energy(x0))
    raw_js_a_vals: list[float] = []
    raw_js_b_vals: list[float] = []
    for i, state in enumerate(x0.tolist()):
        Ki = int(K_list[i])
        p0 = np.zeros((Ki,), dtype=float)
        if 0 <= int(state) < Ki:
            p0[int(state)] = 1.0
        else:
            p0[:] = 1.0 / max(1, Ki)
        raw_js_a_vals.append(float(js_divergence(p0, np.asarray(p_ref_a[i, :Ki], dtype=float))))
        raw_js_b_vals.append(float(js_divergence(p0, np.asarray(p_ref_b[i, :Ki], dtype=float))))

    return {
        "success_a": success_a,
        "success_b": success_b,
        "js_a_under_a": js_a_under_a,
        "js_b_under_a": js_b_under_a,
        "js_a_under_b": js_a_under_b,
        "js_b_under_b": js_b_under_b,
        "novelty_under_a": novelty_under_a,
        "novelty_under_b": novelty_under_b,
        "deltae_mean_under_a": deltae_mean_under_a,
        "deltae_mean_under_b": deltae_mean_under_b,
        "success_js_eval_under_a": success_js_eval_under_a,
        "success_js_eval_under_b": success_js_eval_under_b,
        "success_js_eval_node_under_a": success_js_eval_node_under_a,
        "success_js_eval_node_under_b": success_js_eval_node_under_b,
        "success_js_eval_edge_under_a": success_js_eval_edge_under_a,
        "success_js_eval_edge_under_b": success_js_eval_edge_under_b,
        "raw_deltae": np.float32(raw_deltae),
        "raw_js_a": np.float32(np.mean(np.asarray(raw_js_a_vals, dtype=float))) if raw_js_a_vals else np.float32(np.nan),
        "raw_js_b": np.float32(np.mean(np.asarray(raw_js_b_vals, dtype=float))) if raw_js_b_vals else np.float32(np.nan),
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
    X = np.asarray(X, dtype=int)
    if X.ndim != 2:
        raise ValueError(f"Potts energy labels must have shape (frames, residues); got {X.shape}.")
    if X.shape[1] != len(model.h):
        raise ValueError(
            f"Potts energy labels contain {X.shape[1]} residues, but the model contains {len(model.h)}."
        )

    # Persist the decomposition used to obtain the total. This keeps analysis
    # generation independent from later residue-selection visualization.
    node_energies = np.column_stack(
        [np.asarray(field)[X[:, residue]] for residue, field in enumerate(model.h)]
    ) if model.h else np.zeros((X.shape[0], 0), dtype=float)
    edge_index = np.asarray(model.edges, dtype=np.int32).reshape((-1, 2))
    edge_energies = np.column_stack(
        [model.J[(int(r), int(s))][X[:, int(r)], X[:, int(s)]] for r, s in model.edges]
    ) if model.edges else np.zeros((X.shape[0], 0), dtype=float)
    energies = np.sum(node_energies, axis=1) + np.sum(edge_energies, axis=1)
    if energies is None or energies.size == 0:
        return {
            "energies": np.asarray([], dtype=float),
            "node_energies": node_energies,
            "edge_energies": edge_energies,
            "edge_index": edge_index,
            "energy_mean": None,
            "energy_median": None,
            "energy_min": None,
            "energy_max": None,
        }
    payload: Dict[str, Any] = {
        "energies": energies,
        "node_energies": node_energies,
        "edge_energies": edge_energies,
        "edge_index": edge_index,
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
    model_refs: Sequence[str] | None = None,
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    n_workers: int | None = None,
    analysis_edge_mode: str | None = None,
    analysis_contact_cutoff: float | None = None,
    analysis_contact_atom_mode: str | None = None,
    analysis_contact_state_ids: Sequence[str] | None = None,
    analysis_contact_pdbs: Sequence[str] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Dict[str, Any]:
    """
    Compute Sampling Explorer analyses for one cluster:
      - MD-vs-sample distribution metrics (node JS + edge JS) for all MD/non-MD pairs
      - optional per-sample energies under a selected Potts model
    """
    from phase.potts.orchestration import run_potts_analysis_local

    return run_potts_analysis_local(
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        model_ref=model_ref,
        model_refs=model_refs,
        md_label_mode=md_label_mode,
        drop_invalid=drop_invalid,
        n_workers=n_workers,
        analysis_edge_mode=analysis_edge_mode,
        analysis_contact_cutoff=analysis_contact_cutoff,
        analysis_contact_atom_mode=analysis_contact_atom_mode,
        analysis_contact_state_ids=analysis_contact_state_ids,
        analysis_contact_pdbs=analysis_contact_pdbs,
        progress_callback=progress_callback,
    ).get("metadata", {})


def compute_lambda_sweep_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str,
    model_b_ref: str,
    lambda_sample_ids: Sequence[str],
    lambdas: Sequence[float],
    reference_sample_ids: Sequence[str] | None = None,
    ref_md_sample_ids: Sequence[str] | None = None,
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
      - Node/edge JS divergence vs flexible reference/comparison ensembles, as curves vs λ
      - Combined match curves D(λ) to each comparison reference: α*JS_node_mean + (1-α)*JS_edge_mean

    Returns a dict of arrays ready to be persisted into analysis.npz plus metadata helpers.
    """
    if len(lambda_sample_ids) != len(lambdas):
        raise ValueError("lambda_sample_ids and lambdas must have the same length.")
    reference_sample_ids = [str(v).strip() for v in (reference_sample_ids or ref_md_sample_ids or []) if str(v).strip()]
    if len(reference_sample_ids) < 3:
        raise ValueError("reference_sample_ids must contain at least 3 sample ids (A, B, and one comparison).")
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
    ref_marginals: list[list[np.ndarray]] = []
    ref_p2_flat: list[np.ndarray] = []
    ref_names: list[str] = []

    for sid in reference_sample_ids:
        entry = next((s for s in samples if s.get("sample_id") == sid), None)
        if not entry:
            raise FileNotFoundError(f"Reference sample not found: {sid}")
        ref_entries.append(entry)
        ref_names.append(str(entry.get("name") or sid))
        X_ref = _load_labels(entry, md_mode=str(entry.get("type") or "") == "md_eval")
        if X_ref.ndim != 2 or X_ref.size == 0:
            raise ValueError(f"Reference sample is empty: {sid}")
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
    n_ref = len(reference_sample_ids)

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

    comparison_ref_indices = np.asarray(list(range(2, n_ref)), dtype=int)
    lambda_star_index_by_reference = np.full((comparison_ref_indices.shape[0],), -1, dtype=int)
    lambda_star_by_reference = np.full((comparison_ref_indices.shape[0],), np.nan, dtype=float)
    match_min_by_reference = np.full((comparison_ref_indices.shape[0],), np.nan, dtype=float)
    for out_idx, ref_idx in enumerate(comparison_ref_indices.tolist()):
        match_curve = combined[ref_idx]
        finite_mask = np.isfinite(match_curve)
        if finite_mask.any():
            best_idx = int(np.nanargmin(match_curve))
            lambda_star_index_by_reference[out_idx] = best_idx
            lambda_star_by_reference[out_idx] = float(lambdas_sorted[best_idx])
            match_min_by_reference[out_idx] = float(match_curve[best_idx])

    match_ref_index = int(comparison_ref_indices[0]) if comparison_ref_indices.size else -1
    if comparison_ref_indices.size:
        best_idx = int(lambda_star_index_by_reference[0])
        best_lambda = float(lambda_star_by_reference[0])
        best_value = float(match_min_by_reference[0])
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
        "reference_sample_ids": list(reference_sample_ids),
        "reference_sample_names": ref_names,
        "comparison_sample_ids": list(reference_sample_ids[2:]),
        "comparison_sample_names": ref_names[2:],
        "ref_md_sample_ids": list(reference_sample_ids),
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
        "comparison_ref_indices": np.asarray(comparison_ref_indices, dtype=int),
        "lambda_star_index": int(best_idx),
        "lambda_star": float(best_lambda),
        "match_min": float(best_value),
        "lambda_star_index_by_reference": np.asarray(lambda_star_index_by_reference, dtype=int),
        "lambda_star_by_reference": np.asarray(lambda_star_by_reference, dtype=float),
        "match_min_by_reference": np.asarray(match_min_by_reference, dtype=float),
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
    Backward-compatible local entry point.

    The Gibbs-relaxation implementation now lives in `phase.potts.orchestration`
    and uses explicit preparation / worker / aggregation phases.
    """
    from phase.potts.orchestration import run_gibbs_relaxation_local

    return run_gibbs_relaxation_local(
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        start_sample_id=start_sample_id,
        model_ref=model_ref,
        beta=beta,
        n_start_frames=n_start_frames,
        gibbs_sweeps=gibbs_sweeps,
        seed=seed,
        start_label_mode=start_label_mode,
        drop_invalid=drop_invalid,
        n_workers=n_workers,
        progress_callback=progress_callback,
    )


def run_ligand_completion_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str,
    model_b_ref: str,
    md_sample_id: str,
    constrained_residues: Sequence[str | int],
    reference_sample_id_a: str | None = None,
    reference_sample_id_b: str | None = None,
    sampler: str = "sa",
    lambda_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    n_start_frames: int = 100,
    n_samples_per_frame: int = 100,
    n_steps: int = 1000,
    tail_steps: int = 200,
    target_window_size: int = 11,
    target_pseudocount: float = 1e-3,
    epsilon_logpenalty: float = 1e-8,
    constraint_weight_mode: str = "uniform",
    constraint_weights: Sequence[float] | None = None,
    constraint_weight_min: float = 0.0,
    constraint_weight_max: float = 1.0,
    constraint_source_mode: str = "manual",
    constraint_delta_js_analysis_id: str | None = None,
    constraint_delta_js_sample_id: str | None = None,
    constraint_auto_top_k: int = 12,
    constraint_auto_edge_alpha: float = 0.3,
    constraint_auto_exclude_success: bool = True,
    gibbs_beta: float = 1.0,
    sa_beta_hot: float = 0.8,
    sa_beta_cold: float = 50.0,
    sa_schedule: str = "geom",
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    success_metric_mode: str = "deltae",
    delta_js_experiment_id: str | None = None,
    delta_js_analysis_id: str | None = None,
    delta_js_filter_setup_id: str | None = None,
    delta_js_filter_edge_alpha: float = 0.75,
    delta_js_d_residue_min: float = 0.0,
    delta_js_d_residue_max: float | None = None,
    delta_js_d_edge_min: float = 0.0,
    delta_js_d_edge_max: float | None = None,
    delta_js_node_edge_alpha: float | None = None,
    js_success_threshold: float = 0.15,
    js_success_margin: float = 0.02,
    deltae_margin: float = 0.0,
    completion_target_success: float = 0.7,
    completion_cost_if_unreached: float | None = None,
    n_workers: int = 1,
    seed: int = 0,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict[str, Any]:
    """
    Backward-compatible local entry point.

    The ligand-completion implementation now lives in `phase.potts.orchestration`
    and uses explicit preparation / worker / aggregation phases.
    """
    from phase.potts.orchestration import run_ligand_completion_local

    return run_ligand_completion_local(
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        model_a_ref=model_a_ref,
        model_b_ref=model_b_ref,
        md_sample_id=md_sample_id,
        constrained_residues=constrained_residues,
        reference_sample_id_a=reference_sample_id_a,
        reference_sample_id_b=reference_sample_id_b,
        sampler=sampler,
        lambda_values=lambda_values,
        n_start_frames=n_start_frames,
        n_samples_per_frame=n_samples_per_frame,
        n_steps=n_steps,
        tail_steps=tail_steps,
        target_window_size=target_window_size,
        target_pseudocount=target_pseudocount,
        epsilon_logpenalty=epsilon_logpenalty,
        constraint_weight_mode=constraint_weight_mode,
        constraint_weights=constraint_weights,
        constraint_weight_min=constraint_weight_min,
        constraint_weight_max=constraint_weight_max,
        constraint_source_mode=constraint_source_mode,
        constraint_delta_js_analysis_id=constraint_delta_js_analysis_id,
        constraint_delta_js_sample_id=constraint_delta_js_sample_id,
        constraint_auto_top_k=constraint_auto_top_k,
        constraint_auto_edge_alpha=constraint_auto_edge_alpha,
        constraint_auto_exclude_success=constraint_auto_exclude_success,
        gibbs_beta=gibbs_beta,
        sa_beta_hot=sa_beta_hot,
        sa_beta_cold=sa_beta_cold,
        sa_schedule=sa_schedule,
        md_label_mode=md_label_mode,
        drop_invalid=drop_invalid,
        success_metric_mode=success_metric_mode,
        delta_js_experiment_id=delta_js_experiment_id,
        delta_js_analysis_id=delta_js_analysis_id,
        delta_js_filter_setup_id=delta_js_filter_setup_id,
        delta_js_filter_edge_alpha=delta_js_filter_edge_alpha,
        delta_js_d_residue_min=delta_js_d_residue_min,
        delta_js_d_residue_max=delta_js_d_residue_max,
        delta_js_d_edge_min=delta_js_d_edge_min,
        delta_js_d_edge_max=delta_js_d_edge_max,
        delta_js_node_edge_alpha=delta_js_node_edge_alpha,
        js_success_threshold=js_success_threshold,
        js_success_margin=js_success_margin,
        deltae_margin=deltae_margin,
        completion_target_success=completion_target_success,
        completion_cost_if_unreached=completion_cost_if_unreached,
        n_workers=n_workers,
        seed=seed,
        progress_callback=progress_callback,
    )


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


def _robust_center_scale(values: np.ndarray, *, eps: float = 1e-9) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0
    return center, scale


def _residue_neighbors_with_self(n_residues: int, edges: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    neighbors: list[set[int]] = [set([i]) for i in range(int(n_residues))]
    for raw_r, raw_s in edges:
        r = int(raw_r)
        s = int(raw_s)
        if r < 0 or s < 0 or r >= int(n_residues) or s >= int(n_residues) or r == s:
            continue
        neighbors[r].add(s)
        neighbors[s].add(r)
    return [np.asarray(sorted(ids), dtype=int) for ids in neighbors]


def _selected_edge_neighbor_lists(selected_edges: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    incident: list[list[int]] = [[] for _ in range(len(selected_edges))]
    residue_to_cols: dict[int, list[int]] = {}
    for col, (raw_r, raw_s) in enumerate(selected_edges):
        r = int(raw_r)
        s = int(raw_s)
        residue_to_cols.setdefault(r, []).append(col)
        residue_to_cols.setdefault(s, []).append(col)
    for col, (raw_r, raw_s) in enumerate(selected_edges):
        r = int(raw_r)
        s = int(raw_s)
        linked = set([col])
        linked.update(residue_to_cols.get(r, []))
        linked.update(residue_to_cols.get(s, []))
        incident[col] = sorted(linked)
    return [np.asarray(cols, dtype=int) for cols in incident]


def _compute_local_node_energies(X: np.ndarray, model: PottsModel, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    labels = np.asarray(X, dtype=int)
    t_count, n_residues = labels.shape
    out = np.zeros((t_count, n_residues), dtype=np.float32)
    for ridx in range(n_residues):
        out[:, ridx] = np.asarray(model.h[ridx], dtype=float)[labels[:, ridx]].astype(np.float32, copy=False)
    for raw_r, raw_s in edges:
        r = int(raw_r)
        s = int(raw_s)
        vals = np.asarray(model.coupling(r, s), dtype=float)[labels[:, r], labels[:, s]].astype(np.float32, copy=False)
        out[:, r] += 0.5 * vals
        out[:, s] += 0.5 * vals
    return out


def _compute_selected_edge_energies(
    X: np.ndarray,
    model: PottsModel,
    selected_edges: Sequence[tuple[int, int]],
) -> np.ndarray:
    labels = np.asarray(X, dtype=int)
    t_count = labels.shape[0]
    edge_count = len(selected_edges)
    out = np.zeros((t_count, edge_count), dtype=np.float32)
    for col, (raw_r, raw_s) in enumerate(selected_edges):
        r = int(raw_r)
        s = int(raw_s)
        out[:, col] = np.asarray(model.coupling(r, s), dtype=float)[labels[:, r], labels[:, s]].astype(
            np.float32, copy=False
        )
    return out


def _compute_residue_frustration_raw(local_energies: np.ndarray, neighbors_with_self: Sequence[np.ndarray]) -> np.ndarray:
    local = np.asarray(local_energies, dtype=np.float32)
    if local.ndim != 2:
        raise ValueError("local_energies must be 2D.")
    t_count, n_residues = local.shape
    out = np.zeros((t_count, n_residues), dtype=np.float32)
    for ridx in range(n_residues):
        cols = neighbors_with_self[ridx] if ridx < len(neighbors_with_self) else np.asarray([ridx], dtype=int)
        if not isinstance(cols, np.ndarray) or cols.size == 0:
            cols = np.asarray([ridx], dtype=int)
        baseline = np.mean(local[:, cols], axis=1, dtype=np.float32)
        out[:, ridx] = np.abs(local[:, ridx] - baseline)
    return out


def _compute_edge_frustration_raw(edge_energies: np.ndarray, edge_neighbors: Sequence[np.ndarray]) -> np.ndarray:
    edge_vals = np.asarray(edge_energies, dtype=np.float32)
    if edge_vals.ndim != 2:
        raise ValueError("edge_energies must be 2D.")
    if edge_vals.shape[1] == 0:
        return np.zeros_like(edge_vals, dtype=np.float32)
    frame_mean = np.mean(edge_vals, axis=1, dtype=np.float32)
    out = np.zeros_like(edge_vals, dtype=np.float32)
    for col in range(edge_vals.shape[1]):
        cols = edge_neighbors[col] if col < len(edge_neighbors) else np.asarray([col], dtype=int)
        if not isinstance(cols, np.ndarray) or cols.size <= 1:
            baseline = frame_mean
        else:
            baseline = np.mean(edge_vals[:, cols], axis=1, dtype=np.float32)
        out[:, col] = np.abs(edge_vals[:, col] - baseline)
    return out


def _compute_endpoint_frame_scores(
    node_sym: np.ndarray,
    node_pol: np.ndarray,
    *,
    top_j: int = 10,
) -> dict[str, np.ndarray]:
    node_sym = np.asarray(node_sym, dtype=np.float32)
    node_pol = np.asarray(node_pol, dtype=np.float32)
    if node_sym.ndim != 2 or node_pol.ndim != 2:
        raise ValueError("node_sym and node_pol must be 2D.")
    if node_sym.shape != node_pol.shape:
        raise ValueError("node_sym and node_pol must have the same shape.")
    t_count, n_residues = node_sym.shape
    if n_residues <= 0:
        j = 0
        score_sym_topj = np.zeros((t_count,), dtype=np.float32)
        score_pol_abs_topj = np.zeros((t_count,), dtype=np.float32)
    else:
        j = max(1, min(int(top_j), int(n_residues)))
        node_sym_sorted = np.sort(node_sym, axis=1)[:, ::-1]
        node_pol_abs_sorted = np.sort(np.abs(node_pol), axis=1)[:, ::-1]
        score_sym_topj = np.sum(node_sym_sorted[:, :j], axis=1, dtype=np.float32)
        score_pol_abs_topj = np.sum(node_pol_abs_sorted[:, :j], axis=1, dtype=np.float32)
    score_sym_mean = np.mean(node_sym, axis=1, dtype=np.float32) if t_count else np.zeros((0,), dtype=np.float32)
    score_pol_abs_mean = (
        np.mean(np.abs(node_pol), axis=1, dtype=np.float32) if t_count else np.zeros((0,), dtype=np.float32)
    )
    rank_sym_topj = np.argsort(score_sym_topj)[::-1].astype(np.int32) if t_count else np.zeros((0,), dtype=np.int32)
    rank_pol_abs_topj = (
        np.argsort(score_pol_abs_topj)[::-1].astype(np.int32) if t_count else np.zeros((0,), dtype=np.int32)
    )
    frame_index = np.arange(t_count, dtype=np.int32)
    return {
        "analysis_format_version": np.asarray([2], dtype=np.int32),
        "ranking_default_top_j": np.asarray([j], dtype=np.int32),
        "frame_index": frame_index,
        "frame_index_filtered": frame_index.copy(),
        "frame_score_node_sym_topj_default": np.asarray(score_sym_topj, dtype=np.float32),
        "frame_score_node_sym_mean": np.asarray(score_sym_mean, dtype=np.float32),
        "frame_score_node_pol_abs_topj_default": np.asarray(score_pol_abs_topj, dtype=np.float32),
        "frame_score_node_pol_abs_mean": np.asarray(score_pol_abs_mean, dtype=np.float32),
        "frame_rank_node_sym_topj_default": np.asarray(rank_sym_topj, dtype=np.int32),
        "frame_rank_node_pol_abs_topj_default": np.asarray(rank_pol_abs_topj, dtype=np.int32),
    }


def ensure_endpoint_framewise_rankings(npz_path: str | Path, *, default_top_j: int = 10) -> dict[str, np.ndarray]:
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    required = (
        "frame_score_node_sym_topj_default",
        "frame_score_node_sym_mean",
        "frame_score_node_pol_abs_topj_default",
        "frame_score_node_pol_abs_mean",
        "frame_rank_node_sym_topj_default",
        "frame_rank_node_pol_abs_topj_default",
        "frame_index",
    )
    if all(key in payload for key in required):
        return payload
    node_sym = np.asarray(payload.get("frustration_node_sym_framewise"), dtype=np.float32)
    node_pol = np.asarray(payload.get("frustration_node_pol_framewise"), dtype=np.float32)
    ranking = _compute_endpoint_frame_scores(node_sym, node_pol, top_j=default_top_j)
    payload.update(ranking)
    np.savez_compressed(path, **payload)
    return payload


def load_endpoint_framewise_payload(
    npz_path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    include_ranks: bool = True,
    include_edges: bool = False,
    default_top_j: int = 10,
) -> dict[str, Any]:
    payload = ensure_endpoint_framewise_rankings(npz_path, default_top_j=default_top_j)
    frame_index = np.asarray(
        payload.get("frame_index_filtered", payload.get("frame_index", np.zeros((0,), dtype=np.int32))),
        dtype=np.int32,
    )
    frame_count = int(frame_index.shape[0])
    start_i = max(0, int(start or 0))
    step_i = max(1, int(step or 1))
    stop_i = frame_count if stop is None else max(start_i, min(frame_count, int(stop)))
    selection = np.arange(start_i, stop_i, step_i, dtype=np.int32)

    def _slice(name: str) -> list[Any]:
        arr = np.asarray(payload.get(name))
        if arr.ndim == 0:
            return [arr.item()]
        return np.asarray(arr[selection]).tolist()

    out: dict[str, Any] = {
        "frame_count": frame_count,
        "slice": {
            "start": start_i,
            "stop": stop_i,
            "step": step_i,
            "frame_indices": np.asarray(frame_index[selection], dtype=np.int32).tolist(),
        },
        "node": {
            "sym": _slice("frustration_node_sym_framewise"),
            "pol": _slice("frustration_node_pol_framewise"),
            "global_sym": _slice("global_node_sym_framewise"),
            "global_pol": _slice("global_node_pol_framewise"),
        },
        "analysis_format_version": int(
            np.asarray(payload.get("analysis_format_version", np.asarray([1], dtype=np.int32))).ravel()[0]
        ),
    }
    if include_edges:
        out["edge"] = {
            "sym": _slice("frustration_edge_sym_framewise"),
            "pol": _slice("frustration_edge_pol_framewise"),
            "global_sym": _slice("global_edge_sym_framewise"),
            "global_pol": _slice("global_edge_pol_framewise"),
            "selected_edge_indices": np.asarray(payload.get("selected_edge_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32).tolist(),
        }
    if include_ranks:
        out["ranking"] = {
            "default_top_j": int(np.asarray(payload.get("ranking_default_top_j", np.asarray([default_top_j], dtype=np.int32))).ravel()[0]),
            "score_node_sym_topj_default": np.asarray(payload.get("frame_score_node_sym_topj_default", np.zeros((0,), dtype=np.float32)), dtype=np.float32).tolist(),
            "score_node_sym_mean": np.asarray(payload.get("frame_score_node_sym_mean", np.zeros((0,), dtype=np.float32)), dtype=np.float32).tolist(),
            "score_node_pol_abs_topj_default": np.asarray(payload.get("frame_score_node_pol_abs_topj_default", np.zeros((0,), dtype=np.float32)), dtype=np.float32).tolist(),
            "score_node_pol_abs_mean": np.asarray(payload.get("frame_score_node_pol_abs_mean", np.zeros((0,), dtype=np.float32)), dtype=np.float32).tolist(),
            "rank_node_sym_topj_default": np.asarray(payload.get("frame_rank_node_sym_topj_default", np.zeros((0,), dtype=np.int32)), dtype=np.int32).tolist(),
            "rank_node_pol_abs_topj_default": np.asarray(payload.get("frame_rank_node_pol_abs_topj_default", np.zeros((0,), dtype=np.int32)), dtype=np.int32).tolist(),
        }
    return out


def _run_endpoint_frustration_batch(
    payloads: Sequence[dict[str, Any]],
    *,
    max_workers: int = 1,
    progress_callback: Callable[[str, int, int], None] | None = None,
    progress_label: str = "Computing endpoint frustration",
) -> list[dict[str, Any]]:
    n_payloads = int(len(payloads))
    if n_payloads <= 0:
        return []
    workers = max(1, int(max_workers))
    out_rows: list[dict[str, Any] | None] = [None] * n_payloads
    if progress_callback:
        progress_callback(progress_label, 0, n_payloads)
    if workers <= 1:
        for row, payload in enumerate(payloads):
            out_rows[row] = _endpoint_frustration_sample_worker(payload)
            if progress_callback:
                progress_callback(progress_label, row + 1, n_payloads)
    else:
        workers = min(workers, n_payloads)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_endpoint_frustration_sample_worker, payloads[row]): row for row in range(n_payloads)}
            done = 0
            for future in as_completed(futures):
                row = futures[future]
                out_rows[row] = future.result()
                done += 1
                if progress_callback:
                    progress_callback(progress_label, done, n_payloads)
    if any(v is None for v in out_rows):
        raise RuntimeError("Missing worker output while computing endpoint frustration batch.")
    return [row for row in out_rows if row is not None]


def _endpoint_resolve_model(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_id: str,
    system_dir: Path,
    ref: str,
) -> tuple[PottsModel, str | None, str, Path, str]:
    model_id = None
    model_name = None
    model_path = Path(str(ref))
    if not model_path.suffix:
        model_id = str(ref)
        models = store.list_potts_models(project_id, system_id, cluster_id)
        entry = next((m for m in models if m.get("model_id") == model_id), None)
        if not entry or not entry.get("path"):
            raise FileNotFoundError(f"Potts model_id not found on this system: {model_id}")
        model_name = str(entry.get("name") or model_id)
        model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
    else:
        if not model_path.is_absolute():
            model_path = store.resolve_path(project_id, system_id, str(model_path))
        model_name = model_path.stem
    if not model_path.exists():
        raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
    return (
        zero_sum_gauge_model(load_potts_model(str(model_path))),
        model_id,
        str(model_name),
        model_path,
        _relativize(model_path, system_dir),
    )


def _endpoint_resolve_sample_path(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_dir: Path,
    entry: dict[str, Any],
) -> Path:
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


def _endpoint_load_labels(
    *,
    sample_path: Path,
    md_label_mode: str,
    drop_invalid: bool,
) -> tuple[np.ndarray, int]:
    labels, invalid_count, _ = _endpoint_load_labels_with_frames(
        sample_path=sample_path,
        md_label_mode=md_label_mode,
        drop_invalid=drop_invalid,
    )
    return labels, invalid_count


def _endpoint_load_labels_with_frames(
    *,
    sample_path: Path,
    md_label_mode: str,
    drop_invalid: bool,
) -> tuple[np.ndarray, int, np.ndarray]:
    s = load_sample_npz(sample_path)
    X = s.labels
    frame_indices = (
        np.asarray(s.frame_indices, dtype=np.int64)
        if s.frame_indices is not None and s.frame_indices.shape[0] == X.shape[0]
        else np.arange(X.shape[0], dtype=np.int64)
    )
    invalid_count = 0
    if md_label_mode in {"halo", "labels_halo"} and s.labels_halo is not None:
        X = s.labels_halo
    if drop_invalid and s.invalid_mask is not None:
        invalid_mask = np.asarray(s.invalid_mask, dtype=bool)
        invalid_count = int(np.count_nonzero(invalid_mask))
        keep = ~invalid_mask
        if keep.shape[0] == X.shape[0]:
            X = X[keep]
            frame_indices = frame_indices[keep]
    return np.asarray(X, dtype=int), invalid_count, np.asarray(frame_indices, dtype=np.int64)


def _endpoint_frustration_sample_worker(payload: dict[str, Any]) -> dict[str, Any]:
    project_id = str(payload["project_id"])
    system_id = str(payload["system_id"])
    cluster_id = str(payload["cluster_id"])
    model_a_path = Path(str(payload["model_a_path"]))
    model_b_path = Path(str(payload["model_b_path"]))
    sample_id = str(payload["sample_id"])
    sample_label = str(payload.get("sample_label") or sample_id)
    sample_type = str(payload.get("sample_type") or "sample")
    sample_path = Path(str(payload["sample_path"]))
    md_label_mode = str(payload.get("md_label_mode") or "assigned").strip().lower()
    drop_invalid = bool(payload.get("drop_invalid", True))
    selected_edges = [tuple(int(v) for v in edge) for edge in list(payload.get("selected_edges") or [])]
    top_edge_indices = np.asarray(payload.get("top_edge_indices", []), dtype=np.int32)

    store = ProjectStore(base_dir=Path(os.getenv("PHASE_DATA_ROOT", "/app/data")) / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    model_a, _, _, _, _ = _endpoint_resolve_model(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        system_dir=cluster_dirs["system_dir"],
        ref=str(model_a_path),
    )
    model_b, _, _, _, _ = _endpoint_resolve_model(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        system_dir=cluster_dirs["system_dir"],
        ref=str(model_b_path),
    )

    n_residues = int(len(model_a.h))
    k_list = [int(k) for k in model_a.K_list()]
    k_max = int(max(k_list)) if k_list else 0
    edges_a = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_a.edges or []) if int(r) != int(s)}
    edges_b = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or []) if int(r) != int(s)}
    edges = sorted(edges_a & edges_b)
    residue_neighbors = _residue_neighbors_with_self(n_residues, edges)
    edge_neighbors = _selected_edge_neighbor_lists(selected_edges)

    dh_list: list[np.ndarray] = []
    for ridx in range(n_residues):
        a = np.asarray(model_a.h[ridx], dtype=float).ravel()
        b = np.asarray(model_b.h[ridx], dtype=float).ravel()
        if a.shape != b.shape:
            raise ValueError(f"Model alphabets do not match at residue {ridx}: {a.shape} vs {b.shape}")
        dh_list.append(a - b)

    dJ: dict[tuple[int, int], np.ndarray] = {}
    for edge in selected_edges:
        dJ[edge] = np.asarray(model_a.coupling(*edge), dtype=float) - np.asarray(model_b.coupling(*edge), dtype=float)

    labels, invalid_count = _endpoint_load_labels(
        sample_path=sample_path,
        md_label_mode=md_label_mode,
        drop_invalid=drop_invalid,
    )
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(f"Sample labels are empty: {sample_id}")
    if int(labels.shape[1]) != n_residues:
        raise ValueError(
            f"Sample labels do not match model size for {sample_id}: got N={labels.shape[1]}, expected {n_residues}"
        )
    if np.min(labels) < 0:
        raise ValueError(
            f"Sample contains negative labels for {sample_id}. "
            "Use md_label_mode='assigned' or remap unassigned labels before analysis."
        )
    for ridx in range(n_residues):
        ki = int(k_list[ridx])
        if ki <= 0:
            continue
        mx = int(np.max(labels[:, ridx])) if labels.shape[0] else -1
        if mx >= ki:
            raise ValueError(
                f"Sample labels out of range for {sample_id} at residue {ridx}: max={mx}, expected in [0,{ki-1}]"
            )

    n_frames = int(labels.shape[0])
    p_node_row = np.zeros((n_residues, k_max), dtype=np.float32)
    q_residue_row = np.zeros((n_residues,), dtype=np.float32)
    for ridx in range(n_residues):
        ki = int(k_list[ridx])
        counts = np.bincount(np.asarray(labels[:, ridx], dtype=int), minlength=ki).astype(np.float32, copy=False)
        p = counts / float(n_frames) if n_frames > 0 else np.zeros((ki,), dtype=np.float32)
        p_node_row[ridx, :ki] = p
        mask = (np.asarray(dh_list[ridx], dtype=float) < 0).astype(np.float32, copy=False)
        q_residue_row[ridx] = float(np.sum(p * mask)) if p.size else np.nan

    q_edge_row = np.zeros((len(selected_edges),), dtype=np.float32)
    for col, edge in enumerate(selected_edges):
        r, s = edge
        vals = dJ[edge][labels[:, r], labels[:, s]]
        q_edge_row[col] = float(np.mean(vals < 0)) if vals.size else np.nan

    # Global endpoint separation per frame: ΔE = E_model_A - E_model_B.
    delta_energy = np.asarray(model_a.energy_batch(labels) - model_b.energy_batch(labels), dtype=np.float32)

    local_node_a = _compute_local_node_energies(labels, model_a, edges)
    local_node_b = _compute_local_node_energies(labels, model_b, edges)
    raw_node_a = _compute_residue_frustration_raw(local_node_a, residue_neighbors)
    raw_node_b = _compute_residue_frustration_raw(local_node_b, residue_neighbors)
    node_center_a, node_scale_a = _robust_center_scale(raw_node_a)
    node_center_b, node_scale_b = _robust_center_scale(raw_node_b)
    node_a_z = (raw_node_a - float(node_center_a)) / float(node_scale_a)
    node_b_z = (raw_node_b - float(node_center_b)) / float(node_scale_b)
    node_sym = 0.5 * (node_a_z + node_b_z)
    node_pol = node_a_z - node_b_z
    global_node_sym_series = np.mean(node_sym, axis=1, dtype=np.float32)
    global_node_pol_series = np.mean(node_pol, axis=1, dtype=np.float32)

    edge_a = _compute_selected_edge_energies(labels, model_a, selected_edges)
    edge_b = _compute_selected_edge_energies(labels, model_b, selected_edges)
    raw_edge_a = _compute_edge_frustration_raw(edge_a, edge_neighbors)
    raw_edge_b = _compute_edge_frustration_raw(edge_b, edge_neighbors)
    edge_center_a, edge_scale_a = _robust_center_scale(raw_edge_a)
    edge_center_b, edge_scale_b = _robust_center_scale(raw_edge_b)
    edge_a_z = (raw_edge_a - float(edge_center_a)) / float(edge_scale_a) if selected_edges else raw_edge_a
    edge_b_z = (raw_edge_b - float(edge_center_b)) / float(edge_scale_b) if selected_edges else raw_edge_b
    edge_sym = 0.5 * (edge_a_z + edge_b_z)
    edge_pol = edge_a_z - edge_b_z
    global_edge_sym_series = (
        np.mean(edge_sym, axis=1, dtype=np.float32) if selected_edges else np.zeros((n_frames,), dtype=np.float32)
    )
    global_edge_pol_series = (
        np.mean(edge_pol, axis=1, dtype=np.float32) if selected_edges else np.zeros((n_frames,), dtype=np.float32)
    )

    if selected_edges:
        edge_sym_mean = np.mean(edge_sym, axis=0, dtype=np.float32).astype(np.float32, copy=False)
        edge_sym_std = np.std(edge_sym, axis=0, dtype=np.float32).astype(np.float32, copy=False)
        edge_sym_median = np.median(edge_sym, axis=0).astype(np.float32, copy=False)
        edge_pol_mean = np.mean(edge_pol, axis=0, dtype=np.float32).astype(np.float32, copy=False)
        edge_pol_std = np.std(edge_pol, axis=0, dtype=np.float32).astype(np.float32, copy=False)
        edge_pol_median = np.median(edge_pol, axis=0).astype(np.float32, copy=False)
        global_edge_sym_mean = float(np.mean(global_edge_sym_series)) if global_edge_sym_series.size else np.nan
        global_edge_sym_std = float(np.std(global_edge_sym_series)) if global_edge_sym_series.size else np.nan
        global_edge_pol_mean = float(np.mean(global_edge_pol_series)) if global_edge_pol_series.size else np.nan
        global_edge_pol_std = float(np.std(global_edge_pol_series)) if global_edge_pol_series.size else np.nan
    else:
        edge_sym_mean = np.zeros((0,), dtype=np.float32)
        edge_sym_std = np.zeros((0,), dtype=np.float32)
        edge_sym_median = np.zeros((0,), dtype=np.float32)
        edge_pol_mean = np.zeros((0,), dtype=np.float32)
        edge_pol_std = np.zeros((0,), dtype=np.float32)
        edge_pol_median = np.zeros((0,), dtype=np.float32)
        global_edge_sym_mean = 0.0
        global_edge_sym_std = 0.0
        global_edge_pol_mean = 0.0
        global_edge_pol_std = 0.0

    framewise = {
        "frustration_node_sym_framewise": np.asarray(node_sym, dtype=np.float32),
        "frustration_node_pol_framewise": np.asarray(node_pol, dtype=np.float32),
        "frustration_edge_sym_framewise": np.asarray(edge_sym, dtype=np.float32),
        "frustration_edge_pol_framewise": np.asarray(edge_pol, dtype=np.float32),
        "global_node_sym_framewise": np.asarray(global_node_sym_series, dtype=np.float32),
        "global_node_pol_framewise": np.asarray(global_node_pol_series, dtype=np.float32),
        "global_edge_sym_framewise": np.asarray(global_edge_sym_series, dtype=np.float32),
        "global_edge_pol_framewise": np.asarray(global_edge_pol_series, dtype=np.float32),
        "delta_energy_framewise": np.asarray(delta_energy, dtype=np.float32),
        "frame_count": np.asarray([n_frames], dtype=np.int32),
        "selected_edge_indices": np.asarray(top_edge_indices, dtype=np.int32),
    }
    framewise.update(_compute_endpoint_frame_scores(node_sym, node_pol, top_j=10))

    return {
        "sample_id": sample_id,
        "sample_label": sample_label,
        "sample_type": sample_type,
        "frame_count": n_frames,
        "invalid_count": int(invalid_count),
        "top_edge_indices": np.asarray(top_edge_indices, dtype=np.int32),
        "p_node": np.asarray(p_node_row, dtype=np.float32),
        "q_residue_all": np.asarray(q_residue_row, dtype=np.float32),
        "q_edge": np.asarray(q_edge_row, dtype=np.float32),
        "delta_energy": np.asarray(delta_energy, dtype=np.float32),
        "frustration_node_sym_mean": np.mean(node_sym, axis=0, dtype=np.float32).astype(np.float32, copy=False),
        "frustration_node_sym_std": np.std(node_sym, axis=0, dtype=np.float32).astype(np.float32, copy=False),
        "frustration_node_sym_median": np.median(node_sym, axis=0).astype(np.float32, copy=False),
        "frustration_node_pol_mean": np.mean(node_pol, axis=0, dtype=np.float32).astype(np.float32, copy=False),
        "frustration_node_pol_std": np.std(node_pol, axis=0, dtype=np.float32).astype(np.float32, copy=False),
        "frustration_node_pol_median": np.median(node_pol, axis=0).astype(np.float32, copy=False),
        "frustration_edge_sym_mean": np.asarray(edge_sym_mean, dtype=np.float32),
        "frustration_edge_sym_std": np.asarray(edge_sym_std, dtype=np.float32),
        "frustration_edge_sym_median": np.asarray(edge_sym_median, dtype=np.float32),
        "frustration_edge_pol_mean": np.asarray(edge_pol_mean, dtype=np.float32),
        "frustration_edge_pol_std": np.asarray(edge_pol_std, dtype=np.float32),
        "frustration_edge_pol_median": np.asarray(edge_pol_median, dtype=np.float32),
        "node_norm_center_a": float(node_center_a),
        "node_norm_scale_a": float(node_scale_a),
        "node_norm_center_b": float(node_center_b),
        "node_norm_scale_b": float(node_scale_b),
        "edge_norm_center_a": float(edge_center_a),
        "edge_norm_scale_a": float(edge_scale_a),
        "edge_norm_center_b": float(edge_center_b),
        "edge_norm_scale_b": float(edge_scale_b),
        "global_node_sym_mean": float(np.mean(global_node_sym_series)) if global_node_sym_series.size else np.nan,
        "global_node_sym_std": float(np.std(global_node_sym_series)) if global_node_sym_series.size else np.nan,
        "global_node_pol_mean": float(np.mean(global_node_pol_series)) if global_node_pol_series.size else np.nan,
        "global_node_pol_std": float(np.std(global_node_pol_series)) if global_node_pol_series.size else np.nan,
        "global_edge_sym_mean": global_edge_sym_mean,
        "global_edge_sym_std": global_edge_sym_std,
        "global_edge_pol_mean": global_edge_pol_mean,
        "global_edge_pol_std": global_edge_pol_std,
        "framewise": framewise,
    }


def upsert_endpoint_frustration_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str,
    model_b_ref: str,
    sample_ids: Sequence[str],
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    top_k_edges: int = 2000,
    n_workers: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """
    Store interpretable endpoint-local analysis for a fixed (A,B) model pair.

    Main artifact (`analysis.npz`) contains compact per-sample summaries used by the UI:
      - node/edge commitment
      - node/edge frustration summaries (mean/std/median; symmetric + polarity channels)
      - robust normalization parameters used for frustration scaling

    Per-sample framewise frustration arrays are written under:
      clusters/<cluster_id>/analyses/endpoint_frustration/<analysis_id>/samples/<sample_id>.npz
    """
    md_label_mode = (md_label_mode or "assigned").strip().lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise ValueError("md_label_mode must be 'assigned' or 'halo'.")
    top_k_edges = int(top_k_edges)
    if top_k_edges < 1:
        raise ValueError("top_k_edges must be >= 1.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    model_a, model_a_id, model_a_name, model_a_path_abs, model_a_path = _endpoint_resolve_model(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        system_dir=system_dir,
        ref=model_a_ref,
    )
    model_b, model_b_id, model_b_name, model_b_path_abs, model_b_path = _endpoint_resolve_model(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        system_dir=system_dir,
        ref=model_b_ref,
    )
    if model_a_id and model_b_id and model_a_id == model_b_id:
        raise ValueError("Select two different models.")
    if len(model_a.h) != len(model_b.h):
        raise ValueError("Model sizes do not match.")
    n_residues = int(len(model_a.h))
    if n_residues <= 0:
        raise ValueError("Invalid Potts model size.")
    k_list = [int(k) for k in model_a.K_list()]
    k_max = int(max(k_list)) if k_list else 0
    if k_max <= 0:
        raise ValueError("Invalid Potts alphabet size.")

    edges_a = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_a.edges or []) if int(r) != int(s)}
    edges_b = {(min(int(r), int(s)), max(int(r), int(s))) for r, s in (model_b.edges or []) if int(r) != int(s)}
    edges = sorted(edges_a & edges_b)

    dh_list: list[np.ndarray] = []
    for ridx in range(n_residues):
        a = np.asarray(model_a.h[ridx], dtype=float).ravel()
        b = np.asarray(model_b.h[ridx], dtype=float).ravel()
        if a.shape != b.shape:
            raise ValueError(f"Model alphabets do not match at residue {ridx}: {a.shape} vs {b.shape}")
        dh_list.append(a - b)
    dh = np.zeros((n_residues, k_max), dtype=np.float32)
    for ridx in range(n_residues):
        ki = int(dh_list[ridx].shape[0])
        if ki > 0:
            dh[ridx, :ki] = np.asarray(dh_list[ridx], dtype=np.float32)

    dJ: dict[tuple[int, int], np.ndarray] = {}
    for edge in edges:
        dJ[edge] = np.asarray(model_a.coupling(*edge), dtype=float) - np.asarray(model_b.coupling(*edge), dtype=float)

    d_residue = np.zeros((n_residues,), dtype=np.float32)
    for ridx in range(n_residues):
        d_residue[ridx] = float(np.linalg.norm(np.asarray(dh_list[ridx], dtype=float).ravel(), ord=2))
    d_edge = np.zeros((len(edges),), dtype=np.float32)
    for eidx, edge in enumerate(edges):
        d_edge[eidx] = float(np.linalg.norm(np.asarray(dJ[edge], dtype=float).ravel(), ord=2))

    top_k_e = min(top_k_edges, len(edges))
    top_edge_indices = np.argsort(d_edge)[::-1][:top_k_e].astype(int) if top_k_e > 0 else np.zeros((0,), dtype=int)
    selected_edges = [edges[int(eidx)] for eidx in top_edge_indices.tolist()]

    key = json.dumps(
        {
            "analysis_type": "endpoint_frustration",
            "model_a_id": model_a_id or model_a_path,
            "model_b_id": model_b_id or model_b_path,
            "md_label_mode": md_label_mode,
            "drop_invalid": bool(drop_invalid),
            "top_k_edges": int(top_k_e),
        },
        sort_keys=True,
    )
    analysis_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    analyses_root = _ensure_analysis_dir(cluster_dir, "endpoint_frustration")
    analysis_dir = analyses_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
    framewise_root = analysis_dir / "samples"
    framewise_root.mkdir(parents=True, exist_ok=True)

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

    samples = store.list_samples(project_id, system_id, cluster_id)
    sample_by_id: dict[str, dict[str, Any]] = {str(s.get("sample_id")): s for s in samples if s.get("sample_id")}

    sample_labels: list[str] = []
    sample_types: list[str] = []
    sample_frame_counts = np.zeros((len(merged),), dtype=np.int32)
    sample_invalid_counts = np.zeros((len(merged),), dtype=np.int32)
    q_residue_all = np.zeros((len(merged), n_residues), dtype=np.float32)
    p_node = np.zeros((len(merged), n_residues, k_max), dtype=np.float32)
    q_edge = np.zeros((len(merged), top_k_e), dtype=np.float32)
    frustration_node_sym_mean = np.zeros((len(merged), n_residues), dtype=np.float32)
    frustration_node_sym_std = np.zeros((len(merged), n_residues), dtype=np.float32)
    frustration_node_sym_median = np.zeros((len(merged), n_residues), dtype=np.float32)
    frustration_node_pol_mean = np.zeros((len(merged), n_residues), dtype=np.float32)
    frustration_node_pol_std = np.zeros((len(merged), n_residues), dtype=np.float32)
    frustration_node_pol_median = np.zeros((len(merged), n_residues), dtype=np.float32)
    frustration_edge_sym_mean = np.zeros((len(merged), top_k_e), dtype=np.float32)
    frustration_edge_sym_std = np.zeros((len(merged), top_k_e), dtype=np.float32)
    frustration_edge_sym_median = np.zeros((len(merged), top_k_e), dtype=np.float32)
    frustration_edge_pol_mean = np.zeros((len(merged), top_k_e), dtype=np.float32)
    frustration_edge_pol_std = np.zeros((len(merged), top_k_e), dtype=np.float32)
    frustration_edge_pol_median = np.zeros((len(merged), top_k_e), dtype=np.float32)
    node_norm_center_a = np.zeros((len(merged),), dtype=np.float32)
    node_norm_scale_a = np.ones((len(merged),), dtype=np.float32)
    node_norm_center_b = np.zeros((len(merged),), dtype=np.float32)
    node_norm_scale_b = np.ones((len(merged),), dtype=np.float32)
    edge_norm_center_a = np.zeros((len(merged),), dtype=np.float32)
    edge_norm_scale_a = np.ones((len(merged),), dtype=np.float32)
    edge_norm_center_b = np.zeros((len(merged),), dtype=np.float32)
    edge_norm_scale_b = np.ones((len(merged),), dtype=np.float32)
    global_node_sym_mean = np.zeros((len(merged),), dtype=np.float32)
    global_node_sym_std = np.zeros((len(merged),), dtype=np.float32)
    global_node_pol_mean = np.zeros((len(merged),), dtype=np.float32)
    global_node_pol_std = np.zeros((len(merged),), dtype=np.float32)
    global_edge_sym_mean = np.zeros((len(merged),), dtype=np.float32)
    global_edge_sym_std = np.zeros((len(merged),), dtype=np.float32)
    global_edge_pol_mean = np.zeros((len(merged),), dtype=np.float32)
    global_edge_pol_std = np.zeros((len(merged),), dtype=np.float32)
    delta_energy_all: list[np.ndarray] = []
    delta_energy_mean = np.zeros((len(merged),), dtype=np.float32)
    delta_energy_std = np.zeros((len(merged),), dtype=np.float32)
    delta_energy_median = np.zeros((len(merged),), dtype=np.float32)
    delta_energy_min = np.zeros((len(merged),), dtype=np.float32)
    delta_energy_max = np.zeros((len(merged),), dtype=np.float32)
    payloads: list[dict[str, Any]] = []
    for sid in merged:
        entry = sample_by_id.get(sid)
        if not entry:
            raise FileNotFoundError(f"Sample not found on this cluster: {sid}")
        payloads.append(
            {
                "project_id": project_id,
                "system_id": system_id,
                "cluster_id": cluster_id,
                "model_a_path": str(model_a_path_abs),
                "model_b_path": str(model_b_path_abs),
                "sample_id": sid,
                "sample_label": str(entry.get("name") or sid),
                "sample_type": str(entry.get("type") or "sample"),
                "sample_path": str(
                    _endpoint_resolve_sample_path(
                        store=store,
                        project_id=project_id,
                        system_id=system_id,
                        cluster_dir=cluster_dir,
                        entry=entry,
                    )
                ),
                "md_label_mode": md_label_mode,
                "drop_invalid": bool(drop_invalid),
                "selected_edges": [list(edge) for edge in selected_edges],
                "top_edge_indices": np.asarray(top_edge_indices, dtype=np.int32).tolist(),
            }
        )

    if n_workers is None or int(n_workers) <= 0:
        workers_used = max(1, min(int(os.cpu_count() or 1), max(1, len(payloads))))
    else:
        workers_used = max(1, min(int(n_workers), max(1, len(payloads))))

    out_rows = _run_endpoint_frustration_batch(
        payloads,
        max_workers=workers_used,
        progress_callback=progress_callback,
        progress_label="Computing endpoint frustration",
    )

    for row, out_row in enumerate(out_rows):
        sid = str(out_row["sample_id"])
        sample_labels.append(str(out_row["sample_label"]))
        sample_types.append(str(out_row["sample_type"]))
        sample_frame_counts[row] = int(out_row["frame_count"])
        sample_invalid_counts[row] = int(out_row["invalid_count"])
        p_node[row] = np.asarray(out_row["p_node"], dtype=np.float32)
        q_residue_all[row] = np.asarray(out_row["q_residue_all"], dtype=np.float32)
        q_edge[row] = np.asarray(out_row["q_edge"], dtype=np.float32)
        de_row = np.asarray(out_row.get("delta_energy", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        delta_energy_all.append(de_row)
        if de_row.size:
            delta_energy_mean[row] = float(np.mean(de_row))
            delta_energy_std[row] = float(np.std(de_row))
            delta_energy_median[row] = float(np.median(de_row))
            delta_energy_min[row] = float(np.min(de_row))
            delta_energy_max[row] = float(np.max(de_row))
        else:
            delta_energy_mean[row] = np.nan
            delta_energy_std[row] = np.nan
            delta_energy_median[row] = np.nan
            delta_energy_min[row] = np.nan
            delta_energy_max[row] = np.nan
        frustration_node_sym_mean[row] = np.asarray(out_row["frustration_node_sym_mean"], dtype=np.float32)
        frustration_node_sym_std[row] = np.asarray(out_row["frustration_node_sym_std"], dtype=np.float32)
        frustration_node_sym_median[row] = np.asarray(out_row["frustration_node_sym_median"], dtype=np.float32)
        frustration_node_pol_mean[row] = np.asarray(out_row["frustration_node_pol_mean"], dtype=np.float32)
        frustration_node_pol_std[row] = np.asarray(out_row["frustration_node_pol_std"], dtype=np.float32)
        frustration_node_pol_median[row] = np.asarray(out_row["frustration_node_pol_median"], dtype=np.float32)
        frustration_edge_sym_mean[row] = np.asarray(out_row["frustration_edge_sym_mean"], dtype=np.float32)
        frustration_edge_sym_std[row] = np.asarray(out_row["frustration_edge_sym_std"], dtype=np.float32)
        frustration_edge_sym_median[row] = np.asarray(out_row["frustration_edge_sym_median"], dtype=np.float32)
        frustration_edge_pol_mean[row] = np.asarray(out_row["frustration_edge_pol_mean"], dtype=np.float32)
        frustration_edge_pol_std[row] = np.asarray(out_row["frustration_edge_pol_std"], dtype=np.float32)
        frustration_edge_pol_median[row] = np.asarray(out_row["frustration_edge_pol_median"], dtype=np.float32)
        node_norm_center_a[row] = float(out_row["node_norm_center_a"])
        node_norm_scale_a[row] = float(out_row["node_norm_scale_a"])
        node_norm_center_b[row] = float(out_row["node_norm_center_b"])
        node_norm_scale_b[row] = float(out_row["node_norm_scale_b"])
        edge_norm_center_a[row] = float(out_row["edge_norm_center_a"])
        edge_norm_scale_a[row] = float(out_row["edge_norm_scale_a"])
        edge_norm_center_b[row] = float(out_row["edge_norm_center_b"])
        edge_norm_scale_b[row] = float(out_row["edge_norm_scale_b"])
        global_node_sym_mean[row] = float(out_row["global_node_sym_mean"])
        global_node_sym_std[row] = float(out_row["global_node_sym_std"])
        global_node_pol_mean[row] = float(out_row["global_node_pol_mean"])
        global_node_pol_std[row] = float(out_row["global_node_pol_std"])
        global_edge_sym_mean[row] = float(out_row["global_edge_sym_mean"])
        global_edge_sym_std[row] = float(out_row["global_edge_sym_std"])
        global_edge_pol_mean[row] = float(out_row["global_edge_pol_mean"])
        global_edge_pol_std[row] = float(out_row["global_edge_pol_std"])

        framewise_npz = framewise_root / f"{sid}.npz"
        np.savez_compressed(framewise_npz, **out_row["framewise"])

    for old in framewise_root.glob("*.npz"):
        if old.stem not in seen:
            try:
                old.unlink()
            except Exception:
                pass

    energy_bins = 80
    de_concat = np.concatenate(delta_energy_all, axis=0) if delta_energy_all else np.zeros((0,), dtype=np.float32)
    if de_concat.size:
        lo = float(np.nanmin(de_concat))
        hi = float(np.nanmax(de_concat))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        pad = 1e-6 * (hi - lo)
        delta_energy_bins = np.linspace(lo - pad, hi + pad, energy_bins + 1, dtype=np.float32)
    else:
        delta_energy_bins = np.linspace(-1.0, 1.0, energy_bins + 1, dtype=np.float32)
    delta_energy_hist = np.zeros((len(merged), energy_bins), dtype=np.float32)
    for row, de_row in enumerate(delta_energy_all):
        if de_row.size:
            hist, _ = np.histogram(np.asarray(de_row, dtype=float), bins=np.asarray(delta_energy_bins, dtype=float), density=True)
            delta_energy_hist[row] = np.asarray(hist, dtype=np.float32)

    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([2], dtype=np.int32),
        sample_framewise_available=np.asarray([1], dtype=np.int32),
        framewise_default_start=np.asarray([0], dtype=np.int32),
        framewise_default_stop=np.asarray([100], dtype=np.int32),
        framewise_default_step=np.asarray([1], dtype=np.int32),
        framewise_default_top_j=np.asarray([10], dtype=np.int32),
        edges=np.asarray(edges, dtype=np.int32),
        D_residue=np.asarray(d_residue, dtype=np.float32),
        D_edge=np.asarray(d_edge, dtype=np.float32),
        top_edge_indices=np.asarray(top_edge_indices, dtype=np.int32),
        sample_ids=np.asarray(merged, dtype=str),
        sample_labels=np.asarray(sample_labels, dtype=str),
        sample_types=np.asarray(sample_types, dtype=str),
        sample_frame_counts=np.asarray(sample_frame_counts, dtype=np.int32),
        sample_invalid_counts=np.asarray(sample_invalid_counts, dtype=np.int32),
        K_list=np.asarray(k_list, dtype=np.int32),
        dh=np.asarray(dh, dtype=np.float32),
        p_node=np.asarray(p_node, dtype=np.float32),
        q_residue_all=np.asarray(q_residue_all, dtype=np.float32),
        q_edge=np.asarray(q_edge, dtype=np.float32),
        delta_energy_bins=np.asarray(delta_energy_bins, dtype=np.float32),
        delta_energy_hist=np.asarray(delta_energy_hist, dtype=np.float32),
        delta_energy_mean=np.asarray(delta_energy_mean, dtype=np.float32),
        delta_energy_std=np.asarray(delta_energy_std, dtype=np.float32),
        delta_energy_median=np.asarray(delta_energy_median, dtype=np.float32),
        delta_energy_min=np.asarray(delta_energy_min, dtype=np.float32),
        delta_energy_max=np.asarray(delta_energy_max, dtype=np.float32),
        frustration_node_sym_mean=np.asarray(frustration_node_sym_mean, dtype=np.float32),
        frustration_node_sym_std=np.asarray(frustration_node_sym_std, dtype=np.float32),
        frustration_node_sym_median=np.asarray(frustration_node_sym_median, dtype=np.float32),
        frustration_node_pol_mean=np.asarray(frustration_node_pol_mean, dtype=np.float32),
        frustration_node_pol_std=np.asarray(frustration_node_pol_std, dtype=np.float32),
        frustration_node_pol_median=np.asarray(frustration_node_pol_median, dtype=np.float32),
        frustration_edge_sym_mean=np.asarray(frustration_edge_sym_mean, dtype=np.float32),
        frustration_edge_sym_std=np.asarray(frustration_edge_sym_std, dtype=np.float32),
        frustration_edge_sym_median=np.asarray(frustration_edge_sym_median, dtype=np.float32),
        frustration_edge_pol_mean=np.asarray(frustration_edge_pol_mean, dtype=np.float32),
        frustration_edge_pol_std=np.asarray(frustration_edge_pol_std, dtype=np.float32),
        frustration_edge_pol_median=np.asarray(frustration_edge_pol_median, dtype=np.float32),
        node_norm_center_a=np.asarray(node_norm_center_a, dtype=np.float32),
        node_norm_scale_a=np.asarray(node_norm_scale_a, dtype=np.float32),
        node_norm_center_b=np.asarray(node_norm_center_b, dtype=np.float32),
        node_norm_scale_b=np.asarray(node_norm_scale_b, dtype=np.float32),
        edge_norm_center_a=np.asarray(edge_norm_center_a, dtype=np.float32),
        edge_norm_scale_a=np.asarray(edge_norm_scale_a, dtype=np.float32),
        edge_norm_center_b=np.asarray(edge_norm_center_b, dtype=np.float32),
        edge_norm_scale_b=np.asarray(edge_norm_scale_b, dtype=np.float32),
        global_node_sym_mean=np.asarray(global_node_sym_mean, dtype=np.float32),
        global_node_sym_std=np.asarray(global_node_sym_std, dtype=np.float32),
        global_node_pol_mean=np.asarray(global_node_pol_mean, dtype=np.float32),
        global_node_pol_std=np.asarray(global_node_pol_std, dtype=np.float32),
        global_edge_sym_mean=np.asarray(global_edge_sym_mean, dtype=np.float32),
        global_edge_sym_std=np.asarray(global_edge_sym_std, dtype=np.float32),
        global_edge_pol_mean=np.asarray(global_edge_pol_mean, dtype=np.float32),
        global_edge_pol_std=np.asarray(global_edge_pol_std, dtype=np.float32),
    )

    now = _utc_now()
    created_at = now
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = str(old.get("created_at") or created_at)
        except Exception:
            created_at = now

    residue_rank = np.argsort(d_residue)[::-1][: min(10, n_residues)].astype(int).tolist()
    frustration_rank = np.argsort(np.nanmean(frustration_node_sym_mean, axis=0))[::-1][: min(10, n_residues)].astype(int).tolist()
    meta = {
        "analysis_id": analysis_id,
        "analysis_type": "endpoint_frustration",
        "analysis_format_version": 2,
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
        "top_k_edges": int(top_k_e),
        "sample_framewise_available": True,
        "framewise_default_window": {"start": 0, "stop": 100, "step": 1},
        "framewise_default_top_j": 10,
        "workers_used": int(workers_used),
        "paths": {
            "analysis_npz": str(npz_path.relative_to(system_dir)),
            "sample_framewise_dir": str(framewise_root.relative_to(system_dir)),
        },
        "summary": {
            "n_residues": int(n_residues),
            "n_edges": int(len(edges)),
            "n_selected_edges": int(top_k_e),
            "n_samples": int(len(merged)),
            "workers_used": int(workers_used),
            "sample_ids": merged,
            "delta_energy": {
                "definition": "E_model_A - E_model_B",
                "bins": int(energy_bins),
            },
            "top_residues_by_commitment_weight": [
                {"residue_index": int(idx), "score": float(d_residue[int(idx)])} for idx in residue_rank
            ],
            "top_residues_by_mean_frustration": [
                {"residue_index": int(idx), "score": float(np.nanmean(frustration_node_sym_mean[:, int(idx)]))}
                for idx in frustration_rank
            ],
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    return {"metadata": _convert_nan_to_none(meta), "analysis_npz": str(npz_path), "analysis_dir": str(analysis_dir)}


def _compute_delta_energy_components(
    labels: np.ndarray,
    model_a: PottsModel,
    model_b: PottsModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame node and edge contributions for E_A - E_B."""
    X = np.asarray(labels, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("labels must be 2D.")
    n_frames, n_residues = X.shape
    node = np.zeros((n_frames, n_residues), dtype=np.float32)
    for r in range(n_residues):
        node[:, r] = np.asarray(model_a.h[r][X[:, r]] - model_b.h[r][X[:, r]], dtype=np.float32)

    edge_set = {
        tuple(sorted((int(r), int(s))))
        for r, s in list(model_a.edges or []) + list(model_b.edges or [])
        if int(r) != int(s)
    }
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    edge = np.zeros((n_frames, int(edges.shape[0])), dtype=np.float32)
    for col, (r_raw, s_raw) in enumerate(edges.tolist()):
        r, s = int(r_raw), int(s_raw)
        vals = np.zeros((n_frames,), dtype=np.float32)
        if (r, s) in model_a.J:
            vals += np.asarray(model_a.J[(r, s)][X[:, r], X[:, s]], dtype=np.float32)
        if (r, s) in model_b.J:
            vals -= np.asarray(model_b.J[(r, s)][X[:, r], X[:, s]], dtype=np.float32)
        edge[:, col] = vals
    return node, edge, edges


def _delta_energy_sample_worker(payload: dict[str, Any]) -> dict[str, Any]:
    model_a_path = Path(str(payload["model_a_path"]))
    model_b_path = Path(str(payload["model_b_path"]))
    sample_id = str(payload["sample_id"])
    sample_label = str(payload.get("sample_label") or sample_id)
    sample_type = str(payload.get("sample_type") or "sample")
    sample_path = Path(str(payload["sample_path"]))
    md_label_mode = str(payload.get("md_label_mode") or "assigned").strip().lower()
    drop_invalid = bool(payload.get("drop_invalid", True))
    frame_limit = int(payload.get("frame_limit") or 0)
    seed = int(payload.get("seed") or 0)

    model_a = zero_sum_gauge_model(load_potts_model(str(model_a_path)))
    model_b = zero_sum_gauge_model(load_potts_model(str(model_b_path)))
    if len(model_a.h) != len(model_b.h):
        raise ValueError("Model sizes do not match.")
    n_residues = int(len(model_a.h))

    labels, invalid_count, source_frame_indices = _endpoint_load_labels_with_frames(
        sample_path=sample_path,
        md_label_mode=md_label_mode,
        drop_invalid=drop_invalid,
    )
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(f"Sample labels are empty: {sample_id}")
    if int(labels.shape[1]) != n_residues:
        raise ValueError(
            f"Sample labels do not match model size for {sample_id}: got N={labels.shape[1]}, expected {n_residues}"
        )
    if np.min(labels) < 0:
        raise ValueError(f"Sample contains negative labels for {sample_id}; use assigned labels or drop invalid frames.")

    available_frames = int(labels.shape[0])
    selected_indices = np.arange(available_frames, dtype=np.int32)
    if frame_limit > 0 and frame_limit < available_frames:
        rng = np.random.default_rng(seed + int(zlib.adler32(sample_id.encode("utf-8"))))
        selected_indices = np.sort(rng.choice(available_frames, size=frame_limit, replace=False)).astype(np.int32)
        labels = labels[selected_indices]
        source_frame_indices = source_frame_indices[selected_indices]

    node_delta, edge_delta, edge_index = _compute_delta_energy_components(labels, model_a, model_b)
    delta_energy = np.asarray(node_delta.sum(axis=1) + edge_delta.sum(axis=1), dtype=np.float32)
    return {
        "sample_id": sample_id,
        "sample_label": sample_label,
        "sample_type": sample_type,
        "available_frame_count": available_frames,
        "used_frame_count": int(labels.shape[0]),
        "invalid_count": int(invalid_count),
        "frame_limit": int(frame_limit),
        "selected_frame_indices": np.asarray(selected_indices, dtype=np.int32),
        "source_frame_indices": np.asarray(source_frame_indices, dtype=np.int64),
        "delta_energy": delta_energy,
        "delta_node_energy": node_delta,
        "delta_edge_energy": edge_delta,
        "edge_index": edge_index,
    }


def _run_delta_energy_batch(
    payloads: Sequence[dict[str, Any]],
    *,
    max_workers: int = 1,
    progress_callback: Callable[[str, int, int], None] | None = None,
    progress_label: str = "Computing delta energies",
) -> list[dict[str, Any]]:
    n_payloads = int(len(payloads))
    if n_payloads <= 0:
        return []
    workers = max(1, int(max_workers))
    out_rows: list[dict[str, Any] | None] = [None] * n_payloads
    if progress_callback:
        progress_callback(progress_label, 0, n_payloads)
    if workers <= 1:
        for row, payload in enumerate(payloads):
            out_rows[row] = _delta_energy_sample_worker(payload)
            if progress_callback:
                progress_callback(progress_label, row + 1, n_payloads)
    else:
        workers = min(workers, n_payloads)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_delta_energy_sample_worker, payloads[row]): row for row in range(n_payloads)}
            done = 0
            for future in as_completed(futures):
                row = futures[future]
                out_rows[row] = future.result()
                done += 1
                if progress_callback:
                    progress_callback(progress_label, done, n_payloads)
    if any(v is None for v in out_rows):
        raise RuntimeError("Missing worker output while computing delta-energy batch.")
    return [row for row in out_rows if row is not None]


def upsert_delta_energy_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str,
    model_b_ref: str,
    sample_ids: Sequence[str],
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    frame_limits: dict[str, int] | None = None,
    seed: int = 0,
    energy_bins: int = 80,
    n_workers: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    md_label_mode = (md_label_mode or "assigned").strip().lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise ValueError("md_label_mode must be 'assigned' or 'halo'.")
    energy_bins = int(energy_bins)
    if energy_bins < 5:
        raise ValueError("energy_bins must be >= 5.")

    requested = [str(s).strip() for s in sample_ids if str(s).strip()]
    seen: set[str] = set()
    requested = [sid for sid in requested if not (sid in seen or seen.add(sid))]
    if not requested:
        raise ValueError("No samples selected.")

    frame_limits = {str(k): max(0, int(v or 0)) for k, v in (frame_limits or {}).items()}
    seed = int(seed or 0)

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    model_a, model_a_id, model_a_name, model_a_path_abs, model_a_path = _endpoint_resolve_model(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        system_dir=system_dir,
        ref=model_a_ref,
    )
    model_b, model_b_id, model_b_name, model_b_path_abs, model_b_path = _endpoint_resolve_model(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        system_dir=system_dir,
        ref=model_b_ref,
    )
    if model_a_id and model_b_id and model_a_id == model_b_id:
        raise ValueError("Select two different models.")
    if len(model_a.h) != len(model_b.h):
        raise ValueError("Model sizes do not match.")

    samples = store.list_samples(project_id, system_id, cluster_id)
    sample_by_id: dict[str, dict[str, Any]] = {str(s.get("sample_id")): s for s in samples if s.get("sample_id")}

    sample_aliases: dict[str, str] = {}
    for entry in samples:
        sid = str(entry.get("sample_id") or "").strip()
        if not sid:
            continue
        aliases = {
            sid,
            str(entry.get("name") or "").strip(),
            str(entry.get("state_id") or "").strip(),
            str(entry.get("state_name") or "").strip(),
            str((entry.get("metadata") or {}).get("state_id") or "").strip() if isinstance(entry.get("metadata"), dict) else "",
            str((entry.get("metadata") or {}).get("state_name") or "").strip() if isinstance(entry.get("metadata"), dict) else "",
        }
        name = str(entry.get("name") or "").strip()
        sample_type = str(entry.get("type") or "").strip().lower()
        if sample_type == "md_eval" and name.lower().startswith("md "):
            aliases.add(name[3:].strip())
        for alias in aliases:
            key_alias = str(alias or "").strip().lower()
            if key_alias and key_alias not in sample_aliases:
                sample_aliases[key_alias] = sid

    resolved_requested: list[str] = []
    unknown_requested: list[str] = []
    resolved_frame_limits: dict[str, int] = {}
    for ref in requested:
        sid = sample_by_id.get(ref) and ref
        if not sid:
            sid = sample_aliases.get(ref.strip().lower())
        if not sid:
            unknown_requested.append(ref)
            continue
        limit = int(frame_limits.get(ref, frame_limits.get(sid, 0)))
        if sid not in resolved_requested:
            resolved_requested.append(sid)
            resolved_frame_limits[sid] = limit
        elif limit > 0:
            resolved_frame_limits[sid] = limit
    if unknown_requested:
        available = sorted(
            {
                str(entry.get("sample_id") or "")
                for entry in samples
                if entry.get("sample_id")
            }
            | {
                str(entry.get("name") or "")
                for entry in samples
                if entry.get("name")
            }
        )
        raise FileNotFoundError(
            "Delta-energy sample/state reference(s) not found on this cluster: "
            f"{unknown_requested}. Available sample ids/names include: {available[:40]}"
            + (" ..." if len(available) > 40 else "")
        )
    if not resolved_requested:
        raise ValueError("No valid samples selected.")

    key = json.dumps(
        {
            "analysis_type": "delta_energy",
            "model_a_id": model_a_id or model_a_path,
            "model_b_id": model_b_id or model_b_path,
            "sample_ids": resolved_requested,
            "frame_limits": {sid: int(resolved_frame_limits.get(sid, 0)) for sid in resolved_requested},
            "seed": int(seed),
            "md_label_mode": md_label_mode,
            "drop_invalid": bool(drop_invalid),
            "energy_bins": int(energy_bins),
        },
        sort_keys=True,
    )
    analysis_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    analyses_root = _ensure_analysis_dir(cluster_dir, "delta_energy")
    analysis_dir = analyses_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME

    payloads: list[dict[str, Any]] = []
    for sid in resolved_requested:
        entry = sample_by_id.get(sid)
        if not entry:
            raise FileNotFoundError(f"Sample not found on this cluster: {sid}")
        payloads.append(
            {
                "model_a_path": str(model_a_path_abs),
                "model_b_path": str(model_b_path_abs),
                "sample_id": sid,
                "sample_label": str(entry.get("name") or sid),
                "sample_type": str(entry.get("type") or "sample"),
                "sample_path": str(
                    _endpoint_resolve_sample_path(
                        store=store,
                        project_id=project_id,
                        system_id=system_id,
                        cluster_dir=cluster_dir,
                        entry=entry,
                    )
                ),
                "md_label_mode": md_label_mode,
                "drop_invalid": bool(drop_invalid),
                "frame_limit": int(resolved_frame_limits.get(sid, 0)),
                "seed": int(seed),
            }
        )

    workers_used = max(1, min(int(n_workers or os.cpu_count() or 1), len(payloads)))
    out_rows = _run_delta_energy_batch(
        payloads,
        max_workers=workers_used,
        progress_callback=progress_callback,
        progress_label="Computing delta energies",
    )
    computed_ids = [str(row.get("sample_id") or "") for row in out_rows]
    if computed_ids != resolved_requested:
        raise RuntimeError(
            "Delta-energy worker outputs do not match requested samples: "
            f"requested={resolved_requested}, computed={computed_ids}"
        )

    sample_labels = [str(row["sample_label"]) for row in out_rows]
    sample_types = [str(row["sample_type"]) for row in out_rows]
    available_counts = np.asarray([int(row["available_frame_count"]) for row in out_rows], dtype=np.int32)
    used_counts = np.asarray([int(row["used_frame_count"]) for row in out_rows], dtype=np.int32)
    invalid_counts = np.asarray([int(row["invalid_count"]) for row in out_rows], dtype=np.int32)
    limits = np.asarray([int(row["frame_limit"]) for row in out_rows], dtype=np.int32)
    delta_energy_all = [np.asarray(row["delta_energy"], dtype=np.float32) for row in out_rows]

    component_dir = analysis_dir / "samples"
    component_dir.mkdir(parents=True, exist_ok=True)
    component_paths: list[str] = []
    edge_index = np.asarray(out_rows[0].get("edge_index", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32) if out_rows else np.zeros((0, 2), dtype=np.int32)
    for row in out_rows:
        sid = str(row.get("sample_id") or "sample")
        component_path = component_dir / f"{sid}.npz"
        np.savez_compressed(
            component_path,
            sample_id=np.asarray([sid], dtype=str),
            selected_frame_indices=np.asarray(row.get("selected_frame_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
            source_frame_indices=np.asarray(row.get("source_frame_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64),
            delta_energy=np.asarray(row["delta_energy"], dtype=np.float32),
            delta_node_energy=np.asarray(row.get("delta_node_energy", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32),
            delta_edge_energy=np.asarray(row.get("delta_edge_energy", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32),
            edge_index=np.asarray(row.get("edge_index", edge_index), dtype=np.int32),
        )
        component_paths.append(str(component_path.relative_to(system_dir)))

    cluster_npz_path = cluster_dir / "cluster.npz"
    residue_keys_for_output = np.asarray([], dtype=str)
    if cluster_npz_path.exists():
        try:
            with np.load(cluster_npz_path, allow_pickle=False) as cluster_npz:
                if "residue_keys" in cluster_npz.files:
                    residue_keys_for_output = np.asarray(cluster_npz["residue_keys"], dtype=str)
        except Exception:
            residue_keys_for_output = np.asarray([], dtype=str)

    de_concat = np.concatenate(delta_energy_all, axis=0) if delta_energy_all else np.zeros((0,), dtype=np.float32)
    if de_concat.size:
        lo = float(np.nanmin(de_concat))
        hi = float(np.nanmax(de_concat))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        pad = 1e-6 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, energy_bins + 1, dtype=np.float32)
    else:
        bins = np.linspace(-1.0, 1.0, energy_bins + 1, dtype=np.float32)

    hist = np.zeros((len(out_rows), energy_bins), dtype=np.float32)
    means = np.zeros((len(out_rows),), dtype=np.float32)
    stds = np.zeros((len(out_rows),), dtype=np.float32)
    medians = np.zeros((len(out_rows),), dtype=np.float32)
    mins = np.zeros((len(out_rows),), dtype=np.float32)
    maxs = np.zeros((len(out_rows),), dtype=np.float32)
    for row, de in enumerate(delta_energy_all):
        if de.size:
            h, _ = np.histogram(np.asarray(de, dtype=float), bins=np.asarray(bins, dtype=float), density=True)
            hist[row] = np.asarray(h, dtype=np.float32)
            means[row] = float(np.mean(de))
            stds[row] = float(np.std(de))
            medians[row] = float(np.median(de))
            mins[row] = float(np.min(de))
            maxs[row] = float(np.max(de))
        else:
            means[row] = stds[row] = medians[row] = mins[row] = maxs[row] = np.nan

    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([1], dtype=np.int32),
        sample_ids=np.asarray(resolved_requested, dtype=str),
        requested_sample_refs=np.asarray(requested, dtype=str),
        sample_labels=np.asarray(sample_labels, dtype=str),
        sample_types=np.asarray(sample_types, dtype=str),
        sample_frame_counts=used_counts,
        sample_available_frame_counts=available_counts,
        sample_invalid_counts=invalid_counts,
        sample_frame_limits=limits,
        component_sample_paths=np.asarray(component_paths, dtype=str),
        residue_keys=residue_keys_for_output,
        edge_index=np.asarray(edge_index, dtype=np.int32),
        delta_energy_bins=np.asarray(bins, dtype=np.float32),
        delta_energy_hist=np.asarray(hist, dtype=np.float32),
        delta_energy_mean=means,
        delta_energy_std=stds,
        delta_energy_median=medians,
        delta_energy_min=mins,
        delta_energy_max=maxs,
    )

    now = _utc_now()
    meta = {
        "analysis_id": analysis_id,
        "analysis_type": "delta_energy",
        "analysis_format_version": 1,
        "created_at": now,
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
        "seed": int(seed),
        "energy_bins": int(energy_bins),
        "requested_sample_refs": requested,
        "sample_ids": resolved_requested,
        "frame_limits": {sid: int(resolved_frame_limits.get(sid, 0)) for sid in resolved_requested},
        "component_sample_paths": {sid: component_paths[i] for i, sid in enumerate(resolved_requested) if i < len(component_paths)},
        "paths": {"analysis_npz": str(npz_path.relative_to(system_dir))},
        "summary": {
            "n_requested_refs": int(len(requested)),
            "n_samples": int(len(resolved_requested)),
            "requested_sample_refs": requested,
            "sample_ids": resolved_requested,
            "sample_frame_counts": used_counts.tolist(),
            "sample_available_frame_counts": available_counts.tolist(),
            "workers_used": int(workers_used),
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    return {"metadata": _convert_nan_to_none(meta), "analysis_npz": str(npz_path), "analysis_dir": str(analysis_dir)}


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
