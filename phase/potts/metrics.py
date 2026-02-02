from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def marginals(labels: np.ndarray, K: Sequence[int]) -> List[np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    T, N = labels.shape
    K = list(map(int, K))
    out = []
    for r in range(N):
        counts = np.bincount(labels[:, r], minlength=K[r]).astype(float)
        p = counts / max(1, counts.sum())
        out.append(p)
    return out


def pairwise_joints_on_edges(
    labels: np.ndarray,
    K: Sequence[int],
    edges: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    T, N = labels.shape
    K = list(map(int, K))
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for r, s in edges:
        Kr, Ks = K[r], K[s]
        counts = np.zeros((Kr, Ks), dtype=float)
        np.add.at(counts, (labels[:, r], labels[:, s]), 1.0)
        p = counts / max(1, counts.sum())
        out[(r, s)] = p
    return out


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def per_residue_js(
    p_list: List[np.ndarray],
    q_list: List[np.ndarray],
) -> np.ndarray:
    if len(p_list) != len(q_list):
        raise ValueError("Mismatched residue count.")
    out = np.zeros(len(p_list), dtype=float)
    for r in range(len(p_list)):
        out[r] = js_divergence(p_list[r], q_list[r])
    return out


def avg_js_over_edges(
    P_a: Dict[Tuple[int, int], np.ndarray],
    P_b: Dict[Tuple[int, int], np.ndarray],
    edges: Sequence[Tuple[int, int]],
) -> float:
    vals: List[float] = []
    for e in edges:
        pa = P_a[e].ravel()
        pb = P_b[e].ravel()
        vals.append(js_divergence(pa, pb))
    return float(np.mean(vals)) if vals else float("nan")


def combined_distance(
    X_a: np.ndarray,
    X_b: np.ndarray,
    *,
    K: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    w_marg: float = 1.0,
    w_pair: float = 1.0,
) -> float:
    """
    A single scalar distance between two sample sets in reduced space.

    Default combines:
      - mean per-residue JS divergence of marginals
      - mean JS divergence of pairwise edge joint distributions

    This is designed for beta_eff calibration: "which beta makes Gibbs samples most similar to SA samples?"
    """
    p_a = marginals(X_a, K)
    p_b = marginals(X_b, K)
    js_m = float(np.mean(per_residue_js(p_a, p_b)))

    if w_pair != 0.0 and len(edges) > 0:
        P_a = pairwise_joints_on_edges(X_a, K, edges)
        P_b = pairwise_joints_on_edges(X_b, K, edges)
        js_p = avg_js_over_edges(P_a, P_b, edges)
    else:
        js_p = 0.0

    return w_marg * js_m + w_pair * js_p


def pad_pairwise_joints(
    P: Dict[Tuple[int, int], np.ndarray],
    edges: Sequence[Tuple[int, int]],
    K: Sequence[int],
    *,
    pad_value: float = np.nan,
) -> np.ndarray:
    if not edges:
        return np.zeros((0, 0, 0), dtype=float)
    max_k = max(map(int, K)) if len(K) else 0
    out = np.full((len(edges), max_k, max_k), pad_value, dtype=float)
    for idx, (r, s) in enumerate(edges):
        mat = P[(r, s)]
        out[idx, : mat.shape[0], : mat.shape[1]] = mat
    return out


def pairwise_joints_padded(
    labels: np.ndarray,
    K: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    *,
    pad_value: float = np.nan,
) -> np.ndarray:
    P = pairwise_joints_on_edges(labels, K, edges)
    return pad_pairwise_joints(P, edges, K, pad_value=pad_value)


def per_residue_js_from_padded(
    p_a: np.ndarray,
    p_b: np.ndarray,
    K: Sequence[int],
) -> np.ndarray:
    K = list(map(int, K))
    if p_a.shape != p_b.shape or p_a.ndim != 2:
        raise ValueError("Expected matching 2D padded marginal matrices.")
    out = np.zeros(len(K), dtype=float)
    for r, Kr in enumerate(K):
        if Kr <= 0:
            out[r] = float("nan")
            continue
        out[r] = js_divergence(p_a[r, :Kr], p_b[r, :Kr])
    return out


def per_edge_js_from_padded(
    p2_a: np.ndarray,
    p2_b: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    K: Sequence[int],
) -> np.ndarray:
    if p2_a.shape != p2_b.shape or p2_a.ndim != 3:
        raise ValueError("Expected matching 3D padded pairwise matrices.")
    K = list(map(int, K))
    out = np.zeros(len(edges), dtype=float)
    for idx, (r, s) in enumerate(edges):
        Kr, Ks = K[r], K[s]
        if Kr <= 0 or Ks <= 0:
            out[idx] = float("nan")
            continue
        a = p2_a[idx, :Kr, :Ks].ravel()
        b = p2_b[idx, :Kr, :Ks].ravel()
        out[idx] = js_divergence(a, b)
    return out
