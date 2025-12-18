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
