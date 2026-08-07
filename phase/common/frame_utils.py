from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def select_frames_per_cluster(
    labels: np.ndarray,
    frame_indices: np.ndarray,
    *,
    residue_index: int,
    max_per_cluster: int,
    max_total: int = 500,
) -> Tuple[list[int], Dict[int, int]]:
    """Select evenly spaced frames from every assigned cluster of one residue."""
    values = np.asarray(labels, dtype=np.int32)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if values.ndim != 2 or frames.ndim != 1 or frames.shape[0] != values.shape[0]:
        raise ValueError("Labels and frame indices are not aligned.")
    if residue_index < 0 or residue_index >= values.shape[1]:
        raise ValueError("Residue index is outside the labels array.")
    if max_per_cluster <= 0 or max_total <= 0:
        raise ValueError("Frame limits must be positive.")

    selected_by_cluster: Dict[int, list[int]] = {}
    column = values[:, residue_index]
    for cluster_id in sorted(int(v) for v in np.unique(column) if int(v) >= 0):
        candidates = frames[column == cluster_id]
        if candidates.size > max_per_cluster:
            positions = np.rint(np.linspace(0, candidates.size - 1, num=max_per_cluster)).astype(int)
            candidates = candidates[np.unique(positions)]
        selected_by_cluster[cluster_id] = [int(v) for v in candidates.tolist()]

    # Round-robin truncation keeps the global browser limit balanced across clusters.
    selected: list[int] = []
    offset = 0
    while len(selected) < max_total:
        added = False
        for cluster_id in sorted(selected_by_cluster):
            rows = selected_by_cluster[cluster_id]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) >= max_total:
                    break
        if not added:
            break
        offset += 1
    selected_set = set(selected)
    counts = {
        cluster_id: sum(1 for frame in cluster_frames if frame in selected_set)
        for cluster_id, cluster_frames in selected_by_cluster.items()
    }
    return sorted(selected), counts

