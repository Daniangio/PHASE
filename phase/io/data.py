from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Any

import numpy as np


UnassignedPolicy = Literal["drop_frames", "treat_as_state", "error"]


@dataclass(frozen=True)
class TorsionDataset:
    residue_keys: np.ndarray          # (N,)
    labels: np.ndarray                # (T, N) in {0..K_r-1} (and maybe -1 pre-sanitize)
    cluster_counts: np.ndarray        # (N,) K_r per residue (excluding -1 unless treat_as_state)
    edges: List[Tuple[int, int]]      # unique undirected (r<s)
    frame_state_ids: Optional[np.ndarray] = None
    frame_metastable_ids: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


def _dedup_edges(edge_index: np.ndarray, n_res: int) -> List[Tuple[int, int]]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"contact_edge_index must have shape (2, E). Got {edge_index.shape}.")
    pairs = set()
    for a, b in edge_index.T:
        r, s = int(a), int(b)
        if r == s:
            continue
        if r < 0 or s < 0 or r >= n_res or s >= n_res:
            continue
        if r > s:
            r, s = s, r
        pairs.add((r, s))
    return sorted(pairs)


def load_npz(
    path: str,
    *,
    unassigned_policy: UnassignedPolicy = "drop_frames",
    allow_missing_edges: bool = False,
) -> TorsionDataset:
    """
    Expects keys similar to your clustering output:
      - residue_keys: (N,)
      - merged__labels: (T,N) (may contain -1)
      - merged__cluster_counts: (N,)
      - contact_edge_index: (2,E)
    """
    data = np.load(path, allow_pickle=True)

    residue_keys = data["residue_keys"]
    if "merged__labels_assigned" in data:
        labels = np.asarray(data["merged__labels_assigned"], dtype=int)
    else:
        labels = np.asarray(data["merged__labels"], dtype=int)
    cluster_counts = np.asarray(data["merged__cluster_counts"], dtype=int)

    edges: List[Tuple[int, int]] = []
    if "contact_edge_index" not in data:
        if not allow_missing_edges:
            raise KeyError("contact_edge_index not found in npz.")
    else:
        edges = _dedup_edges(np.asarray(data["contact_edge_index"]), n_res=labels.shape[1])

    frame_state_ids = None
    if "merged__frame_state_ids" in data:
        frame_state_ids = np.asarray(data["merged__frame_state_ids"])
    frame_metastable_ids = None
    if "merged__frame_metastable_ids" in data:
        frame_metastable_ids = np.asarray(data["merged__frame_metastable_ids"])
    frame_indices = None
    if "merged__frame_indices" in data:
        frame_indices = np.asarray(data["merged__frame_indices"])

    metadata = None
    if "metadata_json" in data:
        try:
            raw = data["metadata_json"]
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            metadata = json.loads(str(raw))
        except Exception:
            metadata = None

    ds = TorsionDataset(
        residue_keys=residue_keys,
        labels=labels,
        cluster_counts=cluster_counts,
        edges=edges,
        frame_state_ids=frame_state_ids,
        frame_metastable_ids=frame_metastable_ids,
        frame_indices=frame_indices,
        metadata=metadata,
    )
    return sanitize_dataset(ds, unassigned_policy=unassigned_policy)


def sanitize_dataset(
    ds: TorsionDataset,
    *,
    unassigned_policy: UnassignedPolicy = "drop_frames",
) -> TorsionDataset:
    """
    Ensures labels are in valid ranges.
    Handles -1 according to policy:
      - drop_frames: remove frames with any -1
      - treat_as_state: map -1 -> K_r for each residue; cluster_counts += 1
      - error: raise if -1 exists
    """
    labels = np.array(ds.labels, copy=True)
    K = np.array(ds.cluster_counts, copy=True)
    frame_state_ids = ds.frame_state_ids
    frame_metastable_ids = ds.frame_metastable_ids
    frame_indices = ds.frame_indices
    T, N = labels.shape

    has_unassigned = np.any(labels < 0)
    if has_unassigned:
        if unassigned_policy == "error":
            bad = np.where(labels < 0)
            raise ValueError(f"Found unassigned labels (-1) at indices like {list(zip(bad[0][:5], bad[1][:5]))}.")

        if unassigned_policy == "drop_frames":
            keep = np.all(labels >= 0, axis=1)
            labels = labels[keep]
            if frame_state_ids is not None and frame_state_ids.shape[0] == keep.shape[0]:
                frame_state_ids = frame_state_ids[keep]
            if frame_metastable_ids is not None and frame_metastable_ids.shape[0] == keep.shape[0]:
                frame_metastable_ids = frame_metastable_ids[keep]
            if frame_indices is not None and frame_indices.shape[0] == keep.shape[0]:
                frame_indices = frame_indices[keep]
        elif unassigned_policy == "treat_as_state":
            # per residue, map -1 -> K[r], then K[r] += 1
            for r in range(N):
                mask = labels[:, r] < 0
                if np.any(mask):
                    labels[mask, r] = K[r]
                    K[r] += 1
        else:
            raise ValueError(f"Unknown unassigned_policy={unassigned_policy}")

    # Validate ranges
    for r in range(N):
        if np.any(labels[:, r] < 0):
            raise ValueError("Negative label remained after sanitize.")
        if np.any(labels[:, r] >= K[r]):
            mx = labels[:, r].max()
            raise ValueError(f"Label out of range at residue {r}: max={mx} but K={K[r]}.")

    return TorsionDataset(
        residue_keys=ds.residue_keys,
        labels=labels,
        cluster_counts=K,
        edges=ds.edges,
        frame_state_ids=frame_state_ids,
        frame_metastable_ids=frame_metastable_ids,
        frame_indices=frame_indices,
        metadata=ds.metadata,
    )
