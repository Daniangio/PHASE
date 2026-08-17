from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


SAMPLE_NPZ_FILENAME = "sample.npz"


def _as_string_array(values: np.ndarray | list[str] | tuple[str, ...]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray([], dtype="U1")
    flat = [str(v) for v in arr.reshape(-1).tolist()]
    max_len = max(1, max(len(v) for v in flat))
    return np.asarray(flat, dtype=f"U{max_len}").reshape(arr.shape)


@dataclass(frozen=True)
class SampleNPZ:
    """
    Minimal, uniform sample artifact used across:
      - MD evaluation samples (cluster assignment of MD frames)
      - Potts sampling (Gibbs / SA)

    Required:
      - labels: (T, N) int32

    Optional:
      - invalid_mask: (T,) bool (e.g. SA constraint violations)
      - valid_counts: (T,) int32 (e.g. #valid residues per SA read)
      - labels_halo: (T, N) int32 (MD-only; "halo" label mode)
      - frame_indices: (T,) int64 (MD-only; original trajectory frame indices)
      - frame_state_ids: (T,) str (MD-only; original state id per frame, when mixing sources)

    SA artifacts can additionally contain audit arrays written through ``extra``:
      - sa_initial_labels: (T, N) exact labels supplied to each annealing read
      - sa_initial_md_frame_indices: (T,) source MD frame, or -1 for non-MD/chain starts
      - sa_hamming_distance: (T,) changed residues from that read's initial state
    """

    labels: np.ndarray
    invalid_mask: Optional[np.ndarray] = None
    valid_counts: Optional[np.ndarray] = None
    labels_halo: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None
    frame_state_ids: Optional[np.ndarray] = None


def save_sample_npz(
    path: str | Path,
    *,
    labels: np.ndarray,
    invalid_mask: np.ndarray | None = None,
    valid_counts: np.ndarray | None = None,
    labels_halo: np.ndarray | None = None,
    frame_indices: np.ndarray | None = None,
    frame_state_ids: np.ndarray | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload: Dict[str, Any] = {
        "labels": np.asarray(labels, dtype=np.int32),
    }
    if invalid_mask is not None:
        payload["invalid_mask"] = np.asarray(invalid_mask, dtype=bool)
    if valid_counts is not None:
        payload["valid_counts"] = np.asarray(valid_counts, dtype=np.int32)
    if labels_halo is not None:
        payload["labels_halo"] = np.asarray(labels_halo, dtype=np.int32)
    if frame_indices is not None:
        payload["frame_indices"] = np.asarray(frame_indices, dtype=np.int64)
    if frame_state_ids is not None:
        payload["frame_state_ids"] = _as_string_array(frame_state_ids)
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_sample_npz(path: str | Path) -> SampleNPZ:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        labels_halo = None
        frame_indices = None
        frame_state_ids = None
        if "labels" in data:
            labels = np.asarray(data["labels"], dtype=np.int32)
            labels_halo = np.asarray(data["labels_halo"], dtype=np.int32) if "labels_halo" in data else None
            frame_indices = np.asarray(data["frame_indices"], dtype=np.int64) if "frame_indices" in data else None
            frame_state_ids = np.asarray(data["frame_state_ids"], dtype=str) if "frame_state_ids" in data else None
        else:
            # Legacy md_eval.npz support
            if "assigned__labels_assigned" in data:
                labels = np.asarray(data["assigned__labels_assigned"], dtype=np.int32)
            elif "assigned__labels" in data:
                labels = np.asarray(data["assigned__labels"], dtype=np.int32)
            else:
                raise KeyError("Sample NPZ missing 'labels' (and legacy assigned__labels keys).")
            if "assigned__labels" in data:
                labels_halo = np.asarray(data["assigned__labels"], dtype=np.int32)
            if "assigned__frame_indices" in data:
                frame_indices = np.asarray(data["assigned__frame_indices"], dtype=np.int64)
            if "assigned__frame_state_id" in data:
                raw = data["assigned__frame_state_id"]
                try:
                    frame_state_ids = np.asarray(raw, dtype=str)
                except Exception:
                    frame_state_ids = None
        invalid_mask = np.asarray(data["invalid_mask"], dtype=bool) if "invalid_mask" in data else None
        valid_counts = np.asarray(data["valid_counts"], dtype=np.int32) if "valid_counts" in data else None
    return SampleNPZ(
        labels=labels,
        invalid_mask=invalid_mask,
        valid_counts=valid_counts,
        labels_halo=labels_halo,
        frame_indices=frame_indices,
        frame_state_ids=frame_state_ids,
    )


def load_sample_labels(path: str | Path, *, label_mode: str = "labels") -> np.ndarray:
    """
    Convenience loader used by analysis.

    label_mode:
      - "labels": primary labels
      - "halo": MD-only halo labels (falls back to primary labels)
    """
    sample = load_sample_npz(path)
    mode = (label_mode or "labels").strip().lower()
    if mode in {"halo", "labels_halo"} and sample.labels_halo is not None:
        return sample.labels_halo
    return sample.labels


def load_sample_valid_mask(path: str | Path) -> np.ndarray:
    """
    Returns a boolean mask of valid rows (True = keep).
    If invalid_mask is missing, all rows are valid.
    """
    sample = load_sample_npz(path)
    if sample.invalid_mask is None:
        return np.ones(sample.labels.shape[0], dtype=bool)
    return ~np.asarray(sample.invalid_mask, dtype=bool)
