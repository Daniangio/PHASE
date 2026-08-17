from __future__ import annotations

from typing import Any

import numpy as np


def apply_model_energy_residue_selection(
    payload_np: dict[str, np.ndarray],
    selected_residue_indices: set[int],
) -> dict[str, np.ndarray]:
    """Replace total energies with selected-node and touching-edge contributions."""
    node = np.asarray(payload_np.get("node_energies", np.asarray([])), dtype=np.float32)
    edge = np.asarray(payload_np.get("edge_energies", np.asarray([])), dtype=np.float32)
    edge_index = np.asarray(payload_np.get("edge_index", np.asarray([])), dtype=np.int32)
    if node.ndim != 2:
        raise ValueError("This energy analysis has no per-node components. Rerun Sampling Explorer analysis.")
    selected = {int(idx) for idx in selected_residue_indices if 0 <= int(idx) < node.shape[1]}
    if not selected:
        raise ValueError("Residue selection did not match any residue in this energy analysis.")

    node_columns = sorted(selected)
    energies = np.sum(node[:, node_columns], axis=1, dtype=np.float64)
    if edge.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[1] != 2 or edge.shape[1] != edge_index.shape[0]:
        raise ValueError("This energy analysis has invalid per-edge components. Rerun Sampling Explorer analysis.")
    edge_mask = np.asarray(
        [(int(r) in selected) or (int(s) in selected) for r, s in edge_index.tolist()],
        dtype=bool,
    )
    if np.any(edge_mask):
        energies += np.sum(edge[:, edge_mask], axis=1, dtype=np.float64)

    out = dict(payload_np)
    out["energies"] = np.asarray(energies, dtype=np.float64)
    return out


def compact_potts_nn_payload(payload_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep_keys = {
        "analysis_format_version",
        "residue_keys",
        "residue_display_labels",
        "edges",
        "per_residue_mean",
        "per_residue_std",
        "per_residue_median",
        "per_residue_q25",
        "per_residue_q75",
        "per_residue_node_mean",
        "per_residue_node_std",
        "per_residue_node_median",
        "per_residue_node_q25",
        "per_residue_node_q75",
        "per_residue_edge_mean",
        "per_residue_edge_std",
        "per_residue_edge_median",
        "per_residue_edge_q25",
        "per_residue_edge_q75",
        "per_edge_mean",
        "per_edge_std",
        "per_edge_median",
        "per_edge_q25",
        "per_edge_q75",
    }
    return {key: np.asarray(value) for key, value in payload_np.items() if key in keep_keys}


def downsample_potts_nn_payload(payload_np: dict[str, np.ndarray], *, row_limit: int, seed: int) -> dict[str, np.ndarray]:
    nn_dist_global = np.asarray(payload_np.get("nn_dist_global", np.asarray([], dtype=float)))
    row_count = int(nn_dist_global.shape[0]) if nn_dist_global.ndim == 1 else 0
    if row_limit <= 0 or row_count <= row_limit:
        payload_np["sampled_unique_row_indices"] = np.arange(row_count, dtype=np.int32)
        payload_np["sampled_unique_row_count"] = np.asarray([row_count], dtype=np.int32)
        payload_np["original_unique_row_count"] = np.asarray([row_count], dtype=np.int32)
        payload_np["downsampled"] = np.asarray([0], dtype=np.int32)
        return payload_np

    rng = np.random.default_rng(int(seed))
    keep_rows = np.sort(rng.choice(row_count, size=int(row_limit), replace=False).astype(np.int32))
    row_keys = {
        "sample_unique_sequences",
        "sample_unique_counts",
        "sample_unique_rep_original_indices",
        "sample_unique_rep_frame_indices",
        "nn_md_unique_idx",
        "nn_md_rep_original_idx",
        "nn_md_rep_frame_idx",
        "nn_dist_global",
        "nn_dist_node",
        "nn_dist_edge",
        "nn_dist_residue_node",
        "nn_dist_residue_edge",
        "nn_dist_residue",
        "nn_dist_edge_per_edge",
    }
    out = dict(payload_np)
    for key in row_keys:
        if key not in out:
            continue
        arr = np.asarray(out[key])
        if arr.ndim >= 1 and int(arr.shape[0]) == row_count:
            out[key] = np.asarray(arr[keep_rows], dtype=arr.dtype)

    if "nn_md_unique_idx" in out and "md_unique_sequences" in out:
        md_idx = np.asarray(out["nn_md_unique_idx"], dtype=np.int32)
        used_md = np.unique(md_idx)
        md_map = {int(old): new for new, old in enumerate(used_md.tolist())}
        out["nn_md_unique_idx"] = np.asarray([md_map[int(v)] for v in md_idx.tolist()], dtype=np.int32)
        for key in {"md_unique_sequences", "md_unique_counts", "md_unique_rep_original_indices", "md_unique_rep_frame_indices"}:
            if key not in out:
                continue
            arr = np.asarray(out[key])
            if arr.ndim >= 1 and arr.shape[0] >= used_md.shape[0]:
                out[key] = np.asarray(arr[used_md], dtype=arr.dtype)

    out.pop("sample_inv", None)
    out.pop("md_inv", None)
    out["sampled_unique_row_indices"] = np.asarray(keep_rows, dtype=np.int32)
    out["sampled_unique_row_count"] = np.asarray([int(keep_rows.shape[0])], dtype=np.int32)
    out["original_unique_row_count"] = np.asarray([row_count], dtype=np.int32)
    out["downsampled"] = np.asarray([1], dtype=np.int32)
    return out


def downsample_model_energy_payload(payload_np: dict[str, np.ndarray], *, row_limit: int, seed: int) -> dict[str, np.ndarray]:
    energies = np.asarray(payload_np.get("energies", np.asarray([], dtype=float)))
    if energies.ndim != 1:
        return payload_np
    row_count = int(energies.shape[0])
    if row_limit <= 0 or row_count <= row_limit:
        payload_np["sampled_row_count"] = np.asarray([row_count], dtype=np.int32)
        payload_np["original_row_count"] = np.asarray([row_count], dtype=np.int32)
        payload_np["downsampled"] = np.asarray([0], dtype=np.int32)
        return payload_np
    rng = np.random.default_rng(int(seed))
    keep_rows = np.sort(rng.choice(row_count, size=int(row_limit), replace=False).astype(np.int32))
    out = dict(payload_np)
    for key in ("energies", "frame_indices", "node_energies", "edge_energies"):
        arr = np.asarray(out.get(key, np.asarray([])))
        if arr.ndim >= 1 and arr.shape[0] == row_count:
            out[key] = np.asarray(arr[keep_rows], dtype=arr.dtype)
    out["sampled_row_indices"] = np.asarray(keep_rows, dtype=np.int32)
    out["sampled_row_count"] = np.asarray([int(keep_rows.shape[0])], dtype=np.int32)
    out["original_row_count"] = np.asarray([row_count], dtype=np.int32)
    out["downsampled"] = np.asarray([1], dtype=np.int32)
    return out
