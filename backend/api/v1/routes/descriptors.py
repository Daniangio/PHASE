import json
from typing import Any, Dict, List, Optional

import MDAnalysis as mda
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.api.v1.common import get_state_or_404, project_store
from phase.io.descriptors import load_descriptor_npz


router = APIRouter()


@router.get(
    "/projects/{project_id}/systems/{system_id}/states/{state_id}/descriptors",
    summary="Preview descriptor angles for a state (for visualization)",
)
async def get_state_descriptors(
    project_id: str,
    system_id: str,
    state_id: str,
    residue_keys: Optional[str] = Query(
        None,
        description="Comma-separated residue keys to include; defaults to all keys for the state.",
    ),
    metastable_ids: Optional[str] = Query(
        None,
        description="Comma-separated metastable IDs to filter frames; defaults to all frames.",
    ),
    cluster_id: Optional[str] = Query(
        None,
        description="ID of a saved cluster NPZ to use for coloring (optional).",
    ),
    cluster_label_mode: str = Query(
        "halo",
        description="Cluster label mode for coloring: 'halo' (default) or 'assigned'.",
    ),
    cluster_variant_id: Optional[str] = Query(
        None,
        description="Cluster variant to use: 'original' (default) or a preview patch id.",
    ),
    max_points: int = Query(
        2000,
        ge=10,
        le=50000,
        description="Maximum number of points returned per residue (down-sampled evenly).",
    ),
):
    """
    Returns a down-sampled set of phi/psi/chi1 angles (in degrees) for the requested state.
    Intended for client-side scatter plotting; not for bulk export.
    """
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    state_meta = get_state_or_404(system, state_id)
    if not state_meta.descriptor_file:
        raise HTTPException(status_code=404, detail="No descriptors stored for this state.")

    descriptor_path = project_store.resolve_path(project_id, system_id, state_meta.descriptor_file)
    if not descriptor_path.exists():
        raise HTTPException(status_code=404, detail="Descriptor file missing on disk.")

    try:
        feature_dict = load_descriptor_npz(descriptor_path)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to load descriptor file: {exc}") from exc

    keys_to_use = list(feature_dict.keys())
    if residue_keys:
        requested = [key.strip() for key in residue_keys.split(",") if key.strip()]
        keys_to_use = [k for k in keys_to_use if k in requested]
        if not keys_to_use:
            raise HTTPException(status_code=400, detail="No matching residue keys found in descriptor file.")

    angles_payload: Dict[str, Any] = {}
    residue_labels: Dict[str, str] = {}
    sample_stride = 1

    # Try to resolve residue names from the stored PDB for nicer labels
    resname_map: Dict[int, str] = {}
    if state_meta.pdb_file:
        try:
            pdb_path = project_store.resolve_path(project_id, system_id, state_meta.pdb_file)
            if pdb_path.exists():
                u = mda.Universe(str(pdb_path))
                for res in u.residues:
                    resname_map[int(res.resid)] = str(res.resname).strip()
        except Exception:
            resname_map = {}

    # --- Metastable filtering ---
    metastable_filter_ids = []
    if metastable_ids:
        metastable_filter_ids = [mid.strip() for mid in metastable_ids.split(",") if mid.strip()]
    meta_id_to_index = {}
    index_to_meta_id = {}
    state_metastables = [
        m for m in (system.metastable_states or []) if m.get("macro_state_id") == state_id
    ]
    if state_metastables:
        for m in state_metastables:
            mid = m.get("metastable_id")
            if mid is None:
                continue
            meta_id_to_index[mid] = m.get("metastable_index")
            if m.get("metastable_index") is not None:
                index_to_meta_id[m.get("metastable_index")] = mid

    # --- Cluster NPZ / Samples ---
    cluster_npz = None
    cluster_meta = None
    merged_lookup = {}
    cluster_residue_indices: Dict[str, int] = {}
    merged_labels_arr: Optional[np.ndarray] = None
    cluster_legend: List[Dict[str, Any]] = []
    cluster_variants: List[Dict[str, Any]] = []
    selected_cluster_variant = "original"
    state_labels_arr: Optional[np.ndarray] = None
    state_labels_arr_halo: Optional[np.ndarray] = None
    state_labels_arr_assigned: Optional[np.ndarray] = None
    state_frame_lookup: Optional[Dict[int, int]] = None
    halo_summary = None
    label_mode = str(cluster_label_mode or "halo").lower()
    if label_mode not in {"halo", "assigned"}:
        raise HTTPException(status_code=400, detail="cluster_label_mode must be 'halo' or 'assigned'.")

    def _build_lookup(entry_key: str, npz_dict, keys_dict):
        """
        Build a mapping (state_id, frame_idx) -> row index.
        Falls back to sequential frame indices for backward-compatible NPZs that
        lack the explicit frame_indices array.
        """
        if not keys_dict or "frame_state_ids" not in keys_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster NPZ is missing frame index metadata for '{entry_key}'. Regenerate clusters.",
            )
        if keys_dict["frame_state_ids"] not in npz_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster NPZ is missing array '{keys_dict['frame_state_ids']}'. Regenerate clusters.",
            )

        frame_states = np.asarray(npz_dict[keys_dict["frame_state_ids"]])
        if "frame_indices" in keys_dict and keys_dict.get("frame_indices") in npz_dict:
            frame_indices = np.asarray(npz_dict[keys_dict["frame_indices"]])
        else:
            frame_indices = np.arange(len(frame_states), dtype=int)

        lookup = {}
        for i, (sid, fidx) in enumerate(zip(frame_states, frame_indices)):
            lookup[(str(sid), int(fidx))] = i
        return lookup

    if cluster_id:
        entry = next((c for c in system.metastable_clusters or [] if c.get("cluster_id") == cluster_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Cluster NPZ not found.")
        sample_entry = next(
            (
                s
                for s in entry.get("samples") or []
                if s.get("type") == "md_eval" and s.get("state_id") == state_meta.state_id
            ),
            None,
        )
        if sample_entry and sample_entry.get("path"):
            sample_path = project_store.resolve_path(project_id, system_id, sample_entry["path"])
            if sample_path.exists():
                sample_npz = np.load(sample_path, allow_pickle=True)
                labels_key = "labels_halo" if label_mode == "halo" else "labels"
                if labels_key in sample_npz:
                    state_labels_arr = sample_npz[labels_key]
                elif "labels" in sample_npz:
                    state_labels_arr = sample_npz["labels"]
                if "frame_indices" in sample_npz:
                    frame_indices = np.asarray(sample_npz["frame_indices"], dtype=int)
                    state_frame_lookup = {int(fidx): idx for idx, fidx in enumerate(frame_indices)}
                elif "assigned__frame_indices" in sample_npz:
                    frame_indices = np.asarray(sample_npz["assigned__frame_indices"], dtype=int)
                    state_frame_lookup = {int(fidx): idx for idx, fidx in enumerate(frame_indices)}
        rel_path = entry.get("path")
        if not rel_path:
            raise HTTPException(status_code=404, detail="Cluster NPZ path missing.")
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
        if not cluster_path.exists():
            raise HTTPException(status_code=404, detail="Cluster NPZ file missing.")
        cluster_npz = np.load(cluster_path, allow_pickle=True)
        try:
            cluster_meta = json.loads(cluster_npz["metadata_json"].item())
        except Exception:
            cluster_meta = None
        if not isinstance(cluster_meta, dict) or not cluster_meta:
            raise HTTPException(status_code=400, detail="Cluster NPZ missing metadata_json. Regenerate clusters.")
        if not cluster_residue_indices:
            cluster_res_keys = list(cluster_meta.get("residue_keys", []))
            cluster_residue_indices = {k: i for i, k in enumerate(cluster_res_keys)}
        patch_entries = [
            p for p in (cluster_meta.get("cluster_patches") or [])
            if isinstance(p, dict) and p.get("patch_id")
        ]
        cluster_variants = [{"id": "original", "label": "Original cluster", "status": "confirmed"}]
        for p in patch_entries:
            rid_keys = ((p.get("residues") or {}).get("keys") or [])
            cluster_variants.append(
                {
                    "id": str(p.get("patch_id")),
                    "label": str(p.get("name") or f"Patch {str(p.get('patch_id'))[:8]}"),
                    "status": str(p.get("status") or "preview"),
                    "residue_keys": [str(v) for v in rid_keys],
                    "created_at": p.get("created_at"),
                }
            )

        variant_entry = None
        requested_variant = str(cluster_variant_id).strip() if cluster_variant_id else "original"
        if requested_variant and requested_variant != "original":
            variant_entry = next((p for p in patch_entries if str(p.get("patch_id")) == requested_variant), None)
        selected_cluster_variant = requested_variant if (requested_variant == "original" or variant_entry) else "original"

        if variant_entry:
            predictions = variant_entry.get("predictions") or {}
            halo_summary = variant_entry.get("halo_summary") if isinstance(variant_entry, dict) else None
            merged_keys = (variant_entry.get("merged") or {}).get("npz_keys") or {}
        else:
            predictions = cluster_meta.get("predictions") or {}
            halo_summary = cluster_meta.get("halo_summary") if isinstance(cluster_meta, dict) else None
            merged_keys = cluster_meta.get("merged", {}).get("npz_keys", {})

        state_pred = predictions.get(f"state:{state_meta.state_id}")
        if state_labels_arr is None and isinstance(state_pred, dict):
            labels_key = state_pred.get("labels_halo")
            if label_mode == "assigned":
                labels_key = state_pred.get("labels_assigned") or labels_key
            if isinstance(labels_key, str) and labels_key in cluster_npz:
                state_labels_arr = cluster_npz[labels_key]
            halo_key = state_pred.get("labels_halo")
            assigned_key = state_pred.get("labels_assigned")
            if isinstance(halo_key, str) and halo_key in cluster_npz:
                state_labels_arr_halo = cluster_npz[halo_key]
            if isinstance(assigned_key, str) and assigned_key in cluster_npz:
                state_labels_arr_assigned = cluster_npz[assigned_key]
            frame_key = state_pred.get("frame_indices")
            if isinstance(frame_key, str) and frame_key in cluster_npz:
                try:
                    frame_indices = np.asarray(cluster_npz[frame_key], dtype=int)
                    state_frame_lookup = {int(fidx): idx for idx, fidx in enumerate(frame_indices)}
                except Exception:
                    state_frame_lookup = None
        if (
            not merged_keys
            or "labels" not in merged_keys
            or "frame_state_ids" not in merged_keys
            or "frame_indices" not in merged_keys
        ):
            raise HTTPException(status_code=400, detail="Cluster NPZ missing merged frame metadata. Regenerate clusters.")
        merged_lookup = _build_lookup("merged", cluster_npz, merged_keys)
        merged_label_key = merged_keys.get("labels_halo") or merged_keys.get("labels")
        if label_mode == "assigned":
            merged_label_key = merged_keys.get("labels_assigned") or merged_label_key
        if not isinstance(merged_label_key, str) or merged_label_key not in cluster_npz:
            raise HTTPException(status_code=400, detail="Selected cluster variant is missing merged labels.")
        merged_labels_arr = cluster_npz[merged_label_key]
        if not cluster_legend:
            unique_clusters = sorted({int(v) for v in np.unique(merged_labels_arr) if int(v) >= 0})
            cluster_legend = [{"id": c, "label": f"Merged c{c}"} for c in unique_clusters]

    # Shared frame selection (metastable filter + sampling) computed once
    labels_meta = None
    needs_meta_labels = bool(metastable_filter_ids) or bool(state_metastables)
    if needs_meta_labels:
        labels_meta = feature_dict.get("metastable_labels")
        if labels_meta is None and state_meta.metastable_labels_file:
            label_path = project_store.resolve_path(project_id, system_id, state_meta.metastable_labels_file)
            if label_path.exists():
                labels_meta = np.load(label_path)
        if labels_meta is None and metastable_filter_ids:
            raise HTTPException(status_code=400, detail="Metastable labels missing for this state.")

    first_arr = feature_dict[keys_to_use[0]]
    total_frames = first_arr.shape[0] if hasattr(first_arr, "shape") else 0
    indices = np.arange(total_frames)
    if metastable_filter_ids:
        selected_idx = {meta_id_to_index.get(mid) for mid in metastable_filter_ids if mid in meta_id_to_index}
        if not selected_idx:
            raise HTTPException(status_code=400, detail="Selected metastable IDs not found on this system.")
        mask = np.isin(labels_meta, list(selected_idx))
        indices = np.where(mask)[0]
        if indices.size == 0:
            raise HTTPException(status_code=400, detail="No frames match selected metastable states for this state.")

    n_frames_filtered = indices.size
    sample_stride = max(1, n_frames_filtered // max_points) if n_frames_filtered > max_points else 1
    sample_indices = indices[::sample_stride]
    n_frames_out = n_frames_filtered

    merged_rows_for_samples = None
    if cluster_npz is not None and merged_lookup:
        merged_rows_for_samples = np.array(
            [merged_lookup.get((state_meta.state_id, int(f)), -1) for f in sample_indices], dtype=int
        )

    for key in keys_to_use:
        arr = feature_dict[key]
        if arr.ndim != 3 or arr.shape[2] < 3:
            continue

        sampled = arr[sample_indices, 0, :]
        phi = (sampled[:, 0] * 180.0 / 3.141592653589793).tolist()
        psi = (sampled[:, 1] * 180.0 / 3.141592653589793).tolist()
        chi1 = (sampled[:, 2] * 180.0 / 3.141592653589793).tolist()
        angles_payload[key] = {"phi": phi, "psi": psi, "chi1": chi1}

        if cluster_npz is not None:
            res_idx = cluster_residue_indices.get(key, None)
            if res_idx is not None:
                if state_labels_arr is not None:
                    if state_frame_lookup is not None:
                        rows = np.array(
                            [state_frame_lookup.get(int(fidx), -1) for fidx in sample_indices],
                            dtype=int,
                        )
                        labels_for_res = np.full(sample_indices.shape[0], -1, dtype=int)
                        valid = rows >= 0
                        if np.any(valid):
                            labels_for_res[valid] = state_labels_arr[rows[valid], res_idx].astype(int)
                    else:
                        if sample_indices.size == 0 or sample_indices.max() < state_labels_arr.shape[0]:
                            labels_for_res = state_labels_arr[sample_indices, res_idx].astype(int)
                        else:
                            safe_rows = np.clip(sample_indices, 0, state_labels_arr.shape[0] - 1)
                            labels_for_res = state_labels_arr[safe_rows, res_idx].astype(int)
                    angles_payload[key]["cluster_labels"] = labels_for_res.tolist()
                elif merged_rows_for_samples is not None and merged_labels_arr is not None:
                    safe_rows = np.clip(merged_rows_for_samples, 0, merged_labels_arr.shape[0] - 1)
                    labels_for_res = merged_labels_arr[safe_rows, res_idx].astype(int)
                    labels_for_res[merged_rows_for_samples < 0] = -1
                    angles_payload[key]["cluster_labels"] = labels_for_res.tolist()

        label = key
        selection = (state_meta.residue_mapping or {}).get(key) or ""
        resid_tokens = [
            tok for tok in selection.replace("resid", "").split() if tok.strip().lstrip("-").isdigit()
        ]
        resid_val = int(resid_tokens[0]) if resid_tokens else None
        if resid_val is not None and resid_val in resname_map:
            label = f"{key}_{resname_map[resid_val]}"
        residue_labels[key] = label

    if not angles_payload:
        raise HTTPException(status_code=500, detail="Descriptor file contained no usable angle data.")

    halo_payload = {}
    if cluster_id and isinstance(entry, dict) and str(selected_cluster_variant or "original") == "original":
        cluster_residue_keys = (
            [str(v) for v in (cluster_meta or {}).get("residue_keys", [])]
            if isinstance(cluster_meta, dict)
            else []
        )
        if not cluster_residue_keys and cluster_residue_indices:
            cluster_residue_keys = [k for k, _ in sorted(cluster_residue_indices.items(), key=lambda item: item[1])]
        n_residues = len(cluster_residue_keys)
        state_label_map = {str(sid): str(state.name or sid) for sid, state in (system.states or {}).items()}
        meta_label_map = {
            str(m.get("metastable_id")): str(m.get("name") or m.get("default_name") or m.get("metastable_id"))
            for m in (system.metastable_states or [])
            if m.get("metastable_id")
        }
        md_samples = [
            s
            for s in (entry.get("samples") or [])
            if isinstance(s, dict) and str(s.get("type") or "") == "md_eval" and isinstance(s.get("path"), str)
        ]
        by_condition: Dict[str, Dict[str, Any]] = {}
        for sample in sorted(md_samples, key=lambda s: str(s.get("created_at") or "")):
            sid = sample.get("state_id")
            mid = sample.get("metastable_id")
            if sid:
                cond_id = f"state:{sid}"
                cond_label = state_label_map.get(str(sid), str(sample.get("name") or sid))
                cond_type = "macro"
            elif mid:
                cond_id = f"meta:{mid}"
                cond_label = meta_label_map.get(str(mid), str(sample.get("name") or mid))
                cond_type = "metastable"
            else:
                sample_id = str(sample.get("sample_id") or "")
                if not sample_id:
                    continue
                cond_id = f"sample:{sample_id}"
                cond_label = str(sample.get("name") or sample_id)
                cond_type = "md_eval"
            sample_path = project_store.resolve_path(project_id, system_id, sample["path"])
            if not sample_path.exists():
                try:
                    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
                    alt = cluster_dirs["cluster_dir"] / sample["path"]
                    if alt.exists():
                        sample_path = alt
                except Exception:
                    pass
            if not sample_path.exists():
                continue
            try:
                with np.load(sample_path, allow_pickle=True) as sample_npz:
                    if "labels_halo" in sample_npz:
                        labels_arr = np.asarray(sample_npz["labels_halo"], dtype=np.int32)
                    elif "labels" in sample_npz:
                        labels_arr = np.asarray(sample_npz["labels"], dtype=np.int32)
                    else:
                        continue
            except Exception:
                continue
            if labels_arr.ndim != 2:
                continue
            if n_residues <= 0:
                n_residues = int(labels_arr.shape[1])
                if not cluster_residue_keys:
                    cluster_residue_keys = [f"res_{i}" for i in range(n_residues)]
            if labels_arr.shape[1] != n_residues:
                continue
            halo_rate = (
                np.mean(labels_arr == -1, axis=0).astype(float, copy=False)
                if labels_arr.shape[0] > 0
                else np.full(n_residues, np.nan, dtype=float)
            )
            by_condition[cond_id] = {
                "id": cond_id,
                "label": str(cond_label),
                "type": cond_type,
                "rate": halo_rate,
            }
        if by_condition:
            rows = list(by_condition.values())
            halo_payload["halo_rate_residue_keys"] = cluster_residue_keys
            halo_payload["halo_rate_matrix"] = np.stack([row["rate"] for row in rows], axis=0).tolist()
            halo_payload["halo_rate_condition_ids"] = [row["id"] for row in rows]
            halo_payload["halo_rate_condition_labels"] = [row["label"] for row in rows]
            halo_payload["halo_rate_condition_types"] = [row["type"] for row in rows]

    # Backward-compatible fallback for clusters that do not yet have md_eval sample files.
    if not halo_payload and halo_summary and cluster_npz is not None:
        keys = (halo_summary or {}).get("npz_keys", {})
        matrix_key = keys.get("matrix") if isinstance(keys, dict) else None
        ids_key = keys.get("condition_ids") if isinstance(keys, dict) else None
        labels_key = keys.get("condition_labels") if isinstance(keys, dict) else None
        types_key = keys.get("condition_types") if isinstance(keys, dict) else None
        if isinstance(cluster_meta, dict):
            residue_keys = cluster_meta.get("residue_keys")
            if residue_keys:
                halo_payload["halo_rate_residue_keys"] = [str(v) for v in residue_keys]
        if matrix_key in cluster_npz:
            halo_payload["halo_rate_matrix"] = cluster_npz[matrix_key].tolist()
        if ids_key in cluster_npz:
            halo_payload["halo_rate_condition_ids"] = [str(v) for v in cluster_npz[ids_key].tolist()]
        if labels_key in cluster_npz:
            halo_payload["halo_rate_condition_labels"] = [str(v) for v in cluster_npz[labels_key].tolist()]
        if types_key in cluster_npz:
            halo_payload["halo_rate_condition_types"] = [str(v) for v in cluster_npz[types_key].tolist()]

    response = {
        "residue_keys": keys_to_use,
        "residue_mapping": state_meta.residue_mapping or {},
        "residue_labels": residue_labels,
        "n_frames": n_frames_out,
        "sample_stride": sample_stride,
        "angles": angles_payload,
        "cluster_legend": cluster_legend,
        "cluster_variants": cluster_variants,
        "cluster_variant_id": selected_cluster_variant,
        "metastable_labels": labels_meta[sample_indices].astype(int).tolist() if labels_meta is not None else [],
        "metastable_legend": [
            {
                "id": m.get("metastable_id"),
                "index": m.get("metastable_index"),
                "label": m.get("name") or m.get("default_name") or m.get("metastable_id"),
            }
            for m in state_metastables
            if m.get("metastable_index") is not None
        ],
        "metastable_filter_applied": bool(metastable_filter_ids),
        **halo_payload,
    }
    return response
