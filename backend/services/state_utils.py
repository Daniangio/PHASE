from __future__ import annotations

from typing import Any, Dict, List

from backend.services.project_store import DescriptorState, SystemMetadata


def build_analysis_states(system_meta: SystemMetadata) -> List[Dict[str, Any]]:
    """
    Build a unified list of macro + metastable states for selection in analyses.
    """
    merged: List[Dict[str, Any]] = []
    for state_id, state in (system_meta.states or {}).items():
        if isinstance(state, DescriptorState):
            merged.append(
                {
                    "state_id": state_id,
                    "name": state.name,
                    "kind": "macro",
                    "pdb_file": state.pdb_file,
                    "trajectory_file": state.trajectory_file,
                    "descriptor_file": state.descriptor_file,
                    "descriptor_metadata_file": state.descriptor_metadata_file,
                    "n_frames": state.n_frames,
                    "stride": state.stride,
                    "slice_spec": state.slice_spec,
                    "residue_selection": state.residue_selection,
                }
            )
        elif isinstance(state, dict):
            merged.append({**state, "state_id": state_id, "kind": state.get("kind") or "macro"})

    for meta in system_meta.metastable_states or []:
        meta_id = meta.get("metastable_id") or meta.get("id")
        if not meta_id:
            continue
        merged.append(
            {
                "state_id": str(meta_id),
                "name": meta.get("name") or meta.get("default_name") or str(meta_id),
                "kind": "metastable",
                "metastable_id": meta.get("metastable_id") or meta.get("id"),
                "metastable_index": meta.get("metastable_index"),
                "macro_state_id": meta.get("macro_state_id"),
                "macro_state": meta.get("macro_state"),
                "n_frames": meta.get("n_frames"),
            }
        )
    return merged
