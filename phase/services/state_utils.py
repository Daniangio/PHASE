from __future__ import annotations

from typing import Any, Dict, List

from phase.services.project_store import DescriptorState, SystemMetadata


def build_analysis_states(system_meta: SystemMetadata) -> List[Dict[str, Any]]:
    """
    Build a unified list of macro + metastable states for selection in analyses.
    """
    merged: List[Dict[str, Any]] = []
    for state_id, state in (system_meta.states or {}).items():
        if isinstance(state, DescriptorState) or (
            hasattr(state, "name")
            and hasattr(state, "pdb_file")
            and hasattr(state, "descriptor_file")
        ):
            merged.append(
                {
                    "state_id": state_id,
                    "name": getattr(state, "name", state_id),
                    "kind": "macro",
                    "pdb_file": getattr(state, "pdb_file", None),
                    "trajectory_file": getattr(state, "trajectory_file", None),
                    "descriptor_file": getattr(state, "descriptor_file", None),
                    "descriptor_metadata_file": getattr(state, "descriptor_metadata_file", None),
                    "n_frames": getattr(state, "n_frames", 0),
                    "stride": getattr(state, "stride", 1),
                    "slice_spec": getattr(state, "slice_spec", None),
                    "residue_selection": getattr(state, "residue_selection", None),
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
