"""
Metastable recomputation service.

This glues the VAMP/TICA-based metastable pipeline to stored descriptor files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from phase.analysis.vamp_pipeline import run_metastable_pipeline_for_system
from backend.services.project_store import ProjectStore, SystemMetadata, DescriptorState
from backend.services.state_utils import build_analysis_states


def _rel_or_none(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _collect_trajectory_specs(project_id: str, system: SystemMetadata, store: ProjectStore) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for state in system.states.values():
        if not state.descriptor_file:
            continue
        desc_path = store.resolve_path(project_id, system.system_id, state.descriptor_file)
        traj_path = store.resolve_path(project_id, system.system_id, state.trajectory_file) if state.trajectory_file else None
        pdb_path = store.resolve_path(project_id, system.system_id, state.pdb_file) if state.pdb_file else None
        specs.append(
            {
                "trajectory_id": state.state_id,
                "macro_state": state.name,
                "macro_state_id": state.state_id,
                "descriptor_path": str(desc_path),
                "trajectory_path": str(traj_path) if traj_path and traj_path.exists() else None,
                "topology_path": str(pdb_path) if pdb_path and pdb_path.exists() else None,
            }
        )
    return specs


def recompute_metastable_states(
    project_id: str,
    system_id: str,
    *,
    n_microstates: int = 20,
    k_meta_min: int = 1,
    k_meta_max: int = 4,
    tica_lag_frames: int = 5,
    tica_dim: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Run metastable clustering for all descriptor-ready trajectories in a system.
    """
    store = ProjectStore()
    system = store.get_system(project_id, system_id)
    descriptor_states = [s for s in system.states.values() if s.descriptor_file]
    if not descriptor_states:
        raise ValueError("No descriptor-ready states available for metastable analysis.")

    dirs = store.ensure_directories(project_id, system_id)
    system_dir = dirs["system_dir"]
    output_dir = system_dir / "metastable"

    specs = _collect_trajectory_specs(project_id, system, store)
    if not specs:
        raise ValueError("No trajectories with descriptors found for metastable analysis.")

    results = run_metastable_pipeline_for_system(
        specs,
        output_dir=output_dir,
        n_microstates=n_microstates,
        k_meta_min=k_meta_min,
        k_meta_max=k_meta_max,
        tica_lag_frames=tica_lag_frames,
        tica_dim=tica_dim,
        random_state=random_state,
    )

    # Update per-trajectory metastable label files
    for macro_res in results.get("macro_results", []):
        label_map = macro_res.get("labels_per_trajectory") or {}
        for traj_id, label_path in label_map.items():
            state = system.states.get(traj_id)
            if not state:
                continue
            rel_label = _rel_or_none(Path(label_path), system_dir)
            state.metastable_labels_file = rel_label

    # Collect metastable state summaries
    metastable_states: List[Dict[str, Any]] = []
    for macro_res in results.get("macro_results", []):
        for meta in macro_res.get("metastable_states", []):
            meta_copy = dict(meta)
            rep = meta_copy.get("representative_pdb")
            if rep:
                meta_copy["representative_pdb"] = _rel_or_none(Path(rep), system_dir)
            if "name" not in meta_copy:
                meta_copy["name"] = meta_copy.get("default_name")
            metastable_states.append(meta_copy)

    system.metastable_states = metastable_states
    system.metastable_model_dir = _rel_or_none(output_dir, system_dir)
    system.analysis_states = build_analysis_states(system)
    store.save_system(system)

    return {
        "metastable_states": metastable_states,
        "model_dir": system.metastable_model_dir,
    }
