"""
Persistent storage helpers for project and descriptor metadata.

This module centralizes all filesystem paths used to store projects,
systems, uploaded structures, and computed descriptor artifacts.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import shutil
import numpy as np


# Allow overriding the data root (e.g., to point to a larger, persistent volume).
DATA_ROOT = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
PROJECTS_DIR = DATA_ROOT / "projects"
SelectionInput = Union[Dict[str, str], List[str]]
CLUSTER_METADATA_FILENAME = "cluster_metadata.json"
MODEL_METADATA_FILENAME = "model_metadata.json"
SAMPLE_METADATA_FILENAME = "sample_metadata.json"
STATES_METADATA_FILENAME = "states_metadata.json"
METASTABLE_METADATA_FILENAME = "metastable_metadata.json"


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    tmp_path.replace(path)


def _utc_now() -> str:
    return datetime.utcnow().isoformat()


def _read_cluster_npz_metadata(cluster_npz: Path) -> Dict[str, Any]:
    """Best-effort metadata extraction from cluster.npz when sidecar JSON is missing."""
    if not cluster_npz.exists():
        return {}
    try:
        with np.load(cluster_npz, allow_pickle=False) as data:
            if "metadata_json" not in data:
                return {}
            raw = data["metadata_json"]
            try:
                if isinstance(raw, np.ndarray):
                    raw = raw.item()
            except Exception:
                return {}
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            if not isinstance(raw, str) or not raw.strip():
                return {}
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


@dataclass
class DescriptorState:
    """Metadata about one state inside a system."""

    state_id: str
    name: str
    pdb_file: Optional[str] = None
    descriptor_file: Optional[str] = None
    descriptor_metadata_file: Optional[str] = None
    trajectory_file: Optional[str] = None
    n_frames: int = 0
    stride: int = 1
    source_traj: Optional[str] = None
    slice_spec: Optional[str] = None
    resid_shift: int = 0
    residue_selection: Optional[str] = None
    residue_keys: List[str] = field(default_factory=list)
    residue_mapping: Dict[str, str] = field(default_factory=dict)
    metastable_labels_file: Optional[str] = None
    role: Optional[str] = None  # Legacy compatibility
    created_at: str = field(default_factory=_utc_now)


@dataclass
class SystemMetadata:
    """Metadata stored alongside the descriptor NPZ files."""

    system_id: str
    project_id: str
    name: str
    description: Optional[str]
    created_at: str
    status: str = "processing"
    macro_locked: bool = False
    metastable_locked: bool = False
    analysis_mode: Optional[str] = None
    residue_selections: Optional[SelectionInput] = None
    residue_selections_mapping: Dict[str, str] = field(default_factory=dict)
    descriptor_keys: List[str] = field(default_factory=list)
    descriptor_metadata_file: Optional[str] = None
    metastable_model_dir: Optional[str] = None
    metastable_states: List[Dict[str, Any]] = field(default_factory=list)
    metastable_clusters: List[Dict[str, Any]] = field(default_factory=list)
    analysis_states: List[Dict[str, Any]] = field(default_factory=list)
    states: Dict[str, DescriptorState] = field(default_factory=dict)


@dataclass
class ProjectMetadata:
    """High-level information about a project."""

    project_id: str
    name: str
    description: Optional[str]
    created_at: str
    systems: List[str] = field(default_factory=list)


class ProjectStore:
    """Handles creation and persistence of projects and systems."""

    def __init__(self, base_dir: Path = PROJECTS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Project-level helpers
    # ------------------------------------------------------------------

    def _project_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id

    def _project_meta_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ProjectMetadata:
        project_id = project_id or str(uuid.uuid4())
        project_dir = self._project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=False)

        metadata = ProjectMetadata(
            project_id=project_id,
            name=name,
            description=description,
            created_at=_utc_now(),
            systems=[],
        )
        _write_json(self._project_meta_path(project_id), asdict(metadata))
        return metadata

    def list_projects(self) -> List[ProjectMetadata]:
        projects: List[ProjectMetadata] = []
        for project_dir in sorted(self.base_dir.glob("*")):
            meta_path = project_dir / "project.json"
            if not meta_path.exists():
                continue
            payload = _read_json(meta_path)
            projects.append(ProjectMetadata(**payload))
        return projects

    def get_project(self, project_id: str) -> ProjectMetadata:
        meta_path = self._project_meta_path(project_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        payload = _read_json(meta_path)
        return ProjectMetadata(**payload)

    def _save_project(self, metadata: ProjectMetadata) -> None:
        _write_json(self._project_meta_path(metadata.project_id), asdict(metadata))

    # ------------------------------------------------------------------
    # System-level helpers
    # ------------------------------------------------------------------

    def _systems_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "systems"

    def _system_dir(self, project_id: str, system_id: str) -> Path:
        return self._systems_dir(project_id) / system_id

    def _system_meta_path(self, project_id: str, system_id: str) -> Path:
        return self._system_dir(project_id, system_id) / "system.json"

    def list_systems(self, project_id: str) -> List[SystemMetadata]:
        systems: List[SystemMetadata] = []
        systems_root = self._systems_dir(project_id)
        if not systems_root.exists():
            return systems
        for sys_dir in sorted(systems_root.glob("*")):
            meta_path = sys_dir / "system.json"
            if not meta_path.exists():
                continue
            payload = _read_json(meta_path)
            system_meta = self._decode_system(payload)
            self._hydrate_system(system_meta)
            systems.append(system_meta)
        return systems

    def get_system(self, project_id: str, system_id: str) -> SystemMetadata:
        meta_path = self._system_meta_path(project_id, system_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"System '{system_id}' not found in project '{project_id}'.")
        payload = _read_json(meta_path)
        system_meta = self._decode_system(payload)
        self._hydrate_system(system_meta)
        return system_meta

    def _decode_system(self, payload: Dict[str, Any]) -> SystemMetadata:
        decoded_states: Dict[str, DescriptorState] = {}
        for key, state_payload in (payload.get("states") or {}).items():
            # Backward compatibility with legacy payloads that keyed states by role.
            state_id = state_payload.get("state_id") or key
            name = state_payload.get("name") or state_payload.get("role") or state_id
            merged_payload = {
                **state_payload,
                "state_id": state_id,
                "name": name,
            }
            decoded_states[state_id] = DescriptorState(**merged_payload)

        payload = {**payload, "states": decoded_states}
        return SystemMetadata(**payload)

    def _encode_system(self, metadata: SystemMetadata) -> Dict[str, Any]:
        payload = asdict(metadata)
        # States are now hydrated from folder contents (structures/descriptors/trajectories)
        # and not persisted in system.json.
        for key in (
            "descriptor_keys",
            "analysis_states",
            "metastable_clusters",
            "states",
            "metastable_states",
            "metastable_model_dir",
            "metastable_locked",
            "analysis_mode",
        ):
            payload.pop(key, None)
        return payload

    def create_system(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        residue_selections: Optional[SelectionInput] = None,
        system_id: Optional[str] = None,
    ) -> SystemMetadata:
        project_meta = self.get_project(project_id)
        system_id = system_id or str(uuid.uuid4())
        system_dir = self._system_dir(project_id, system_id)
        system_dir.mkdir(parents=True, exist_ok=False)
        (system_dir / "structures").mkdir(parents=True, exist_ok=True)
        (system_dir / "descriptors").mkdir(parents=True, exist_ok=True)

        metadata = SystemMetadata(
            system_id=system_id,
            project_id=project_id,
            name=name or f"System {system_id[:8]}",
            description=description,
            created_at=_utc_now(),
            residue_selections=residue_selections,
            states={},
        )
        _write_json(self._system_meta_path(project_id, system_id), self._encode_system(metadata))

        project_meta.systems.append(system_id)
        self._save_project(project_meta)

        return metadata

    def save_system(self, metadata: SystemMetadata) -> None:
        self._sync_states_metadata(metadata)
        self._sync_metastable_metadata(metadata)
        self._sync_cluster_metadata(metadata)
        meta_path = self._system_meta_path(metadata.project_id, metadata.system_id)
        _write_json(meta_path, self._encode_system(metadata))

    def rename_state(self, project_id: str, system_id: str, state_id: str, new_name: str) -> SystemMetadata:
        system_meta = self.get_system(project_id, system_id)
        if state_id not in system_meta.states:
            raise FileNotFoundError(f"State '{state_id}' not found in system '{system_id}'.")
        
        system_meta.states[state_id].name = new_name
        self.save_system(system_meta)
        return system_meta

    # ------------------------------------------------------------------
    # Filesystem utilities
    # ------------------------------------------------------------------

    def ensure_directories(self, project_id: str, system_id: str) -> Dict[str, Path]:
        system_dir = self._system_dir(project_id, system_id)
        structures_dir = system_dir / "structures"
        descriptors_dir = system_dir / "descriptors"
        trajectories_dir = system_dir / "trajectories"
        clusters_dir = system_dir / "clusters"
        tmp_dir = system_dir / "tmp"
        structures_dir.mkdir(parents=True, exist_ok=True)
        descriptors_dir.mkdir(parents=True, exist_ok=True)
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        clusters_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return {
            "system_dir": system_dir,
            "structures_dir": structures_dir,
            "descriptors_dir": descriptors_dir,
            "trajectories_dir": trajectories_dir,
            "clusters_dir": clusters_dir,
            "tmp_dir": tmp_dir,
        }

    def ensure_cluster_directories(
        self,
        project_id: str,
        system_id: str,
        cluster_id: str,
    ) -> Dict[str, Path]:
        dirs = self.ensure_directories(project_id, system_id)
        clusters_dir = dirs["clusters_dir"]
        cluster_dir = clusters_dir / cluster_id
        potts_models_dir = cluster_dir / "potts_models"
        samples_dir = cluster_dir / "samples"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        potts_models_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        return {
            "system_dir": dirs["system_dir"],
            "clusters_dir": clusters_dir,
            "cluster_dir": cluster_dir,
            "potts_models_dir": potts_models_dir,
            "samples_dir": samples_dir,
        }

    def _cluster_dir(self, project_id: str, system_id: str, cluster_id: str) -> Path:
        return self._system_dir(project_id, system_id) / "clusters" / cluster_id

    def _cluster_metadata_path(self, project_id: str, system_id: str, cluster_id: str) -> Path:
        return self._cluster_dir(project_id, system_id, cluster_id) / CLUSTER_METADATA_FILENAME

    def _model_metadata_path(self, project_id: str, system_id: str, cluster_id: str, model_id: str) -> Path:
        return self._cluster_dir(project_id, system_id, cluster_id) / "potts_models" / model_id / MODEL_METADATA_FILENAME

    def _sample_metadata_path(self, project_id: str, system_id: str, cluster_id: str, sample_id: str) -> Path:
        return self._cluster_dir(project_id, system_id, cluster_id) / "samples" / sample_id / SAMPLE_METADATA_FILENAME

    def _states_metadata_path(self, project_id: str, system_id: str) -> Path:
        return self._system_dir(project_id, system_id) / STATES_METADATA_FILENAME

    def _metastable_metadata_path(self, project_id: str, system_id: str) -> Path:
        return self._system_dir(project_id, system_id) / METASTABLE_METADATA_FILENAME

    def list_cluster_entries(self, project_id: str, system_id: str) -> List[Dict[str, Any]]:
        system_dir = self._system_dir(project_id, system_id)
        clusters_dir = system_dir / "clusters"
        if not clusters_dir.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for cluster_dir in sorted(p for p in clusters_dir.iterdir() if p.is_dir()):
            meta_path = cluster_dir / CLUSTER_METADATA_FILENAME
            cluster_npz = cluster_dir / "cluster.npz"
            meta: Dict[str, Any]
            if meta_path.exists():
                try:
                    meta = _read_json(meta_path)
                except Exception:
                    meta = {}
            else:
                meta = {}

            if not meta and cluster_npz.exists():
                # Backward compatibility: discover clusters from folder + cluster.npz even
                # when sidecar metadata was never created.
                npz_meta = _read_cluster_npz_metadata(cluster_npz)
                cluster_id = cluster_dir.name
                selected_ids = npz_meta.get("selected_state_ids") or npz_meta.get("selected_metastable_ids") or []
                meta = {
                    "cluster_id": cluster_id,
                    "name": npz_meta.get("cluster_name") or npz_meta.get("name") or cluster_id,
                    "status": "finished",
                    "progress": 100,
                    "status_message": "Complete",
                    "path": str(cluster_npz.relative_to(system_dir)),
                    "generated_at": npz_meta.get("generated_at"),
                    "state_ids": selected_ids,
                    "metastable_ids": selected_ids,
                    "contact_edge_count": npz_meta.get("contact_edge_count"),
                    "cluster_algorithm": npz_meta.get("cluster_algorithm") or "density_peaks",
                    "algorithm_params": npz_meta.get("cluster_params") or {},
                }
                try:
                    _write_json(meta_path, meta)
                except Exception:
                    pass

            if not meta:
                continue
            if isinstance(meta, dict):
                cleaned = False
                if "assigned_state_paths" in meta:
                    meta.pop("assigned_state_paths", None)
                    cleaned = True
                if "assigned_metastable_paths" in meta:
                    meta.pop("assigned_metastable_paths", None)
                    cleaned = True
                if cleaned:
                    try:
                        _write_json(meta_path, meta)
                    except Exception:
                        pass
            cluster_id = meta.get("cluster_id") or cluster_dir.name
            meta["cluster_id"] = cluster_id
            if not meta.get("path"):
                if cluster_npz.exists():
                    try:
                        meta["path"] = str(cluster_npz.relative_to(system_dir))
                    except Exception:
                        meta["path"] = str(cluster_npz)
            meta["potts_models"] = self.list_potts_models(project_id, system_id, cluster_id)
            meta["samples"] = self.list_samples(project_id, system_id, cluster_id)
            entries.append(meta)
        return entries

    def get_cluster_entry(self, project_id: str, system_id: str, cluster_id: str) -> Dict[str, Any]:
        entries = self.list_cluster_entries(project_id, system_id)
        entry = next((c for c in entries if c.get("cluster_id") == cluster_id), None)
        if not entry:
            raise FileNotFoundError(f"Cluster '{cluster_id}' not found.")
        return entry

    def list_potts_models(self, project_id: str, system_id: str, cluster_id: str) -> List[Dict[str, Any]]:
        system_dir = self._system_dir(project_id, system_id)
        models_dir = self._cluster_dir(project_id, system_id, cluster_id) / "potts_models"
        if not models_dir.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for model_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
            meta_path = model_dir / MODEL_METADATA_FILENAME
            if not meta_path.exists():
                continue
            try:
                meta = _read_json(meta_path)
            except Exception:
                continue
            model_id = meta.get("model_id") or model_dir.name
            meta["model_id"] = model_id
            if not meta.get("path"):
                npz_files = sorted(p for p in model_dir.iterdir() if p.is_file() and p.suffix == ".npz")
                if len(npz_files) == 1:
                    try:
                        meta["path"] = str(npz_files[0].relative_to(system_dir))
                    except Exception:
                        meta["path"] = str(npz_files[0])
            entries.append(meta)
        return entries

    def list_samples(self, project_id: str, system_id: str, cluster_id: str) -> List[Dict[str, Any]]:
        system_dir = self._system_dir(project_id, system_id)
        samples_dir = self._cluster_dir(project_id, system_id, cluster_id) / "samples"
        if not samples_dir.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for sample_dir in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
            meta_path = sample_dir / SAMPLE_METADATA_FILENAME
            if not meta_path.exists():
                continue
            try:
                meta = _read_json(meta_path)
            except Exception:
                continue
            sample_id = meta.get("sample_id") or sample_dir.name
            meta["sample_id"] = sample_id
            if not meta.get("path"):
                npz_files = sorted(p for p in sample_dir.iterdir() if p.is_file() and p.suffix == ".npz")
                if len(npz_files) == 1:
                    try:
                        meta["path"] = str(npz_files[0].relative_to(system_dir))
                    except Exception:
                        meta["path"] = str(npz_files[0])
            entries.append(meta)
        return entries

    def _hydrate_system(self, system_meta: SystemMetadata) -> None:
        state_sidecar = self._load_states_metadata(system_meta.project_id, system_meta.system_id)
        metastable_sidecar = self._load_metastable_metadata(system_meta.project_id, system_meta.system_id)
        if metastable_sidecar is not None:
            system_meta.metastable_states = list(metastable_sidecar.get("metastable_states") or [])
            system_meta.metastable_model_dir = metastable_sidecar.get("metastable_model_dir")
            system_meta.metastable_locked = bool(metastable_sidecar.get("metastable_locked", False))
            system_meta.analysis_mode = metastable_sidecar.get("analysis_mode")

        system_meta.states = self._scan_states_from_disk(
            system_meta.project_id,
            system_meta.system_id,
            state_sidecar if state_sidecar is not None else (system_meta.states or {}),
        )
        descriptor_keys = set()
        for state in system_meta.states.values():
            residue_keys = getattr(state, "residue_keys", None) or []
            descriptor_keys.update(residue_keys)
        system_meta.descriptor_keys = sorted(descriptor_keys)
        try:
            from phase.services.state_utils import build_analysis_states
            system_meta.analysis_states = build_analysis_states(system_meta)
        except Exception:
            system_meta.analysis_states = []
        try:
            system_meta.metastable_clusters = self.list_cluster_entries(
                system_meta.project_id,
                system_meta.system_id,
            )
        except Exception:
            system_meta.metastable_clusters = []

    def _load_states_metadata(self, project_id: str, system_id: str) -> Optional[Dict[str, DescriptorState]]:
        path = self._states_metadata_path(project_id, system_id)
        if not path.exists():
            return None
        try:
            payload = _read_json(path)
        except Exception:
            return None
        raw_states = payload.get("states")
        if not isinstance(raw_states, dict):
            raw_states = payload if isinstance(payload, dict) else {}
        out: Dict[str, DescriptorState] = {}
        for sid, raw in raw_states.items():
            if not isinstance(raw, dict):
                continue
            state_id = str(raw.get("state_id") or sid)
            name = str(raw.get("name") or state_id)
            merged = {**raw, "state_id": state_id, "name": name}
            try:
                out[state_id] = DescriptorState(**merged)
            except Exception:
                continue
        return out

    def _load_metastable_metadata(self, project_id: str, system_id: str) -> Optional[Dict[str, Any]]:
        path = self._metastable_metadata_path(project_id, system_id)
        if not path.exists():
            return None
        try:
            payload = _read_json(path)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _scan_states_from_disk(
        self,
        project_id: str,
        system_id: str,
        existing_states: Dict[str, DescriptorState],
    ) -> Dict[str, DescriptorState]:
        dirs = self.ensure_directories(project_id, system_id)
        system_dir = dirs["system_dir"]
        structures_dir = dirs["structures_dir"]
        descriptors_dir = dirs["descriptors_dir"]
        trajectories_dir = dirs["trajectories_dir"]

        states: Dict[str, DescriptorState] = {}
        for sid, st in (existing_states or {}).items():
            state_id = getattr(st, "state_id", None) or sid
            states[state_id] = st if isinstance(st, DescriptorState) else DescriptorState(**dict(st))

        discovered_ids: set[str] = set()
        for p in structures_dir.glob("*.pdb"):
            discovered_ids.add(p.stem)
        for p in descriptors_dir.glob("*_descriptors.npz"):
            discovered_ids.add(p.name[: -len("_descriptors.npz")])
        for p in descriptors_dir.glob("*_descriptor_metadata.json"):
            discovered_ids.add(p.name[: -len("_descriptor_metadata.json")])

        for state_id in sorted(discovered_ids):
            st = states.get(state_id)
            if not st:
                st = DescriptorState(state_id=state_id, name=state_id)
                states[state_id] = st

            if not st.name:
                st.name = state_id

            pdb_path = structures_dir / f"{state_id}.pdb"
            if pdb_path.exists():
                st.pdb_file = str(pdb_path.relative_to(system_dir))

            desc_npz = descriptors_dir / f"{state_id}_descriptors.npz"
            if desc_npz.exists():
                st.descriptor_file = str(desc_npz.relative_to(system_dir))

            desc_meta = descriptors_dir / f"{state_id}_descriptor_metadata.json"
            if desc_meta.exists():
                st.descriptor_metadata_file = str(desc_meta.relative_to(system_dir))
                try:
                    payload = _read_json(desc_meta)
                except Exception:
                    payload = {}
                keys = payload.get("descriptor_keys")
                if isinstance(keys, list):
                    st.residue_keys = [str(v) for v in keys]
                mapping = payload.get("residue_mapping")
                if isinstance(mapping, dict):
                    st.residue_mapping = {str(k): str(v) for k, v in mapping.items()}
                n_frames = payload.get("n_frames")
                if isinstance(n_frames, (int, float)):
                    st.n_frames = max(0, int(n_frames))
                selection = payload.get("residue_selection")
                if isinstance(selection, str):
                    st.residue_selection = selection
                resid_shift = payload.get("resid_shift")
                if isinstance(resid_shift, (int, float)):
                    st.resid_shift = int(resid_shift)
                state_name = payload.get("state_name")
                if isinstance(state_name, str) and state_name.strip():
                    st.name = state_name.strip()

            if not st.trajectory_file:
                traj_candidates = sorted(p for p in trajectories_dir.glob(f"{state_id}.*") if p.is_file())
                if traj_candidates:
                    st.trajectory_file = str(traj_candidates[0].relative_to(system_dir))
                    st.source_traj = traj_candidates[0].name

        return states

    def _sync_states_metadata(self, metadata: SystemMetadata) -> None:
        path = self._states_metadata_path(metadata.project_id, metadata.system_id)
        states_payload: Dict[str, Any] = {}
        for sid, st in (metadata.states or {}).items():
            state_id = getattr(st, "state_id", None) or sid
            states_payload[state_id] = asdict(st)
            states_payload[state_id]["state_id"] = state_id
        _write_json(path, {"states": states_payload})

    def _sync_metastable_metadata(self, metadata: SystemMetadata) -> None:
        path = self._metastable_metadata_path(metadata.project_id, metadata.system_id)
        payload = {
            "metastable_states": metadata.metastable_states or [],
            "metastable_model_dir": metadata.metastable_model_dir,
            "metastable_locked": bool(metadata.metastable_locked),
            "analysis_mode": metadata.analysis_mode,
        }
        _write_json(path, payload)

    def _sync_cluster_metadata(self, metadata: SystemMetadata) -> None:
        project_id = metadata.project_id
        system_id = metadata.system_id
        system_dir = self._system_dir(project_id, system_id)
        clusters_dir = system_dir / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)
        clusters = metadata.metastable_clusters or []
        expected_clusters = set()
        for entry in clusters:
            if not isinstance(entry, dict):
                continue
            cluster_id = entry.get("cluster_id") or entry.get("id")
            if not cluster_id:
                continue
            expected_clusters.add(cluster_id)
            cluster_meta = {k: v for k, v in entry.items() if k not in ("potts_models", "samples")}
            cluster_meta.pop("assigned_state_paths", None)
            cluster_meta.pop("assigned_metastable_paths", None)
            cluster_meta["cluster_id"] = cluster_id
            if not cluster_meta.get("path"):
                cluster_npz = self._cluster_dir(project_id, system_id, cluster_id) / "cluster.npz"
                if cluster_npz.exists():
                    try:
                        cluster_meta["path"] = str(cluster_npz.relative_to(system_dir))
                    except Exception:
                        cluster_meta["path"] = str(cluster_npz)
            meta_path = self._cluster_metadata_path(project_id, system_id, cluster_id)
            _write_json(meta_path, cluster_meta)

            self._sync_model_metadata(project_id, system_id, cluster_id, entry.get("potts_models") or [])
            self._sync_sample_metadata(project_id, system_id, cluster_id, entry.get("samples") or [])

        for meta_path in clusters_dir.glob(f"*/{CLUSTER_METADATA_FILENAME}"):
            cluster_id = meta_path.parent.name
            if cluster_id not in expected_clusters:
                try:
                    meta_path.unlink()
                except Exception:
                    pass

    def _sync_model_metadata(
        self,
        project_id: str,
        system_id: str,
        cluster_id: str,
        models: List[Dict[str, Any]],
    ) -> None:
        model_root = self._cluster_dir(project_id, system_id, cluster_id) / "potts_models"
        model_root.mkdir(parents=True, exist_ok=True)
        expected = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = model.get("model_id") or model.get("id")
            if not model_id:
                continue
            expected.add(str(model_id))
            model_meta = dict(model)
            model_meta["model_id"] = str(model_id)
            meta_path = self._model_metadata_path(project_id, system_id, cluster_id, str(model_id))
            _write_json(meta_path, model_meta)
        for meta_path in model_root.glob(f"*/{MODEL_METADATA_FILENAME}"):
            model_id = meta_path.parent.name
            if model_id not in expected:
                try:
                    meta_path.unlink()
                except Exception:
                    pass

    def _sync_sample_metadata(
        self,
        project_id: str,
        system_id: str,
        cluster_id: str,
        samples: List[Dict[str, Any]],
    ) -> None:
        sample_root = self._cluster_dir(project_id, system_id, cluster_id) / "samples"
        sample_root.mkdir(parents=True, exist_ok=True)
        expected = set()
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sample_id = sample.get("sample_id") or sample.get("id")
            if not sample_id:
                continue
            expected.add(str(sample_id))
            sample_meta = dict(sample)
            sample_meta["sample_id"] = str(sample_id)
            meta_path = self._sample_metadata_path(project_id, system_id, cluster_id, str(sample_id))
            _write_json(meta_path, sample_meta)
        for meta_path in sample_root.glob(f"*/{SAMPLE_METADATA_FILENAME}"):
            sample_id = meta_path.parent.name
            if sample_id not in expected:
                try:
                    meta_path.unlink()
                except Exception:
                    pass

    def ensure_results_directories(self, project_id: str, system_id: str) -> Dict[str, Path]:
        system_dir = self._system_dir(project_id, system_id)
        results_dir = system_dir / "results"
        jobs_dir = results_dir / "jobs"
        results_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        return {
            "system_dir": system_dir,
            "results_dir": results_dir,
            "jobs_dir": jobs_dir,
        }

    def resolve_path(self, project_id: str, system_id: str, relative_path: str) -> Path:
        return self._system_dir(project_id, system_id) / relative_path

    def delete_system(self, project_id: str, system_id: str) -> None:
        system_dir = self._system_dir(project_id, system_id)
        if not system_dir.exists():
            raise FileNotFoundError(f"System '{system_id}' not found in project '{project_id}'.")
        shutil.rmtree(system_dir)
        # Update project metadata
        project_meta = self.get_project(project_id)
        if system_id in project_meta.systems:
            project_meta.systems.remove(system_id)
            self._save_project(project_meta)

    def delete_project(self, project_id: str) -> None:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        shutil.rmtree(project_dir)
