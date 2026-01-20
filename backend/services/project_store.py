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


# Allow overriding the data root (e.g., to point to a larger, persistent volume).
DATA_ROOT = Path(os.getenv("ALLOSKIN_DATA_ROOT", "/app/data"))
PROJECTS_DIR = DATA_ROOT / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
SelectionInput = Union[Dict[str, str], List[str]]


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

    def create_project(self, name: str, description: Optional[str] = None) -> ProjectMetadata:
        project_id = str(uuid.uuid4())
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
            systems.append(self._decode_system(payload))
        return systems

    def get_system(self, project_id: str, system_id: str) -> SystemMetadata:
        meta_path = self._system_meta_path(project_id, system_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"System '{system_id}' not found in project '{project_id}'.")
        payload = _read_json(meta_path)
        return self._decode_system(payload)

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
        encoded_states: Dict[str, Any] = {}
        for key, state in metadata.states.items():
            state_id = getattr(state, "state_id", None) or key
            state_dict = asdict(state)
            state_dict["state_id"] = state_id
            encoded_states[state_id] = state_dict
        payload["states"] = encoded_states
        return payload

    def create_system(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        residue_selections: Optional[SelectionInput] = None,
    ) -> SystemMetadata:
        project_meta = self.get_project(project_id)
        system_id = str(uuid.uuid4())
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
        tmp_dir = system_dir / "tmp"
        structures_dir.mkdir(parents=True, exist_ok=True)
        descriptors_dir.mkdir(parents=True, exist_ok=True)
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return {
            "system_dir": system_dir,
            "structures_dir": structures_dir,
            "descriptors_dir": descriptors_dir,
            "trajectories_dir": trajectories_dir,
            "tmp_dir": tmp_dir,
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
