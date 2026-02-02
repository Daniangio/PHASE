"""
Shared API utilities and dependencies for v1 routers.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from phase.workflows.macro_states import (
    build_state_descriptors as build_state_descriptors_sync,
    refresh_system_metadata,
    update_system_status,
)
from backend.services.project_store import (
    DescriptorState,
    ProjectMetadata,
    ProjectStore,
    SelectionInput,
    SystemMetadata,
)


DATA_ROOT = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
project_store = ProjectStore()


def serialize_project(meta: ProjectMetadata) -> Dict[str, Any]:
    return asdict(meta)


def serialize_system(meta: SystemMetadata) -> Dict[str, Any]:
    return asdict(meta)


async def stream_upload(upload: UploadFile, destination: Path) -> None:
    """Writes an UploadFile to disk in streaming fashion."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as buffer:
        while chunk := await upload.read(1024 * 1024):
            buffer.write(chunk)


def normalize_stride(label: str, raw_value: int) -> int:
    try:
        stride = int(raw_value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid stride value '{raw_value}' for {label}.")
    if stride <= 0:
        raise HTTPException(status_code=400, detail=f"Stride for {label} must be >= 1.")
    return stride


def stride_to_slice(stride: int) -> Optional[str]:
    return f"::{stride}" if stride > 1 else None


def parse_residue_selections(raw_value: Optional[str], expect_json: bool = False):
    """
    Accepts either legacy JSON objects/arrays or newline-delimited text inputs.
    Returns None, a dict, or a list of selection strings.
    """
    if not raw_value:
        return None
    stripped = raw_value.strip()
    if not stripped:
        return None

    should_parse_json = expect_json or stripped[0] in ("{", "[")

    if should_parse_json:
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if expect_json:
                raise HTTPException(status_code=400, detail="Invalid JSON for residue selections.") from exc
        else:
            if isinstance(data, dict):
                cleaned = {k: str(v).strip() for k, v in data.items() if isinstance(v, str) and v.strip()}
                return cleaned or None
            if isinstance(data, list):
                lines = [str(item).strip() for item in data if str(item).strip()]
                return lines or None
            raise HTTPException(status_code=400, detail="Residue selections JSON must be an object or array.")

    lines = [line.strip() for line in raw_value.splitlines() if line.strip()]
    return lines or None


def get_state_or_404(system_meta: SystemMetadata, state_id: str) -> DescriptorState:
    state = system_meta.states.get(state_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"State '{state_id}' not found for system '{system_meta.system_id}'.")
    return state


def ensure_not_macro_locked(system_meta: SystemMetadata):
    if getattr(system_meta, "macro_locked", False):
        raise HTTPException(status_code=400, detail="System macro-states are locked; no further edits allowed.")


def ensure_not_metastable_locked(system_meta: SystemMetadata):
    if getattr(system_meta, "metastable_locked", False):
        raise HTTPException(status_code=400, detail="Metastable states are locked; recomputation is disabled.")


async def build_state_descriptors(
    project_id: str,
    system_meta: SystemMetadata,
    state_meta: DescriptorState,
    *,
    residue_filter: Optional[str] = None,
    traj_path_override: Optional[Path] = None,
) -> SystemMetadata:
    try:
        return await run_in_threadpool(
            functools.partial(
                build_state_descriptors_sync,
                project_store,
                project_id,
                system_meta,
                state_meta,
                residue_filter=residue_filter,
                traj_path_override=traj_path_override,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def pick_state_pair(system_meta: SystemMetadata, state_a_id: Optional[str], state_b_id: Optional[str]):
    if state_a_id and state_b_id:
        if state_a_id == state_b_id:
            raise HTTPException(status_code=400, detail="Select two different states.")
        return get_state_or_404(system_meta, state_a_id), get_state_or_404(system_meta, state_b_id)

    descriptor_states = [s for s in system_meta.states.values() if s.descriptor_file]
    if len(descriptor_states) < 2:
        raise HTTPException(status_code=400, detail="At least two states with descriptors are required.")
    return descriptor_states[0], descriptor_states[1]


def ensure_system_ready(project_id: str, system_id: str, state_a_id: Optional[str], state_b_id: Optional[str]):
    try:
        system = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{system_id}' not found in project '{project_id}'.",
        )
    state_a, state_b = pick_state_pair(system, state_a_id, state_b_id)
    if not state_a.descriptor_file or not state_b.descriptor_file:
        raise HTTPException(status_code=400, detail="Selected states do not have built descriptors.")
    if not state_a.residue_keys or not state_b.residue_keys:
        raise HTTPException(status_code=400, detail="Selected states are missing descriptor metadata.")
    return system, state_a, state_b


def get_cluster_entry(system: SystemMetadata, cluster_id: str) -> Dict[str, Any]:
    clusters = system.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Cluster NPZ not found.")
    return entry


def get_queue(request: Request):
    """Provides the RQ queue object."""
    if request.app.state.task_queue is None:
        raise HTTPException(status_code=503, detail="Worker queue not initialized. Check Redis connection.")
    return request.app.state.task_queue
