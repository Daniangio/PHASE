import asyncio
import io
import json
import zipfile
from pathlib import Path

from fastapi import UploadFile

import backend.api.v1.routes.projects as projects


def _make_zip_with_project(tmp_path: Path, project_id: str, system_id: str, system_payload: dict) -> bytes:
    root = tmp_path / "src"
    projects_dir = root / "projects" / project_id / "systems" / system_id
    projects_dir.mkdir(parents=True, exist_ok=True)
    (projects_dir.parent.parent / "project.json").write_text(
        json.dumps({"project_id": project_id, "name": "Proj", "description": None, "created_at": "2026-02-02T00:00:00", "systems": [system_id]}, indent=2)
    )
    (projects_dir / "system.json").write_text(json.dumps(system_payload, indent=2))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in (root / "projects").rglob("*"):
            zf.write(path, path.relative_to(root))
    return buf.getvalue()


def test_restore_projects_preserves_system_json(tmp_path: Path, monkeypatch):
    project_id = "p1"
    system_id = "s1"
    system_payload = {
        "system_id": system_id,
        "project_id": project_id,
        "name": "System",
        "description": None,
        "created_at": "2026-02-02T00:00:00",
        "status": "ready",
        "macro_locked": True,
        "metastable_locked": False,
        "analysis_mode": "macro",
        "residue_selections": None,
        "residue_selections_mapping": {},
        "descriptor_metadata_file": None,
        "metastable_model_dir": None,
        "metastable_states": [],
        "states": {},
    }

    zip_bytes = _make_zip_with_project(tmp_path, project_id, system_id, system_payload)
    upload = UploadFile(filename="projects.zip", file=io.BytesIO(zip_bytes))

    data_root = tmp_path / "data"
    monkeypatch.setattr(projects, "DATA_ROOT", data_root)
    projects.project_store.base_dir = data_root / "projects"

    asyncio.run(projects.restore_projects(upload))

    restored_path = data_root / "projects" / project_id / "systems" / system_id / "system.json"
    assert restored_path.exists(), "system.json missing after restore"
    restored = json.loads(restored_path.read_text())
    assert restored == system_payload
