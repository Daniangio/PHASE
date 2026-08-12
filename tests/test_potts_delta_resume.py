from pathlib import Path

from phase.scripts.potts_delta_fit import _resolve_resume_pair


class _Store:
    def __init__(self, system_dir: Path):
        self.system_dir = system_dir

    def resolve_path(self, project_id: str, system_id: str, value: str) -> Path:
        return self.system_dir / value


def _entry(model_id: str, name: str, path: str, kind: str, base: str, *, created: str, **params):
    return {
        "model_id": model_id,
        "name": name,
        "path": path,
        "created_at": created,
        "params": {
            "fit_mode": "delta",
            "delta_kind": kind,
            "base_model": base,
            "state_ids": ["inactive"],
            **params,
        },
    }


def test_resume_from_combined_resolves_delta_and_original_base(tmp_path):
    base_path = tmp_path / "models/base.npz"
    delta_path = tmp_path / "models/delta.npz"
    combined_path = tmp_path / "models/combined.npz"
    for path in (base_path, delta_path, combined_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    delta = _entry(
        "delta-id", "inactive (delta)", "models/delta.npz", "delta_patch", "models/base.npz",
        created="2026-01-01T00:00:00", combined_model_id="combined-id",
    )
    combined = _entry(
        "combined-id", "inactive (combined)", "models/combined.npz", "model_patch", "models/base.npz",
        created="2026-01-01T00:00:00", delta_model_id="delta-id",
    )

    resolved_delta, resolved_combined, resolved_base = _resolve_resume_pair(
        combined, [delta, combined], _Store(tmp_path), "project", "system"
    )

    assert resolved_delta is delta
    assert resolved_combined is combined
    assert resolved_base == base_path


def test_legacy_resume_pair_is_matched_by_name_and_state(tmp_path):
    for relative in ("base.npz", "delta.npz", "combined.npz"):
        (tmp_path / relative).touch()
    delta = _entry(
        "delta-id", "switch (delta)", "delta.npz", "delta_patch", "base.npz",
        created="2026-01-01T00:00:00",
    )
    combined = _entry(
        "combined-id", "switch (combined)", "combined.npz", "model_patch", "base.npz",
        created="2026-01-01T00:00:00",
    )

    resolved_delta, resolved_combined, resolved_base = _resolve_resume_pair(
        combined, [delta, combined], _Store(tmp_path), "project", "system"
    )

    assert resolved_delta is delta
    assert resolved_combined is combined
    assert resolved_base == tmp_path / "base.npz"
