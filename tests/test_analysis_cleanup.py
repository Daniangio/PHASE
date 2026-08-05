from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ["PHASE_DATA_ROOT"] = "/tmp/phase-test-data"

from backend.api.v1 import analysis_cleanup


class _FakeStore:
    def __init__(
        self,
        cluster_dir: Path,
        sample_ids: list[str],
        model_ids: list[str] | None = None,
        *,
        state_ids: list[str] | None = None,
        metastable_ids: list[str] | None = None,
        base_dir: Path | None = None,
        sample_entries: list[dict] | None = None,
    ):
        self._cluster_dir = cluster_dir
        self._sample_ids = sample_ids
        self._model_ids = model_ids or []
        self._state_ids = state_ids or []
        self._metastable_ids = metastable_ids or []
        self.base_dir = base_dir or cluster_dir.parent
        self._sample_entries = sample_entries

    def ensure_cluster_directories(self, project_id: str, system_id: str, cluster_id: str):
        return {"cluster_dir": self._cluster_dir}

    def list_samples(self, project_id: str, system_id: str, cluster_id: str):
        if self._sample_entries is not None:
            return self._sample_entries
        return [{"sample_id": sid} for sid in self._sample_ids]

    def list_cluster_entries(self, project_id: str, system_id: str):
        return [{"cluster_id": "cluster-1", "samples": self.list_samples(project_id, system_id, "cluster-1")}]

    def list_potts_models(self, project_id: str, system_id: str, cluster_id: str):
        return [{"model_id": mid} for mid in self._model_ids]

    def get_system(self, project_id: str, system_id: str):
        return SimpleNamespace(
            states={sid: object() for sid in self._state_ids},
            metastable_states=[{"metastable_id": mid} for mid in self._metastable_ids],
            system_id=system_id,
            project_id=project_id,
        )

    def resolve_path(self, project_id: str, system_id: str, rel_path: str):
        return (self.base_dir / project_id / "systems" / system_id / rel_path).resolve()


def _write_analysis(root: Path, analysis_type: str, analysis_id: str, meta: dict):
    analysis_dir = root / "analyses" / analysis_type / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(analysis_dir / "analysis.npz", dummy=np.asarray([1], dtype=int))
    return analysis_dir


def test_cleanup_orphan_cluster_analyses_removes_md_vs_sample_with_missing_sample(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    _write_analysis(
        cluster_dir,
        "md_vs_sample",
        "a1",
        {
            "analysis_type": "md_vs_sample",
            "analysis_id": "a1",
            "md_sample_id": "md_1",
            "sample_id": "potts_missing",
        },
    )
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(cluster_dir=cluster_dir, sample_ids=["md_1"], model_ids=[]),
    )

    removed = analysis_cleanup.cleanup_orphan_cluster_analyses("p", "s", "c")

    assert removed == 1
    assert not (cluster_dir / "analyses" / "md_vs_sample" / "a1").exists()


def test_remove_md_samples_for_deleted_state_preserves_other_samples(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    for sample_id in ("md_deleted", "md_kept", "potts_kept"):
        sample_dir = cluster_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample.npz").write_bytes(b"sample")
    samples = [
        {"sample_id": "md_deleted", "type": "md_eval", "state_id": "deleted"},
        {"sample_id": "md_kept", "type": "md_eval", "state_id": "kept"},
        {"sample_id": "potts_kept", "type": "potts_sampling", "state_id": "deleted"},
    ]
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(cluster_dir=cluster_dir, sample_ids=[], sample_entries=samples),
    )

    summary = analysis_cleanup.remove_md_samples_for_states("p", "s", ["deleted"])

    assert summary["md_samples_removed"] == 1
    assert summary["removed_md_sample_ids"] == ["md_deleted"]
    assert not (cluster_dir / "samples" / "md_deleted").exists()
    assert (cluster_dir / "samples" / "md_kept").exists()
    assert (cluster_dir / "samples" / "potts_kept").exists()


def test_cleanup_orphan_cluster_analyses_keeps_delta_js_if_some_samples_still_exist(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    _write_analysis(
        cluster_dir,
        "delta_js",
        "djs1",
        {
            "analysis_type": "delta_js",
            "analysis_id": "djs1",
            "model_a_id": "ma",
            "model_b_id": "mb",
            "reference_sample_ids_a": ["ref_a"],
            "reference_sample_ids_b": ["ref_b"],
            "summary": {"sample_ids": ["keep_me", "gone_me"]},
        },
    )
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(cluster_dir=cluster_dir, sample_ids=["ref_a", "ref_b", "keep_me"], model_ids=["ma", "mb"]),
    )

    removed = analysis_cleanup.cleanup_orphan_cluster_analyses("p", "s", "c")

    assert removed == 0
    assert (cluster_dir / "analyses" / "delta_js" / "djs1").exists()


def test_cleanup_orphan_cluster_analyses_removes_delta_js_if_all_tracked_samples_are_gone(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    _write_analysis(
        cluster_dir,
        "delta_js",
        "djs2",
        {
            "analysis_type": "delta_js",
            "analysis_id": "djs2",
            "model_a_id": "ma",
            "model_b_id": "mb",
            "reference_sample_ids_a": ["ref_a"],
            "reference_sample_ids_b": ["ref_b"],
            "summary": {"sample_ids": ["gone_1", "gone_2"]},
        },
    )
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(cluster_dir=cluster_dir, sample_ids=["ref_a", "ref_b"], model_ids=["ma", "mb"]),
    )

    removed = analysis_cleanup.cleanup_orphan_cluster_analyses("p", "s", "c")

    assert removed == 1
    assert not (cluster_dir / "analyses" / "delta_js" / "djs2").exists()


def test_cleanup_orphan_cluster_analyses_removes_analysis_with_missing_state_refs(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    _write_analysis(
        cluster_dir,
        "delta_js",
        "djs3",
        {
            "analysis_type": "delta_js",
            "analysis_id": "djs3",
            "model_a_id": "ma",
            "model_b_id": "mb",
            "contact_state_ids": ["state_ok", "state_missing"],
            "summary": {"sample_ids": ["keep_me"]},
        },
    )
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(
            cluster_dir=cluster_dir,
            sample_ids=["keep_me"],
            model_ids=["ma", "mb"],
            state_ids=["state_ok"],
        ),
    )

    removed = analysis_cleanup.cleanup_orphan_cluster_analyses("p", "s", "c")

    assert removed == 1
    assert not (cluster_dir / "analyses" / "delta_js" / "djs3").exists()


def test_cleanup_state_linked_results_removes_jobs_referencing_missing_states(monkeypatch, tmp_path):
    project_id = "proj"
    system_id = "sys"
    system_dir = tmp_path / "projects" / project_id / "systems" / system_id
    jobs_dir = system_dir / "results" / "jobs"
    artifact_dir = system_dir / "results" / "ligand_completion" / "job-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "dummy.txt").write_text("x", encoding="utf-8")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_type": "ligand_completion",
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "states": {
                "state_a": {"id": "state_ok"},
                "state_b": {"id": "meta_missing"},
            },
        },
        "results": {
            "results_dir": f"projects/{project_id}/systems/{system_id}/results/ligand_completion/job-1",
        },
    }
    (jobs_dir / "job-1.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(
            cluster_dir=tmp_path / "cluster",
            sample_ids=[],
            model_ids=[],
            state_ids=["state_ok"],
            metastable_ids=["meta_ok"],
            base_dir=tmp_path / "projects",
        ),
    )

    removed = analysis_cleanup.cleanup_state_linked_results(project_id, system_id)

    assert removed == 1
    assert not (jobs_dir / "job-1.json").exists()
    assert not artifact_dir.exists()
