import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from phase.potts.analysis_run import load_endpoint_framewise_payload, upsert_endpoint_frustration_analysis
from phase.potts.potts_model import PottsModel, save_potts_model
from phase.potts.sample_io import save_sample_npz
from phase.services.project_store import ProjectStore


def _write_sample(system_dir: Path, cluster_id: str, sample_id: str, meta: dict, *, labels: np.ndarray, invalid_mask=None):
    sample_dir = system_dir / "clusters" / cluster_id / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample.npz"
    save_sample_npz(sample_path, labels=labels, invalid_mask=invalid_mask)
    payload = dict(meta)
    payload.setdefault("sample_id", sample_id)
    payload.setdefault("path", str(sample_path.relative_to(system_dir)))
    payload.setdefault("paths", {"summary_npz": str(sample_path.relative_to(system_dir))})
    (sample_dir / "sample_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_model(model_dir: Path, system_dir: Path, model_id: str, name: str, *, h, J, edges):
    target = model_dir / model_id
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.npz"
    save_potts_model(PottsModel(h=h, J=J, edges=edges), model_path)
    payload = {
        "model_id": model_id,
        "name": name,
        "path": str(model_path.relative_to(system_dir)),
        "params": {"fit_mode": "delta"},
    }
    (target / "model_metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_endpoint_frustration_analysis_writes_summary_and_framewise_npz(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    store.save_system(system)

    cluster_id = "cluster1"
    cluster_dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    np.savez_compressed(
        cluster_dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_10", "res_20"], dtype=str),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )

    edges = [(0, 1)]
    model_a_entry = _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-a",
        "Active-like",
        h=[np.asarray([0.0, 1.0]), np.asarray([0.4, -0.1])],
        J={(0, 1): np.asarray([[0.2, -0.4], [0.1, 0.3]])},
        edges=edges,
    )
    model_b_entry = _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-b",
        "Inactive-like",
        h=[np.asarray([0.5, -0.2]), np.asarray([0.0, 0.2])],
        J={(0, 1): np.asarray([[-0.1, 0.2], [0.0, -0.3]])},
        edges=edges,
    )
    _write_sample(
        system_dir,
        cluster_id,
        "md1",
        {"name": "MD 1", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "sa1",
        {"name": "SA 1", "type": "potts_sampling", "method": "sa"},
        labels=np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int32),
    )

    out = upsert_endpoint_frustration_analysis(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_a_ref="model-a",
        model_b_ref="model-b",
        sample_ids=["md1", "sa1"],
        top_k_edges=10,
    )

    meta = out["metadata"]
    assert meta["analysis_type"] == "endpoint_frustration"
    assert meta["summary"]["n_samples"] == 2
    with np.load(out["analysis_npz"], allow_pickle=False) as data:
        assert np.asarray(data["q_residue_all"]).shape == (2, 2)
        assert np.asarray(data["q_edge"]).shape == (2, 1)
        assert np.asarray(data["frustration_node_sym_mean"]).shape == (2, 2)
        assert np.asarray(data["frustration_edge_sym_mean"]).shape == (2, 1)
        assert np.asarray(data["delta_energy_hist"]).shape[0] == 2
        assert np.asarray(data["delta_energy_bins"]).ndim == 1
        assert np.asarray(data["delta_energy_mean"]).shape == (2,)
        assert np.all(np.asarray(data["node_norm_scale_a"]) > 0)
        assert np.all(np.asarray(data["edge_norm_scale_a"]) > 0)
    framewise_dir = Path(out["analysis_dir"]) / "samples"
    with np.load(framewise_dir / "md1.npz", allow_pickle=False) as data:
        assert np.asarray(data["frustration_node_sym_framewise"]).shape == (3, 2)
        assert np.asarray(data["frustration_edge_sym_framewise"]).shape == (3, 1)
        assert np.asarray(data["delta_energy_framewise"]).shape == (3,)


def test_endpoint_frustration_analysis_drops_invalid_frames(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    store.save_system(system)

    cluster_id = "cluster1"
    cluster_dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    np.savez_compressed(
        cluster_dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_10", "res_20"], dtype=str),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )

    edges = [(0, 1)]
    model_a_entry = _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-a",
        "Active-like",
        h=[np.asarray([0.0, 1.0]), np.asarray([0.4, -0.1])],
        J={(0, 1): np.asarray([[0.2, -0.4], [0.1, 0.3]])},
        edges=edges,
    )
    model_b_entry = _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-b",
        "Inactive-like",
        h=[np.asarray([0.5, -0.2]), np.asarray([0.0, 0.2])],
        J={(0, 1): np.asarray([[-0.1, 0.2], [0.0, -0.3]])},
        edges=edges,
    )
    _write_sample(
        system_dir,
        cluster_id,
        "sa-invalid",
        {"name": "SA invalid", "type": "potts_sampling", "method": "sa"},
        labels=np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int32),
        invalid_mask=np.asarray([False, True, False], dtype=bool),
    )

    out = upsert_endpoint_frustration_analysis(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_a_ref="model-a",
        model_b_ref="model-b",
        sample_ids=["sa-invalid"],
        top_k_edges=5,
        drop_invalid=True,
    )

    with np.load(out["analysis_npz"], allow_pickle=False) as data:
        assert np.asarray(data["sample_frame_counts"]).tolist() == [2]
        assert np.asarray(data["sample_invalid_counts"]).tolist() == [1]
    framewise_dir = Path(out["analysis_dir"]) / "samples"
    with np.load(framewise_dir / "sa-invalid.npz", allow_pickle=False) as data:
        assert np.asarray(data["frame_count"]).tolist() == [2]
        assert np.asarray(data["frustration_node_sym_framewise"]).shape[0] == 2


def test_endpoint_frustration_analysis_parallel_matches_serial(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    store.save_system(system)

    cluster_id = "cluster1"
    cluster_dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    np.savez_compressed(
        cluster_dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_10", "res_20"], dtype=str),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )

    edges = [(0, 1)]
    _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-a",
        "Active-like",
        h=[np.asarray([0.0, 1.0]), np.asarray([0.4, -0.1])],
        J={(0, 1): np.asarray([[0.2, -0.4], [0.1, 0.3]])},
        edges=edges,
    )
    _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-b",
        "Inactive-like",
        h=[np.asarray([0.5, -0.2]), np.asarray([0.0, 0.2])],
        J={(0, 1): np.asarray([[-0.1, 0.2], [0.0, -0.3]])},
        edges=edges,
    )
    _write_sample(
        system_dir,
        cluster_id,
        "md1",
        {"name": "MD 1", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "sa1",
        {"name": "SA 1", "type": "potts_sampling", "method": "sa"},
        labels=np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int32),
    )

    serial = upsert_endpoint_frustration_analysis(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_a_ref="model-a",
        model_b_ref="model-b",
        sample_ids=["md1", "sa1"],
        top_k_edges=10,
        n_workers=1,
    )
    with np.load(serial["analysis_npz"], allow_pickle=False) as data:
        serial_q = np.asarray(data["q_residue_all"], dtype=np.float32)
        serial_f = np.asarray(data["frustration_node_sym_mean"], dtype=np.float32)

    parallel = upsert_endpoint_frustration_analysis(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_a_ref="model-a",
        model_b_ref="model-b",
        sample_ids=["md1", "sa1"],
        top_k_edges=10,
        n_workers=2,
    )
    with np.load(parallel["analysis_npz"], allow_pickle=False) as data:
        np.testing.assert_allclose(np.asarray(data["q_residue_all"], dtype=np.float32), serial_q)
        np.testing.assert_allclose(np.asarray(data["frustration_node_sym_mean"], dtype=np.float32), serial_f)
        assert int(np.asarray(data["sample_frame_counts"], dtype=np.int32).sum()) == 6


def test_endpoint_frustration_progress_callback_reports_sample_progress(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    store.save_system(system)

    cluster_id = "cluster1"
    cluster_dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    np.savez_compressed(
        cluster_dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_10", "res_20"], dtype=str),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )

    edges = [(0, 1)]
    _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-a",
        "Active-like",
        h=[np.asarray([0.0, 1.0]), np.asarray([0.4, -0.1])],
        J={(0, 1): np.asarray([[0.2, -0.4], [0.1, 0.3]])},
        edges=edges,
    )
    _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-b",
        "Inactive-like",
        h=[np.asarray([0.5, -0.2]), np.asarray([0.0, 0.2])],
        J={(0, 1): np.asarray([[-0.1, 0.2], [0.0, -0.3]])},
        edges=edges,
    )
    _write_sample(
        system_dir,
        cluster_id,
        "md1",
        {"name": "MD 1", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "sa1",
        {"name": "SA 1", "type": "potts_sampling", "method": "sa"},
        labels=np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int32),
    )

    progress_events = []

    def progress_cb(message: str, current: int, total: int):
        progress_events.append((message, int(current), int(total)))

    upsert_endpoint_frustration_analysis(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_a_ref="model-a",
        model_b_ref="model-b",
        sample_ids=["md1", "sa1"],
        top_k_edges=10,
        n_workers=1,
        progress_callback=progress_cb,
    )

    assert progress_events
    assert progress_events[0] == ("Computing endpoint frustration", 0, 2)
    assert progress_events[-1] == ("Computing endpoint frustration", 2, 2)


def test_endpoint_frustration_framewise_payload_includes_rankings_and_slice(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    store.save_system(system)

    cluster_id = "cluster1"
    cluster_dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    np.savez_compressed(
        cluster_dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_10", "res_20"], dtype=str),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )
    edges = [(0, 1)]
    _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-a",
        "Active-like",
        h=[np.asarray([0.0, 1.0]), np.asarray([0.4, -0.1])],
        J={(0, 1): np.asarray([[0.2, -0.4], [0.1, 0.3]])},
        edges=edges,
    )
    _write_model(
        cluster_dirs["potts_models_dir"],
        system_dir,
        "model-b",
        "Inactive-like",
        h=[np.asarray([0.5, -0.2]), np.asarray([0.0, 0.2])],
        J={(0, 1): np.asarray([[-0.1, 0.2], [0.0, -0.3]])},
        edges=edges,
    )
    _write_sample(
        system_dir,
        cluster_id,
        "md1",
        {"name": "MD 1", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32),
    )

    out = upsert_endpoint_frustration_analysis(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_a_ref="model-a",
        model_b_ref="model-b",
        sample_ids=["md1"],
        top_k_edges=10,
    )
    framewise_path = Path(out["analysis_dir"]) / "samples" / "md1.npz"
    payload = load_endpoint_framewise_payload(framewise_path, start=1, stop=4, step=2, include_ranks=True, include_edges=False)
    assert payload["analysis_format_version"] == 2
    assert payload["slice"]["frame_indices"] == [1, 3]
    assert len(payload["node"]["sym"]) == 2
    assert "ranking" in payload
    assert len(payload["ranking"]["score_node_sym_topj_default"]) == 4
    assert sorted(payload["ranking"]["rank_node_sym_topj_default"]) == [0, 1, 2, 3]
