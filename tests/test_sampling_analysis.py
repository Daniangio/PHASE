import json
import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from backend import tasks
from backend.api.v1.analysis_payloads import downsample_model_energy_payload
from backend.tasks import run_md_samples_refresh_job
from phase.potts.analysis_run import analyze_cluster_samples
from phase.potts.orchestration import _filter_potts_analysis_samples, _load_additive_potts_analysis_model
from phase.potts.potts_model import PottsModel, save_potts_model
from phase.potts.sample_io import save_sample_npz
from phase.services.project_store import DescriptorState, ProjectStore
from phase.workflows.clustering import _predict_cluster_adp


class _FakeJob:
    def __init__(self, connection=None, origin="phase-jobs", job_id="rq-test-job"):
        self.connection = connection
        self.origin = origin
        self.id = job_id
        self.meta = {}

    def save_meta(self):
        return None


class _FakeRedis:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(str(key))

    def set(self, key, value, ex=None):
        self._store[str(key)] = str(value)
        return True

    def delete(self, *keys):
        for key in keys:
            self._store.pop(str(key), None)
        return True

    def decr(self, key):
        current = int(self._store.get(str(key), "0"))
        current -= 1
        self._store[str(key)] = str(current)
        return current


def _write_prepared_pickle(prepared):
    paths = tasks.orchestration_paths(Path(prepared["analysis_dir"]))
    paths["root"].mkdir(parents=True, exist_ok=True)
    with open(paths["prepared"], "wb") as fh:
        pickle.dump(prepared, fh, protocol=pickle.HIGHEST_PROTOCOL)


def test_sampling_explorer_filters_samples_by_selected_model_or_explicit_ids():
    samples = [
        {"sample_id": "a", "model_id": "model-a"},
        {"sample_id": "b", "model_ids": ["model-b", "model-c"]},
        {"sample_id": "c", "params": {"model_id": "model-a"}},
        {"sample_id": "legacy"},
    ]

    linked = _filter_potts_analysis_samples(samples, model_id="model-a", requested_sample_ids=None)
    assert [sample["sample_id"] for sample in linked] == ["a", "c"]

    explicit = _filter_potts_analysis_samples(samples, model_id="model-a", requested_sample_ids=["b"])
    assert [sample["sample_id"] for sample in explicit] == ["b"]

    additive = _filter_potts_analysis_samples(
        samples,
        model_ids=["model-b", "model-c"],
        requested_sample_ids=None,
    )
    assert [sample["sample_id"] for sample in additive] == ["b"]


def test_sampling_explorer_adds_selected_potts_hamiltonians(tmp_path):
    model_a = PottsModel(
        h=[np.asarray([0.0, 1.0]), np.asarray([0.0, 2.0])],
        J={(0, 1): np.asarray([[0.0, 0.5], [1.0, 1.5]])},
        edges=[(0, 1)],
    )
    model_b = PottsModel(
        h=[np.asarray([0.0, 3.0]), np.asarray([0.0, 4.0])],
        J={(0, 1): np.asarray([[0.0, 2.0], [2.5, 3.0]])},
        edges=[(0, 1)],
    )
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    save_potts_model(model_a, path_a)
    save_potts_model(model_b, path_b)
    labels = np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.int32)

    additive = _load_additive_potts_analysis_model([path_a, path_b])

    expected = model_a.energy_batch(labels) + model_b.energy_batch(labels)
    assert np.allclose(additive.energy_batch(labels), expected)


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


def test_analyze_cluster_samples_reports_all_invalid_sa_sample(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    cluster_id = "cluster1"
    cluster_dir = system_dir / "clusters" / cluster_id
    cluster_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cluster_dir / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        merged__labels_assigned=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )

    _write_sample(
        system_dir,
        cluster_id,
        "md1",
        {"name": "MD 1", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "sa1",
        {"name": "SA 1", "type": "potts_sampling", "method": "sa"},
        labels=np.asarray([[0, 0], [1, 1], [0, 1]], dtype=np.int32),
        invalid_mask=np.asarray([True, True, True], dtype=bool),
    )

    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    out = analyze_cluster_samples(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_ref=None,
        md_label_mode="assigned",
        drop_invalid=True,
    )

    assert out["comparisons_written"] == 0
    assert out["energies_written"] == 0
    assert out["skipped_samples"] == [
        {
            "sample_id": "sa1",
            "sample_name": "SA 1",
            "sample_type": "potts_sampling",
            "sample_method": "sa",
            "stage": "md_vs_sample",
            "reason": "all_frames_invalid",
            "n_frames_total": 3,
            "n_frames_used": 0,
            "invalid_count": 3,
        }
    ]


def test_predict_cluster_adp_single_frame_duplicates_query():
    class DummyDP:
        X = np.zeros((5, 2), dtype=float)

        def predict_cluster_ADP(self, emb, *, maxk, density_est, n_jobs):
            assert emb.shape == (2, 2)
            assert maxk == 1
            return np.asarray([[3], [3]], dtype=np.int32), np.asarray([[4], [4]], dtype=np.int32)

    assigned, halo = _predict_cluster_adp(DummyDP(), np.asarray([[0.1, 0.2]], dtype=float), density_maxk=100)
    assert assigned.tolist() == [3]
    assert halo.tolist() == [4]


def test_predict_cluster_adp_trims_query_to_model_dimensions():
    class DummyDP:
        X = np.zeros((8, 4), dtype=float)

        def predict_cluster_ADP(self, emb, *, maxk, density_est, n_jobs):
            assert emb.shape == (2, 4)
            return np.asarray([[7], [7]], dtype=np.int32), np.asarray([[8], [8]], dtype=np.int32)

    assigned, halo = _predict_cluster_adp(
        DummyDP(),
        np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=float),
        density_maxk=100,
    )
    assert assigned.tolist() == [7]
    assert halo.tolist() == [8]


def test_predict_cluster_adp_falls_back_to_nearest_training_label_on_dadapy_buffer_error():
    class DummyDP:
        X = np.asarray([[0.1, 0.2], [2.0, 2.1], [4.0, 4.1]], dtype=float)
        cluster_assignment = np.asarray([3, 4, 5], dtype=np.int32)
        cluster_assignment_halo = np.asarray([13, 14, 15], dtype=np.int32)

        def predict_cluster_ADP(self, emb, *, maxk, density_est, n_jobs):
            raise ValueError("Buffer has wrong number of dimensions (expected 2, got 1)")

    assigned, halo = _predict_cluster_adp(DummyDP(), np.asarray([[2.05, 2.0]], dtype=float), density_maxk=100)
    assert assigned.tolist() == [4]
    assert halo.tolist() == [14]


def test_run_md_samples_refresh_job_filters_selected_states(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.states["state1"] = DescriptorState(
        state_id="state1",
        name="State 1",
        descriptor_file="states/state1/descriptors.npz",
        storage_key="state1",
    )
    system.states["state2"] = DescriptorState(
        state_id="state2",
        name="State 2",
        descriptor_file="states/state2/descriptors.npz",
        storage_key="state2",
    )
    system.metastable_clusters = [{"cluster_id": "cluster1", "samples": []}]
    store.save_system(system)
    store.ensure_cluster_directories("proj", "sys", "cluster1")
    monkeypatch.setattr("backend.tasks.project_store", store)

    seen = []

    def _fake_evaluate_state_with_models(
        project_id,
        system_id,
        cluster_id,
        state_id,
        store=None,
        sample_id=None,
        workers=1,
        progress_callback=None,
    ):
        seen.append(state_id)
        return {
            "sample_id": sample_id or f"sample-{state_id}",
            "name": state_id,
            "type": "md_eval",
            "method": "md_eval",
            "state_id": state_id,
            "paths": {"summary_npz": f"clusters/{cluster_id}/samples/{sample_id or f'sample-{state_id}'}/sample.npz"},
        }

    monkeypatch.setattr("backend.tasks.evaluate_state_with_models", _fake_evaluate_state_with_models)

    out = run_md_samples_refresh_job(
        "job-1",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster1"},
        {"state_ids": ["state2"], "overwrite": True, "cleanup": True},
    )

    assert seen == ["state2"]
    assert out["status"] == "finished"
    assert out["results"]["states"] == 1
    assert out["results"]["requested_state_ids"] == ["state2"]


def test_analyze_cluster_samples_supports_parallel_local_workers(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    cluster_id = "cluster1"
    cluster_dir = system_dir / "clusters" / cluster_id
    cluster_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cluster_dir / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        merged__labels_assigned=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
        edges=np.asarray([[0, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "md1",
        {"name": "MD 1", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "sa1",
        {"name": "SA 1", "type": "potts_sampling", "method": "sa"},
        labels=np.asarray([[0, 0], [0, 1], [1, 1]], dtype=np.int32),
    )

    model_dir = cluster_dir / "potts_models" / "model1"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.npz"
    save_potts_model(
        PottsModel(
            h=[np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, -1.0], dtype=float)],
            J={(0, 1): np.zeros((2, 2), dtype=float)},
            edges=[(0, 1)],
        ),
        model_path,
    )

    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    out = analyze_cluster_samples(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_ref=str(model_path),
        md_label_mode="assigned",
        drop_invalid=True,
        n_workers=2,
    )

    assert out["comparisons_written"] == 1
    assert out["energies_written"] == 2
    assert out["workers_used"] == 2


def test_run_potts_analysis_job_uses_distributed_rq_workers(tmp_path, monkeypatch):
    redis_conn = _FakeRedis()
    fake_job = _FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "analysis-distributed"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    prepared = {
        "analysis_id": "dist-analysis",
        "analysis_dir": str(analysis_dir),
        "cluster_dir": str(tmp_path / "cluster"),
        "system_dir": str(tmp_path / "system"),
        "comparisons_root": str(tmp_path / "cluster" / "analyses" / "md_vs_sample"),
        "energies_root": str(tmp_path / "cluster" / "analyses" / "model_energy"),
        "project_id": "proj",
        "system_id": "sys",
        "cluster_id": "cluster",
        "model_id": "model-1",
        "model_name": "Model 1",
        "md_label_mode": "assigned",
        "drop_invalid": True,
        "md_samples": 1,
        "other_samples": 1,
        "edges_for_metrics": [(0, 1)],
        "payloads": [
            {"kind": "md_vs_sample"},
            {"kind": "model_energy"},
        ],
        "requested_workers": 0,
    }

    enqueued = []

    class _FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

        def enqueue(self, func, args=(), **kwargs):
            enqueued.append((func.__name__, args, kwargs))
            func(*args)
            return type("FakeEnqueuedJob", (), {"id": kwargs.get("job_id", "fake-job")})()

    prepare_kwargs = {}

    def _fake_prepare(**kwargs):
        prepare_kwargs.update(kwargs)
        _write_prepared_pickle(prepared)
        return prepared

    def _fake_worker(payload):
        if payload.get("kind") == "md_vs_sample":
            return {
                "kind": "md_vs_sample",
                "md_sample": {"sample_id": "md1", "name": "MD 1"},
                "sample": {"sample_id": "sa1", "name": "SA 1", "type": "potts_sampling", "method": "sa"},
                "node_js": np.asarray([0.1, 0.2], dtype=float),
                "edge_js": np.asarray([0.3], dtype=float),
                "summary": {"combined_distance": 0.2, "md_count": 2, "sample_count": 3},
            }
        return {
            "kind": "model_energy",
            "sample": {"sample_id": "sa1", "name": "SA 1", "type": "potts_sampling", "method": "sa"},
            "energies": np.asarray([1.0, 2.0], dtype=float),
            "summary": {"energy_mean": 1.5, "count": 2},
        }

    def _fake_aggregate(prepared_obj, out_rows, workers_used):
        result = {
            "analysis_id": prepared_obj["analysis_id"],
            "analysis_dir": prepared_obj["analysis_dir"],
            "metadata": {
                "comparisons_written": 1,
                "energies_written": 1,
                "workers_used": workers_used,
                "skipped_samples": [],
            },
        }
        with open(tasks.orchestration_paths(Path(prepared_obj["analysis_dir"]))["aggregate"], "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks.project_store, "get_system", lambda *args, **kwargs: type("S", (), {"metastable_clusters": [{"cluster_id": "cluster"}]})())
    monkeypatch.setattr(tasks, "prepare_potts_analysis_batch", _fake_prepare)
    monkeypatch.setattr(tasks, "_potts_analysis_payload_worker", _fake_worker)
    monkeypatch.setattr(tasks, "aggregate_potts_analysis_batch", _fake_aggregate)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 3)
    monkeypatch.setattr(tasks, "Queue", _FakeQueue)

    out = tasks.run_potts_analysis_job(
        "job-dist",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {"model_id": "model-1", "workers": 0, "sample_ids": ["sa1"]},
    )

    assert out["status"] == "finished"
    assert prepare_kwargs["comparison_sample_ids"] == ["sa1"]
    assert out["results"]["summary"]["comparisons_written"] == 1
    assert any(name == "run_potts_analysis_payload_job" for name, _args, _kwargs in enqueued)
    assert any(name == "run_potts_analysis_aggregate_job" for name, _args, _kwargs in enqueued)


def test_downsample_model_energy_payload_limits_rows_and_is_reproducible():
    payload = {"energies": np.arange(100, dtype=float)}
    out_a = downsample_model_energy_payload(payload, row_limit=15, seed=7)
    out_b = downsample_model_energy_payload(payload, row_limit=15, seed=7)

    assert np.asarray(out_a["energies"]).shape == (15,)
    assert np.asarray(out_a["sampled_row_indices"]).shape == (15,)
    assert np.array_equal(np.asarray(out_a["energies"]), np.asarray(out_b["energies"]))
    assert np.array_equal(np.asarray(out_a["sampled_row_indices"]), np.asarray(out_b["sampled_row_indices"]))
    assert int(np.asarray(out_a["sampled_row_count"]).ravel()[0]) == 15
    assert int(np.asarray(out_a["original_row_count"]).ravel()[0]) == 100
    assert int(np.asarray(out_a["downsampled"]).ravel()[0]) == 1
