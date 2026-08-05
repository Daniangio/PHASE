import pickle
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

os.environ["PHASE_DATA_ROOT"] = "/tmp/phase-test-data"
sys.modules.setdefault("dadapy", types.SimpleNamespace(Data=object))

from backend import tasks
from phase.potts.orchestration import orchestration_paths, run_local_payload_batch


def _ordered_worker(payload):
    time.sleep(float(payload["delay"]))
    return {"row": int(payload["row"])}


class FakeJob:
    def __init__(self, connection=None, origin="phase-jobs", job_id="rq-test-job"):
        self.connection = connection
        self.origin = origin
        self.id = job_id
        self.meta = {}

    def save_meta(self):
        return None


class FakeRedis:
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
    paths = orchestration_paths(Path(prepared["analysis_dir"]))
    paths["root"].mkdir(parents=True, exist_ok=True)
    with open(paths["prepared"], "wb") as fh:
        pickle.dump(prepared, fh, protocol=pickle.HIGHEST_PROTOCOL)


def test_run_local_payload_batch_preserves_input_order_with_parallel_workers():
    payloads = [
        {"row": 0, "delay": 0.25},
        {"row": 1, "delay": 0.01},
        {"row": 2, "delay": 0.10},
    ]
    out = run_local_payload_batch(payloads, worker_fn=_ordered_worker, max_workers=2)
    assert [row["row"] for row in out] == [0, 1, 2]


def test_potts_nn_web_job_falls_back_to_serial_inside_single_rq_worker(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "nn-analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_npz = analysis_dir / "analysis.npz"
    prepared = {
        "analysis_id": "nn-local-analysis",
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_npz),
        "payloads": [{"row": row} for row in range(8)],
        "requested_workers": 8,
    }
    calls = {"max_workers": None}

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

    def fake_run_local(payloads, worker_fn, max_workers, progress_callback=None, progress_label=None):
        calls["max_workers"] = max_workers
        return [{"row": payload["row"]} for payload in payloads]

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        analysis_npz.write_text("npz", encoding="utf-8")
        return {
            "analysis_id": prepared_obj["analysis_id"],
            "analysis_dir": prepared_obj["analysis_dir"],
            "analysis_npz": prepared_obj["analysis_npz"],
            "metadata": {
                "analysis_id": prepared_obj["analysis_id"],
                "summary": {"workers": workers_used},
            },
        }

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks, "prepare_potts_nn_mapping_batch", lambda **kwargs: prepared)
    monkeypatch.setattr(tasks, "run_local_payload_batch", fake_run_local)
    monkeypatch.setattr(tasks, "aggregate_potts_nn_mapping_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 1)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)

    out = tasks.run_potts_nearest_neighbor_job(
        "nn-local-job",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {"model_id": "model", "sample_id": "sample", "md_sample_id": "md", "workers": 8},
    )

    assert calls["max_workers"] == 1
    assert out["status"] == "finished"
    assert out["results"]["summary"]["workers"] == 1


def test_ligand_completion_job_falls_back_to_serial_inside_worker(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "analysis-local"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_npz = analysis_dir / "analysis.npz"

    prepared = {
        "analysis_id": "local-analysis",
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_npz),
        "payloads": [{"row": 0}, {"row": 1}],
    }

    calls = {"max_workers": None}

    def fake_prepare(**kwargs):
        return prepared

    def fake_run_local(payloads, worker_fn, max_workers, progress_callback=None, progress_label=None):
        calls["max_workers"] = max_workers
        return [
            {
                "success_a": [0.0, 1.0],
                "success_b": [1.0, 0.0],
                "js_a_under_a": [0.1, 0.1],
                "js_b_under_a": [0.2, 0.2],
                "js_a_under_b": [0.3, 0.3],
                "js_b_under_b": [0.4, 0.4],
                "novelty_under_a": [0.0, 0.0],
                "novelty_under_b": [0.0, 0.0],
                "deltae_mean_under_a": [0.0, 0.0],
                "deltae_mean_under_b": [0.0, 0.0],
                "success_js_eval_under_a": [0.0, 0.0],
                "success_js_eval_under_b": [0.0, 0.0],
                "success_js_eval_node_under_a": [0.0, 0.0],
                "success_js_eval_node_under_b": [0.0, 0.0],
                "success_js_eval_edge_under_a": [0.0, 0.0],
                "success_js_eval_edge_under_b": [0.0, 0.0],
                "raw_deltae": 0.0,
                "raw_js_a": 0.1,
                "raw_js_b": 0.2,
            }
            for _ in payloads
        ]

    def fake_aggregate(prepared, out_rows, workers_used):
        analysis_npz.write_text("npz", encoding="utf-8")
        return {
            "metadata": {"analysis_id": prepared["analysis_id"], "workers": workers_used},
            "analysis_dir": prepared["analysis_dir"],
            "analysis_npz": prepared["analysis_npz"],
        }

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks, "prepare_ligand_completion_batch", fake_prepare)
    monkeypatch.setattr(tasks, "run_local_payload_batch", fake_run_local)
    monkeypatch.setattr(tasks, "aggregate_ligand_completion_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 1)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)

    out = tasks.run_ligand_completion_job(
        "job-local",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {
            "model_a_id": "model-a",
            "model_b_id": "model-b",
            "md_sample_id": "md-sample",
            "constrained_residues": ["res_1"],
            "constraint_source_mode": "manual",
            "workers": 8,
        },
    )

    assert out["status"] == "finished"
    assert calls["max_workers"] == 1
    assert out["results"]["analysis_type"] == "ligand_completion"


def test_ligand_completion_job_uses_distributed_rq_workers(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "analysis-distributed"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_npz = analysis_dir / "analysis.npz"
    analysis_meta = analysis_dir / "analysis_metadata.json"

    prepared = {
        "analysis_id": "dist-analysis",
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_npz),
        "analysis_metadata": str(analysis_meta),
        "payloads": [{"row": 0}, {"row": 1}, {"row": 2}],
    }

    enqueued = []

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

        def enqueue(self, func, args=(), **kwargs):
            enqueued.append((func.__name__, args, kwargs))
            func(*args)
            return type("FakeEnqueuedJob", (), {"id": kwargs.get("job_id", "fake-job")})()

    def fake_prepare(**kwargs):
        _write_prepared_pickle(prepared)
        return prepared

    def fake_worker(payload):
        row = int(payload["row"])
        return {
            "success_a": [float(row), float(row + 1)],
            "success_b": [float(row + 2), float(row + 3)],
            "js_a_under_a": [0.1, 0.1],
            "js_b_under_a": [0.2, 0.2],
            "js_a_under_b": [0.3, 0.3],
            "js_b_under_b": [0.4, 0.4],
            "novelty_under_a": [0.0, 0.0],
            "novelty_under_b": [0.0, 0.0],
            "deltae_mean_under_a": [0.0, 0.0],
            "deltae_mean_under_b": [0.0, 0.0],
            "success_js_eval_under_a": [0.0, 0.0],
            "success_js_eval_under_b": [0.0, 0.0],
            "success_js_eval_node_under_a": [0.0, 0.0],
            "success_js_eval_node_under_b": [0.0, 0.0],
            "success_js_eval_edge_under_a": [0.0, 0.0],
            "success_js_eval_edge_under_b": [0.0, 0.0],
            "raw_deltae": 0.0,
            "raw_js_a": 0.1,
            "raw_js_b": 0.2,
        }

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        analysis_npz.write_text("npz", encoding="utf-8")
        analysis_meta.write_text("{}", encoding="utf-8")
        result = {
            "metadata": {"analysis_id": prepared_obj["analysis_id"], "workers": workers_used},
            "analysis_dir": prepared_obj["analysis_dir"],
            "analysis_npz": prepared_obj["analysis_npz"],
        }
        with open(orchestration_paths(Path(prepared_obj["analysis_dir"]))["aggregate"], "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks, "prepare_ligand_completion_batch", fake_prepare)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 4)
    monkeypatch.setattr(tasks, "_single_frame_ligand_completion_worker", fake_worker)
    monkeypatch.setattr(tasks, "aggregate_ligand_completion_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "run_local_payload_batch", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not run")))

    out = tasks.run_ligand_completion_job(
        "job-distributed",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {
            "model_a_id": "model-a",
            "model_b_id": "model-b",
            "md_sample_id": "md-sample",
            "constrained_residues": ["res_1"],
            "constraint_source_mode": "manual",
        },
    )

    frame_jobs = [name for name, _, _ in enqueued if name == "run_ligand_completion_frame_job"]
    aggregate_jobs = [name for name, _, _ in enqueued if name == "run_ligand_completion_aggregate_job"]
    assert len(frame_jobs) == 3
    assert len(aggregate_jobs) == 1
    assert out["status"] == "finished"
    assert out["results"]["analysis_type"] == "ligand_completion"


def test_gibbs_relaxation_job_falls_back_to_serial_inside_worker(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "gibbs-local"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_npz = analysis_dir / "analysis.npz"

    prepared = {
        "analysis_id": "gibbs-local-analysis",
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_npz),
        "payloads": [{"row": 0}, {"row": 1}],
        "requested_workers": 6,
    }

    calls = {"max_workers": None}

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

    def fake_prepare(**kwargs):
        return prepared

    def fake_run_local(payloads, worker_fn, max_workers, progress_callback=None, progress_label=None):
        calls["max_workers"] = max_workers
        return [
            {
                "first_flip": [1, 2],
                "flip_counts": [[1, 0], [0, 1]],
                "energy_trace": [0.1, 0.2],
            }
            for _ in payloads
        ]

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        analysis_npz.write_text("npz", encoding="utf-8")
        return {
            "metadata": {"analysis_id": prepared_obj["analysis_id"], "workers": workers_used},
            "analysis_dir": prepared_obj["analysis_dir"],
            "analysis_npz": prepared_obj["analysis_npz"],
        }

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks, "prepare_gibbs_relaxation_batch", fake_prepare)
    monkeypatch.setattr(tasks, "run_local_payload_batch", fake_run_local)
    monkeypatch.setattr(tasks, "aggregate_gibbs_relaxation_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 1)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)

    out = tasks.run_gibbs_relaxation_job(
        "job-gibbs-local",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {
            "start_sample_id": "start-sample",
            "model_id": "model-x",
            "workers": 6,
        },
    )

    assert out["status"] == "finished"
    assert calls["max_workers"] == 1
    assert out["results"]["analysis_type"] == "gibbs_relaxation"


def test_gibbs_relaxation_job_uses_distributed_rq_workers(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "gibbs-distributed"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_npz = analysis_dir / "analysis.npz"
    analysis_meta = analysis_dir / "analysis_metadata.json"

    prepared = {
        "analysis_id": "gibbs-dist-analysis",
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_npz),
        "analysis_metadata": str(analysis_meta),
        "payloads": [{"row": 0}, {"row": 1}, {"row": 2}],
        "requested_workers": 3,
    }

    enqueued = []

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

        def enqueue(self, func, args=(), **kwargs):
            enqueued.append((func.__name__, args, kwargs))
            func(*args)
            return type("FakeEnqueuedJob", (), {"id": kwargs.get("job_id", "fake-job")})()

    def fake_prepare(**kwargs):
        _write_prepared_pickle(prepared)
        return prepared

    def fake_worker(payload):
        row = int(payload["row"])
        return {
            "first_flip": [row + 1, row + 2],
            "flip_counts": [[1, 0], [0, 1]],
            "energy_trace": [0.1 + row, 0.2 + row],
        }

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        analysis_npz.write_text("npz", encoding="utf-8")
        analysis_meta.write_text("{}", encoding="utf-8")
        result = {
            "metadata": {"analysis_id": prepared_obj["analysis_id"], "workers": workers_used},
            "analysis_dir": prepared_obj["analysis_dir"],
            "analysis_npz": prepared_obj["analysis_npz"],
        }
        with open(orchestration_paths(Path(prepared_obj["analysis_dir"]))["aggregate"], "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks, "prepare_gibbs_relaxation_batch", fake_prepare)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 4)
    monkeypatch.setattr(tasks, "_gibbs_relax_worker", fake_worker)
    monkeypatch.setattr(tasks, "aggregate_gibbs_relaxation_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "run_local_payload_batch", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not run")))

    out = tasks.run_gibbs_relaxation_job(
        "job-gibbs-distributed",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {
            "start_sample_id": "start-sample",
            "model_id": "model-x",
        },
    )

    frame_jobs = [name for name, _, _ in enqueued if name == "run_gibbs_relaxation_frame_job"]
    aggregate_jobs = [name for name, _, _ in enqueued if name == "run_gibbs_relaxation_aggregate_job"]
    assert len(frame_jobs) == 3
    assert len(aggregate_jobs) == 1
    assert out["status"] == "finished"
    assert out["results"]["analysis_type"] == "gibbs_relaxation"


def test_simulation_job_falls_back_to_serial_inside_worker(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = tmp_path / "samples" / "sample-local"
    sample_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = tmp_path / "cluster.npz"
    cluster_path.write_text("cluster", encoding="utf-8")
    model_path = tmp_path / "model.npz"
    model_path.write_text("model", encoding="utf-8")
    entry = {"cluster_id": "cluster", "path": str(cluster_path), "potts_models": [{"model_id": "model-x", "path": str(model_path), "name": "Model X"}], "samples": []}
    system_meta = SimpleNamespace(name="System", metastable_clusters=[entry])

    prepared = {
        "sample_path": str(sample_dir / "sample.npz"),
        "payloads": [{"row": 0}, {"row": 1}],
        "requested_workers": 8,
        "progress_label": "Running sampling",
    }
    calls = {"max_workers": None}

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

    def fake_prepare(**kwargs):
        return prepared

    def fake_run_local(payloads, worker_fn, max_workers, progress_callback=None, progress_label=None):
        calls["max_workers"] = max_workers
        return [{"labels": [[0, 1], [1, 0]]} for _ in payloads]

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        sample_path = Path(prepared_obj["sample_path"])
        sample_path.write_text("npz", encoding="utf-8")
        return {
            "sample_path": str(sample_path),
            "n_samples": 4,
            "n_residues": 2,
            "workers_used": workers_used,
        }

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks.project_store, "ensure_cluster_directories", lambda *args, **kwargs: {"samples_dir": sample_dir.parent, "system_dir": tmp_path})
    monkeypatch.setattr(tasks.project_store, "get_system", lambda *args, **kwargs: system_meta)
    monkeypatch.setattr(tasks.project_store, "resolve_path", lambda *args, **kwargs: Path(args[-1]))
    monkeypatch.setattr(tasks, "prepare_sampling_batch", fake_prepare)
    monkeypatch.setattr(tasks, "run_local_payload_batch", fake_run_local)
    monkeypatch.setattr(tasks, "aggregate_sampling_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 1)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)
    monkeypatch.setattr(tasks, "_update_cluster_entry", lambda *args, **kwargs: None)

    out = tasks.run_simulation_job(
        "job-sim-local",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {"sampling_method": "gibbs", "potts_model_id": "model-x"},
    )

    assert out["status"] == "finished"
    assert calls["max_workers"] == 1
    assert out["results"]["summary_npz"] is not None


def test_simulation_job_uses_distributed_rq_workers(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = tmp_path / "samples" / "sample-dist"
    sample_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = tmp_path / "cluster.npz"
    cluster_path.write_text("cluster", encoding="utf-8")
    model_path = tmp_path / "model.npz"
    model_path.write_text("model", encoding="utf-8")
    entry = {"cluster_id": "cluster", "path": str(cluster_path), "potts_models": [{"model_id": "model-x", "path": str(model_path), "name": "Model X"}], "samples": []}
    system_meta = SimpleNamespace(name="System", metastable_clusters=[entry])
    prepared = {
        "sample_path": str(sample_dir / "sample.npz"),
        "payloads": [{"row": 0}, {"row": 1}, {"row": 2}],
        "requested_workers": 3,
        "progress_label": "Running sampling",
    }
    enqueued = []

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

        def enqueue(self, func, args=(), **kwargs):
            enqueued.append((func.__name__, args, kwargs))
            func(*args)
            return type("FakeEnqueuedJob", (), {"id": kwargs.get("job_id", "fake-job")})()

    def fake_prepare(**kwargs):
        sample_root = Path(kwargs["results_dir"])
        prepared_obj = dict(prepared)
        prepared_obj["sample_path"] = str(sample_root / "sample.npz")
        _write_prepared_pickle({"analysis_dir": str(sample_root), **prepared_obj})
        return prepared_obj

    def fake_worker(payload):
        return {"labels": [[int(payload["row"]), 1]]}

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        sample_path = Path(prepared_obj["sample_path"])
        sample_path.write_text("npz", encoding="utf-8")
        result = {
            "sample_path": str(sample_path),
            "n_samples": len(out_rows),
            "n_residues": 2,
            "workers_used": workers_used,
        }
        with open(orchestration_paths(sample_path.parent)["aggregate"], "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks.project_store, "ensure_cluster_directories", lambda *args, **kwargs: {"samples_dir": sample_dir.parent, "system_dir": tmp_path})
    monkeypatch.setattr(tasks.project_store, "get_system", lambda *args, **kwargs: system_meta)
    monkeypatch.setattr(tasks.project_store, "resolve_path", lambda *args, **kwargs: Path(args[-1]))
    monkeypatch.setattr(tasks, "prepare_sampling_batch", fake_prepare)
    monkeypatch.setattr(tasks, "run_sampling_chain_payload", fake_worker)
    monkeypatch.setattr(tasks, "aggregate_sampling_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "run_local_payload_batch", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not run")))
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 4)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)
    monkeypatch.setattr(tasks, "_update_cluster_entry", lambda *args, **kwargs: None)

    out = tasks.run_simulation_job(
        "job-sim-dist",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {"sampling_method": "gibbs", "potts_model_id": "model-x"},
    )

    frame_jobs = [name for name, _, _ in enqueued if name == "run_simulation_chain_job"]
    aggregate_jobs = [name for name, _, _ in enqueued if name == "run_simulation_aggregate_job"]
    assert len(frame_jobs) == 3
    assert len(aggregate_jobs) == 1
    assert out["status"] == "finished"
    assert out["results"]["summary_npz"] is not None


def test_lambda_sweep_job_falls_back_to_serial_inside_worker(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "lambda-local"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_npz = analysis_dir / "analysis.npz"

    prepared = {
        "analysis_id": "lambda-local-analysis",
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_npz),
        "payloads": [{"row": 0}, {"row": 1}],
        "requested_workers": 6,
    }
    calls = {"max_workers": None}

    def fake_prepare(**kwargs):
        return prepared

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

    def fake_run_local(payloads, worker_fn, max_workers, progress_callback=None, progress_label=None):
        calls["max_workers"] = max_workers
        return [{"labels": [[0, 1]]} for _ in payloads]

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        analysis_npz.write_text("npz", encoding="utf-8")
        return {
            "metadata": {"analysis_id": prepared_obj["analysis_id"], "workers": workers_used},
            "analysis_id": prepared_obj["analysis_id"],
            "analysis_dir": prepared_obj["analysis_dir"],
            "analysis_npz": prepared_obj["analysis_npz"],
            "sample_ids": ["s1", "s2"],
            "sample_names": ["L1", "L2"],
            "series_id": "series-x",
        }

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks, "prepare_lambda_sweep_batch", fake_prepare)
    monkeypatch.setattr(tasks, "run_local_payload_batch", fake_run_local)
    monkeypatch.setattr(tasks, "aggregate_lambda_sweep_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 1)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)

    out = tasks.run_lambda_sweep_job(
        "job-lambda-local",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {
            "model_a_id": "model-a",
            "model_b_id": "model-b",
            "md_sample_id_1": "md-1",
            "md_sample_id_2": "md-2",
            "md_sample_id_3": "md-3",
        },
    )

    assert out["status"] == "finished"
    assert calls["max_workers"] == 1
    assert out["results"]["analysis_type"] == "lambda_sweep"


def test_lambda_sweep_job_uses_distributed_rq_workers(tmp_path, monkeypatch):
    redis_conn = FakeRedis()
    fake_job = FakeJob(connection=redis_conn)
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tmp_path / "lambda-dist"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_npz = analysis_dir / "analysis.npz"
    prepared = {
        "analysis_id": "lambda-dist-analysis",
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_npz),
        "payloads": [{"row": 0}, {"row": 1}, {"row": 2}],
        "requested_workers": 3,
    }
    enqueued = []

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

        def enqueue(self, func, args=(), **kwargs):
            enqueued.append((func.__name__, args, kwargs))
            func(*args)
            return type("FakeEnqueuedJob", (), {"id": kwargs.get("job_id", "fake-job")})()

    def fake_prepare(**kwargs):
        _write_prepared_pickle(prepared)
        return prepared

    def fake_worker(payload):
        return {"labels": [[int(payload["row"]), 1]]}

    def fake_aggregate(prepared_obj, out_rows, workers_used):
        analysis_npz.write_text("npz", encoding="utf-8")
        result = {
            "metadata": {"analysis_id": prepared_obj["analysis_id"], "workers": workers_used},
            "analysis_id": prepared_obj["analysis_id"],
            "analysis_dir": prepared_obj["analysis_dir"],
            "analysis_npz": prepared_obj["analysis_npz"],
            "sample_ids": ["s1", "s2", "s3"],
            "sample_names": ["L1", "L2", "L3"],
            "series_id": "series-y",
        }
        with open(orchestration_paths(Path(prepared_obj["analysis_dir"]))["aggregate"], "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks, "prepare_lambda_sweep_batch", fake_prepare)
    monkeypatch.setattr(tasks, "run_lambda_sweep_payload", fake_worker)
    monkeypatch.setattr(tasks, "aggregate_lambda_sweep_batch", fake_aggregate)
    monkeypatch.setattr(tasks, "run_local_payload_batch", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not run")))
    monkeypatch.setattr(tasks, "_count_queue_workers", lambda queue: 4)
    monkeypatch.setattr(tasks, "Queue", FakeQueue)

    out = tasks.run_lambda_sweep_job(
        "job-lambda-dist",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster"},
        {
            "model_a_id": "model-a",
            "model_b_id": "model-b",
            "md_sample_id_1": "md-1",
            "md_sample_id_2": "md-2",
            "md_sample_id_3": "md-3",
        },
    )

    frame_jobs = [name for name, _, _ in enqueued if name == "run_lambda_sweep_payload_job"]
    aggregate_jobs = [name for name, _, _ in enqueued if name == "run_lambda_sweep_aggregate_job"]
    assert len(frame_jobs) == 3
    assert len(aggregate_jobs) == 1
    assert out["status"] == "finished"
    assert out["results"]["analysis_type"] == "lambda_sweep"
