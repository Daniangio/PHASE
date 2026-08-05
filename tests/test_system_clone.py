import json
from pathlib import Path

import numpy as np

from phase.services.project_store import DescriptorState, ProjectStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_clone_system_copies_reusable_artifacts_only(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Project", project_id="proj")
    source = store.create_system("proj", name="Source", description="Inherited", system_id="source")
    source_dir = tmp_path / "projects" / "proj" / "systems" / "source"

    state_dir = source_dir / "states" / "active"
    state_dir.mkdir(parents=True)
    (state_dir / "structure.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")
    np.savez_compressed(state_dir / "descriptors.npz", values=np.zeros((2, 3)))
    (state_dir / "trajectory.xtc").write_bytes(b"trajectory")
    source.states["active"] = DescriptorState(
        state_id="active",
        name="Active",
        storage_key="active",
        pdb_file="states/active/structure.pdb",
        descriptor_file="states/active/descriptors.npz",
        trajectory_file="states/active/trajectory.xtc",
    )
    store.save_system(source)

    cluster_dir = source_dir / "clusters" / "cluster-1"
    cluster_dir.mkdir(parents=True)
    np.savez_compressed(cluster_dir / "cluster.npz", labels=np.asarray([0, 1]))
    _write_json(
        cluster_dir / "cluster_metadata.json",
        {
            "cluster_id": "cluster-1",
            "name": "Cluster",
            "path": "clusters/cluster-1/cluster.npz",
            "system_id": "source",
        },
    )
    (cluster_dir / "models").mkdir()
    (cluster_dir / "models" / "density.pkl").write_bytes(b"density-model")

    model_dir = cluster_dir / "potts_models" / "model-1"
    model_dir.mkdir(parents=True)
    np.savez_compressed(model_dir / "model.npz", h=np.zeros((2, 2)))
    _write_json(
        model_dir / "model_metadata.json",
        {
            "model_id": "model-1",
            "name": "Model",
            "system_id": "source",
            "path": str(model_dir / "model.npz"),
        },
    )

    md_dir = cluster_dir / "samples" / "md-1"
    md_dir.mkdir(parents=True)
    np.savez_compressed(md_dir / "sample.npz", labels=np.asarray([[0, 1]]))
    _write_json(
        md_dir / "sample_metadata.json",
        {
            "sample_id": "md-1",
            "type": "md_eval",
            "state_id": "active",
            "path": "clusters/cluster-1/samples/md-1/sample.npz",
        },
    )

    potts_sample_dir = cluster_dir / "samples" / "sample-1"
    potts_sample_dir.mkdir(parents=True)
    np.savez_compressed(potts_sample_dir / "sample.npz", labels=np.asarray([[1, 0]]))
    _write_json(
        potts_sample_dir / "sample_metadata.json",
        {"sample_id": "sample-1", "type": "potts_sampling", "model_id": "model-1"},
    )
    (cluster_dir / "analyses" / "analysis-1").mkdir(parents=True)
    (cluster_dir / "analyses" / "analysis-1" / "result.npz").write_bytes(b"analysis")
    (cluster_dir / "_orchestration").mkdir()
    (source_dir / "results" / "jobs").mkdir(parents=True)
    (source_dir / "ui_setups").mkdir()

    cloned = store.clone_system("proj", "source", name="Clone", system_id="clone")
    clone_dir = tmp_path / "projects" / "proj" / "systems" / "clone"

    assert cloned.system_id == "clone"
    assert cloned.name == "Clone"
    assert cloned.description == "Inherited"
    assert cloned.states["active"].name == "Active"
    assert len(cloned.metastable_clusters) == 1
    assert [model["model_id"] for model in cloned.metastable_clusters[0]["potts_models"]] == ["model-1"]
    assert [sample["sample_id"] for sample in cloned.metastable_clusters[0]["samples"]] == ["md-1"]
    assert (clone_dir / "states" / "active" / "trajectory.xtc").exists()
    assert (clone_dir / "clusters" / "cluster-1" / "models" / "density.pkl").exists()
    assert (clone_dir / "clusters" / "cluster-1" / "potts_models" / "model-1" / "model.npz").exists()
    assert (clone_dir / "clusters" / "cluster-1" / "samples" / "md-1" / "sample.npz").exists()
    assert not (clone_dir / "clusters" / "cluster-1" / "samples" / "sample-1").exists()
    assert not (clone_dir / "clusters" / "cluster-1" / "analyses").exists()
    assert not (clone_dir / "clusters" / "cluster-1" / "_orchestration").exists()
    assert not (clone_dir / "results").exists()
    assert not (clone_dir / "ui_setups").exists()

    model_meta = json.loads(
        (clone_dir / "clusters" / "cluster-1" / "potts_models" / "model-1" / "model_metadata.json").read_text()
    )
    assert model_meta["system_id"] == "clone"
    assert "/systems/clone/" in model_meta["path"]
    assert "/systems/source/" not in model_meta["path"]
    assert store.get_project("proj").systems == ["source", "clone"]
    assert (source_dir / "clusters" / "cluster-1" / "samples" / "sample-1").exists()
