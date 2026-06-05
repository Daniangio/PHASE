import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from phase.potts.potts_model import PottsModel, save_potts_model, zero_sum_gauge_model
from phase.potts.spectral_analysis import frobenius_coupling_matrix, upsert_hamiltonian_spectral_batch
from phase.services.project_store import DescriptorState, ProjectStore


def _write_state_model(model_dir: Path, system_dir: Path, model_id: str, state_id: str, coupling_scale: float):
    target = model_dir / model_id
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / f"model_{state_id}.npz"
    model = PottsModel(
        h=[np.asarray([0.0, 0.2]), np.asarray([0.1, -0.1]), np.asarray([0.3, 0.0])],
        J={
            (0, 1): coupling_scale * np.asarray([[0.2, -0.4], [0.1, 0.3]]),
            (1, 2): coupling_scale * np.asarray([[-0.1, 0.2], [0.0, -0.3]]),
        },
        edges=[(0, 1), (1, 2)],
    )
    save_potts_model(model, model_path)
    meta = {
        "model_id": model_id,
        "name": f"model_{state_id}",
        "path": str(model_path.relative_to(system_dir)),
        "params": {"fit_mode": "standard", "state_ids": [state_id]},
    }
    (target / "model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_frobenius_matrix_uses_zero_sum_gauge():
    model = PottsModel(
        h=[np.asarray([0.0, 1.0]), np.asarray([0.5, -0.2])],
        J={(0, 1): np.asarray([[1.0, 2.0], [3.0, 4.0]])},
        edges=[(0, 1)],
    )
    gauged = zero_sum_gauge_model(model)
    expected = float(np.sqrt(np.sum(np.asarray(gauged.J[(0, 1)]) ** 2)))
    F = frobenius_coupling_matrix(model)
    assert F.shape == (2, 2)
    assert F[0, 0] == 0
    assert np.isclose(F[0, 1], expected)
    assert np.isclose(F[1, 0], expected)


def test_hamiltonian_spectral_batch_is_incremental(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.states = {
        "active": DescriptorState(state_id="active", name="Active"),
        "inactive": DescriptorState(state_id="inactive", name="Inactive"),
        "pas": DescriptorState(state_id="pas", name="PAS"),
    }
    store.save_system(system)
    cluster_id = "cluster1"
    dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    np.savez_compressed(
        dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2", "res_3"], dtype=str),
    )
    _write_state_model(dirs["potts_models_dir"], system_dir, "m-active", "active", 1.0)
    _write_state_model(dirs["potts_models_dir"], system_dir, "m-inactive", "inactive", 0.7)
    _write_state_model(dirs["potts_models_dir"], system_dir, "m-pas", "pas", 1.4)

    first = upsert_hamiltonian_spectral_batch(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        state_ids=["active", "inactive"],
        top_k=2,
    )
    assert first["single_count"] == 2
    assert first["pair_count"] == 1

    pair_root = dirs["cluster_dir"] / "analyses" / "hamiltonian_spectral_pair"
    assert (pair_root / "pair_active__inactive" / "analysis.npz").exists()
    with np.load(pair_root / "pair_active__inactive" / "analysis.npz", allow_pickle=False) as data:
        assert np.asarray(data["matrix"]).shape == (3, 3)
        assert np.asarray(data["top_eigenvectors"]).shape == (2, 3)

    second = upsert_hamiltonian_spectral_batch(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        state_ids=["pas"],
        top_k=2,
    )
    assert second["single_count"] == 1
    assert second["pair_count"] == 2
    assert (pair_root / "pair_active__pas" / "analysis.npz").exists()
    assert (pair_root / "pair_inactive__pas" / "analysis.npz").exists()
