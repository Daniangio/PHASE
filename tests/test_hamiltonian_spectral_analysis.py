import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from phase.potts.potts_model import PottsModel, save_potts_model, zero_sum_gauge_model
from phase.potts.sample_io import save_sample_npz
from phase.potts.spectral_analysis import (
    compute_spectral_intersection_analysis,
    compute_piston_ligand_projection_analysis,
    frobenius_coupling_matrix,
    normalized_laplacian,
    upsert_hamiltonian_spectral_batch,
)
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


def test_normalized_laplacian_uses_nonnegative_adjacency():
    A = np.asarray([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    L, degree = normalized_laplacian(A)
    assert np.allclose(degree, [2.0, 3.0, 1.0])
    assert np.isclose(L[0, 0], 1.0)
    assert np.isclose(L[0, 1], -2.0 / np.sqrt(6.0))
    assert np.isclose(L[1, 2], -1.0 / np.sqrt(3.0))
    assert np.isclose(L[0, 2], 0.0)


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

    single_root = dirs["cluster_dir"] / "analyses" / "hamiltonian_spectral_single"
    with np.load(single_root / "single_active" / "analysis.npz", allow_pickle=False) as data:
        assert np.asarray(data["laplacian_source_matrix"]).shape == (3, 3)
        assert np.asarray(data["laplacian_embedding"]).ndim == 2
        assert np.asarray(data["community_ids"]).shape == (3,)
        assert np.asarray(data["community_core_mask"]).shape == (3,)
        assert np.asarray(data["community_sizes"]).shape[1] == 2
        assert np.asarray(data["community_core_sizes"]).shape[1] == 2
        assert np.asarray(data["community_matrix_order"]).shape == (3,)
        assert np.asarray(data["community_interaction_matrix"]).ndim == 2

    pair_root = dirs["cluster_dir"] / "analyses" / "hamiltonian_spectral_pair"
    assert (pair_root / "pair_active__inactive" / "analysis.npz").exists()
    with np.load(pair_root / "pair_active__inactive" / "analysis.npz", allow_pickle=False) as data:
        assert np.asarray(data["matrix"]).shape == (3, 3)
        assert np.asarray(data["top_eigenvectors"]).shape == (2, 3)
        assert np.asarray(data["laplacian_source_matrix"]).shape == (3, 3)
        assert np.asarray(data["laplacian_matrix"]).shape == (3, 3)
        assert np.asarray(data["laplacian_degree"]).shape == (3,)
        assert np.asarray(data["laplacian_top_eigenvectors"]).shape == (2, 3)
        assert np.asarray(data["laplacian_top_indices"]).shape == (2,)
        assert np.asarray(data["community_ids"]).shape == (3,)
        assert np.asarray(data["community_core_mask"]).shape == (3,)
        assert np.asarray(data["community_interaction_matrix"]).ndim == 2

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

    intersection = compute_spectral_intersection_analysis(
        store=store,
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        single_analysis_id="single_active",
        pair_analysis_id="pair_active__inactive",
        min_group_size=2,
    )
    assert intersection["created"] is True
    meta = intersection["metadata"]
    assert meta["analysis_type"] == "hamiltonian_spectral_intersection"
    assert meta["summary"]["n_residues"] == 3
    intersection_npz = dirs["cluster_dir"] / "analyses" / "hamiltonian_spectral_intersection" / meta["analysis_id"] / "analysis.npz"
    with np.load(intersection_npz, allow_pickle=False) as data:
        assert np.asarray(data["structural_community_ids"]).shape == (3,)
        assert np.asarray(data["functional_community_ids"]).shape == (3,)
        assert np.asarray(data["piston_ids"]).shape == (3,)
        assert np.asarray(data["residue_class_codes"]).shape == (3,)
        assert np.asarray(data["structural_core_mask"]).shape == (3,)
        assert np.asarray(data["functional_core_mask"]).shape == (3,)
        assert np.asarray(data["class_counts"]).shape == (5, 2)
        assert "piston_members_json" in data.files


def test_core_strict_intersection_and_ligand_projection(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.states = {
        "active": DescriptorState(state_id="active", name="Active"),
        "inactive": DescriptorState(state_id="inactive", name="Inactive"),
    }
    store.save_system(system)
    cluster_id = "cluster1"
    dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    root = dirs["cluster_dir"] / "analyses"
    single_dir = root / "hamiltonian_spectral_single" / "single_manual"
    pair_dir = root / "hamiltonian_spectral_pair" / "pair_manual"
    single_dir.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    residue_keys = np.asarray(["res_1", "res_2", "res_3", "res_4"], dtype=str)
    np.savez_compressed(
        single_dir / "analysis.npz",
        residue_keys=residue_keys,
        community_ids=np.asarray([1, 1, 1, 2], dtype=np.int32),
        community_core_mask=np.asarray([True, True, False, True]),
    )
    (single_dir / "analysis_metadata.json").write_text(
        json.dumps({"analysis_id": "single_manual", "analysis_type": "hamiltonian_spectral_single", "mode": "single", "state_id": "active", "state_name": "Active"}),
        encoding="utf-8",
    )
    np.savez_compressed(
        pair_dir / "analysis.npz",
        residue_keys=residue_keys,
        community_ids=np.asarray([4, 4, 4, 5], dtype=np.int32),
        community_core_mask=np.asarray([True, True, True, False]),
        laplacian_top_eigenvectors=np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 0.1, 2.0, 0.0]], dtype=np.float32),
    )
    (pair_dir / "analysis_metadata.json").write_text(
        json.dumps({"analysis_id": "pair_manual", "analysis_type": "hamiltonian_spectral_pair", "mode": "pair", "state_a_id": "active", "state_b_id": "inactive", "state_a_name": "Active", "state_b_name": "Inactive"}),
        encoding="utf-8",
    )
    out = compute_spectral_intersection_analysis(
        store=store,
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        single_analysis_id="single_manual",
        pair_analysis_id="pair_manual",
        min_group_size=2,
    )
    npz_path = Path(out["analysis_npz"])
    with np.load(npz_path, allow_pickle=False) as data:
        assert np.asarray(data["piston_ids"]).tolist() == [1, 1, 0, 0]
        # res_3 is functional core only -> transient switch; res_4 is structural core only -> scaffold.
        assert np.asarray(data["residue_class_codes"]).tolist() == [3, 3, 2, 1]
        assert np.asarray(data["composite_group_sizes"]).tolist() == [2]

    sample_dir = dirs["cluster_dir"] / "samples" / "lig1"
    sample_path = sample_dir / "sample.npz"
    labels = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    save_sample_npz(sample_path, labels=labels)
    rel_sample = sample_path.relative_to(data_root / "projects" / "proj" / "systems" / "sys")
    store.save_sample_entry(
        "proj",
        "sys",
        cluster_id,
        "lig1",
        {"sample_id": "lig1", "name": "Ligand 1", "type": "md_eval", "path": str(rel_sample)},
    )
    proj = compute_piston_ligand_projection_analysis(
        store=store,
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        intersection_analysis_id=out["metadata"]["analysis_id"],
        sample_ids=["lig1"],
    )
    with np.load(proj["analysis_npz"], allow_pickle=False) as data:
        assert np.asarray(data["piston_scores"]).shape == (1, 1)
        assert np.asarray(data["piston_scores"])[0, 0] > 0
        assert np.asarray(data["piston_vector_indices"]).tolist() == [0]
