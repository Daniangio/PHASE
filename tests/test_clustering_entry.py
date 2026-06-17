import numpy as np

from phase.workflows.clustering import (
    _fold_symmetric_chi2_for_clustering,
    _predict_cluster_adp,
    build_cluster_entry,
)


def test_build_cluster_entry_has_expected_fields():
    entry = build_cluster_entry(
        cluster_id="c1",
        cluster_name="cluster_one",
        state_ids=["A", "B"],
        max_cluster_frames=100,
        random_state=0,
        density_maxk=50,
        density_z="auto",
    )
    assert entry["cluster_id"] == "c1"
    assert entry["name"] == "cluster_one"
    assert entry["state_ids"] == ["A", "B"]
    assert entry["algorithm_params"]["density_maxk"] == 50
    assert entry["algorithm_params"]["density_z"] == "auto"


def test_symmetric_chi2_is_folded_for_clustering_copy_only():
    samples = np.asarray([
        [0.0, 0.0, 0.0, 0.0, -np.pi / 2.0],
        [0.0, 0.0, 0.0, 0.0, np.pi / 2.0],
        [0.0, 0.0, 0.0, 0.0, 3.0 * np.pi / 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.pi],
        [0.0, 0.0, 0.0, 0.0, np.pi / 3.0],
        [0.0, 0.0, 0.0, 0.0, -2.0 * np.pi / 3.0],
    ], dtype=float)

    folded, meta = _fold_symmetric_chi2_for_clustering(samples, "res_88_PHE")

    assert meta["enabled"] is True
    assert meta["applied"] is True
    assert meta["version"] == "chi2_double_angle_v2"
    assert np.allclose(folded[:, 4], [np.pi, np.pi, np.pi, 0.0, 0.0, 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
    assert np.allclose(samples[:, 4], [-np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 2.0, 0.0, np.pi, np.pi / 3.0, -2.0 * np.pi / 3.0])


def test_symmetric_chi2_folding_is_not_applied_to_non_symmetric_residue():
    samples = np.asarray([[0.0, 0.0, 0.0, 0.0, -np.pi / 2.0]], dtype=float)

    folded, meta = _fold_symmetric_chi2_for_clustering(samples, "res_88_ARG")

    assert meta["enabled"] is False
    assert "applied" not in meta
    assert np.allclose(folded[:, 4], samples[:, 4])


def test_predict_cluster_adp_applies_saved_symmetric_chi2_metadata():
    class DummyDP:
        X = np.zeros((5, 5), dtype=float)
        phase_descriptor_symmetry = {"enabled": True, "residue_key": "res_88_TYR"}

        def predict_cluster_ADP(self, emb, *, maxk, density_est, n_jobs):
            assert np.isclose(emb[0, 4], np.pi)
            return np.asarray([[2], [2]], dtype=np.int32), np.asarray([[2], [2]], dtype=np.int32)

    assigned, halo = _predict_cluster_adp(
        DummyDP(),
        np.asarray([[0.0, 0.0, 0.0, 0.0, -np.pi / 2.0]], dtype=float),
        density_maxk=100,
    )

    assert assigned.tolist() == [2]
    assert halo.tolist() == [2]
