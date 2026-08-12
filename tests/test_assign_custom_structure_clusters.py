from types import SimpleNamespace

import numpy as np

from scripts.assign_custom_structure_clusters import _fold_samples_for_model_symmetry


def _model(*, enabled: bool, resname: str):
    return SimpleNamespace(
        phase_descriptor_symmetry={
            "enabled": enabled,
            "descriptor": "chi2",
            "descriptor_index": 4,
            "resname": resname,
        }
    )


def test_chi2_symmetry_folds_only_chi2_column_for_symmetric_residue():
    samples = np.asarray([[0.1, 0.2, 0.3, 0.4, -np.pi / 2]], dtype=np.float64)

    folded = _fold_samples_for_model_symmetry(samples, _model(enabled=True, resname="PHE"), residue_resname="PHE")

    np.testing.assert_allclose(folded[:, :4], samples[:, :4])
    np.testing.assert_allclose(folded[:, 4], np.asarray([np.pi]))
    np.testing.assert_allclose(samples[:, 4], np.asarray([-np.pi / 2]))


def test_chi2_symmetry_does_not_fold_non_symmetric_input_residue():
    samples = np.asarray([[0.1, 0.2, 0.3, 0.4, -np.pi / 2]], dtype=np.float64)

    folded = _fold_samples_for_model_symmetry(samples, _model(enabled=True, resname="PHE"), residue_resname="ALA")

    np.testing.assert_allclose(folded, samples)


def test_chi2_symmetry_does_not_fold_leucine():
    samples = np.asarray([[0.1, 0.2, 0.3, 0.4, -np.pi / 2]], dtype=np.float64)

    folded = _fold_samples_for_model_symmetry(
        samples,
        _model(enabled=True, resname="LEU"),
        residue_resname="LEU",
    )

    np.testing.assert_allclose(folded, samples)


def test_chi2_symmetry_rejects_incorrect_model_residue_metadata():
    samples = np.asarray([[0.1, 0.2, 0.3, 0.4, -np.pi / 2]], dtype=np.float64)

    folded = _fold_samples_for_model_symmetry(samples, _model(enabled=True, resname="ALA"), residue_resname="PHE")

    np.testing.assert_allclose(folded, samples)
