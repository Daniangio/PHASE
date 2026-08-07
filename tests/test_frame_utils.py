import numpy as np

from phase.common.frame_utils import select_frames_per_cluster


def test_select_frames_per_cluster_limits_each_cluster():
    labels = np.asarray([[0], [0], [0], [1], [1], [2]], dtype=np.int32)
    frames, counts = select_frames_per_cluster(
        labels,
        np.arange(6),
        residue_index=0,
        max_per_cluster=2,
    )
    assert frames == [0, 2, 3, 4, 5]
    assert counts == {0: 2, 1: 2, 2: 1}


def test_select_frames_per_cluster_balances_global_cap():
    labels = np.asarray([[0], [0], [0], [1], [1], [1]], dtype=np.int32)
    frames, counts = select_frames_per_cluster(
        labels,
        np.arange(6),
        residue_index=0,
        max_per_cluster=3,
        max_total=4,
    )
    assert len(frames) == 4
    assert counts == {0: 2, 1: 2}

