from phase.workflows.clustering import build_cluster_entry


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
