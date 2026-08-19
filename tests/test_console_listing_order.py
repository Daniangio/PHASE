import json
from pathlib import Path

from phase.scripts.offline_browser import _print_rows
from phase.services.project_store import ProjectStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_offline_rows_are_sorted_by_human_label(capsys):
    _print_rows([
        ["000-hash", "zeta"],
        ["fff-hash", "Alpha"],
        ["aaa-hash", "beta"],
    ])

    assert capsys.readouterr().out.splitlines() == [
        "fff-hash|Alpha",
        "aaa-hash|beta",
        "000-hash|zeta",
    ]


def test_store_models_and_samples_are_sorted_by_name(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    cluster_dir = tmp_path / "projects" / "project" / "systems" / "system" / "clusters" / "cluster"

    for item_id, name in (("000-model", "Zulu"), ("fff-model", "alpha")):
        _write_json(
            cluster_dir / "potts_models" / item_id / "model_metadata.json",
            {"model_id": item_id, "name": name},
        )
    for item_id, name in (("000-sample", "Zulu trajectory"), ("fff-sample", "Alpha trajectory")):
        _write_json(
            cluster_dir / "samples" / item_id / "sample_metadata.json",
            {"sample_id": item_id, "name": name},
        )

    assert [row["model_id"] for row in store.list_potts_models("project", "system", "cluster")] == [
        "fff-model",
        "000-model",
    ]
    assert [row["sample_id"] for row in store.list_samples("project", "system", "cluster")] == [
        "fff-sample",
        "000-sample",
    ]
