from __future__ import annotations

from pathlib import Path

from our_work._shared.io.config import resolve_config_paths, resolve_repo_path


def test_resolve_repo_path_anchors_relative_paths_to_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()

    resolved = resolve_repo_path("database", project_root=project_root)

    assert resolved == project_root / "database"


def test_resolve_config_paths_converts_nested_relative_path_fields(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    payload = {
        "paths": {
            "database_dir": "database",
            "output_dir": "outputs/our_work/data_gen/v1",
        },
        "data": {
            "dataset_dir": "outputs/our_work/data_gen/v1",
            "vocab_path": "outputs/our_work/data_gen/v1/vocab/vocab.json",
        },
        "model": {
            "note": "do-not-touch",
        },
    }

    resolved = resolve_config_paths(payload, project_root=project_root)

    assert resolved["paths"]["database_dir"] == str(project_root / "database")
    assert resolved["paths"]["output_dir"] == str(project_root / "outputs" / "our_work" / "data_gen" / "v1")
    assert resolved["data"]["dataset_dir"] == str(project_root / "outputs" / "our_work" / "data_gen" / "v1")
    assert resolved["data"]["vocab_path"] == str(
        project_root / "outputs" / "our_work" / "data_gen" / "v1" / "vocab" / "vocab.json"
    )
    assert resolved["model"]["note"] == "do-not-touch"
    assert payload["paths"]["database_dir"] == "database"
