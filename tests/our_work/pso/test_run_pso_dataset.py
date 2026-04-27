from pathlib import Path

import yaml

from our_work.pso.scripts.run_pso_dataset import (
    build_work_items,
    main,
    progress_work_items,
    resolve_pso_runtime_config,
)


def test_resolve_pso_runtime_config_reads_thickness_range_and_search_settings(tmp_path: Path):
    config = {
        "paths": {"database_dir": str(tmp_path / "database"), "output_dir": str(tmp_path / "output")},
        "data": {"layer_counts": [5], "thickness_range_nm": {"min": 10, "max": 30, "step": 10}},
        "targets": {"include_fixed": True, "include_lorentzian": False},
        "search": {
            "population_size": 8,
            "iterations": 2,
            "batch_size": 4,
            "acceptance_mse_threshold": 0.05,
            "max_accepted_per_target_layer": 3,
        },
        "tmm": {"wavelength_range_um": [2.0, 15.0], "num_points": 8},
    }

    runtime = resolve_pso_runtime_config(config)

    assert runtime["thickness_values_nm"] == [10, 20, 30]
    assert runtime["population_size"] == 8
    assert runtime["max_accepted_per_target_layer"] == 3


def test_build_work_items_can_limit_targets_for_smoke_runs():
    work_items = build_work_items(
        target_ids=["a", "b", "c"],
        layer_counts=[5, 6],
        max_targets=2,
        rank=0,
        world_size=1,
    )

    assert work_items == [("a", 5), ("a", 6), ("b", 5), ("b", 6)]


def test_progress_work_items_uses_tqdm_when_available(monkeypatch):
    calls = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        yield from iterable

    monkeypatch.setattr("our_work.pso.scripts.run_pso_dataset.tqdm", fake_tqdm)

    items = list(progress_work_items([("target-a", 5), ("target-b", 6)], rank=1, world_size=4))

    assert items == [("target-a", 5), ("target-b", 6)]
    assert calls[0]["total"] == 2
    assert calls[0]["desc"] == "pso rank 1/4"


def test_run_pso_dataset_main_writes_dataset_with_mocked_search(monkeypatch, tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    (database_dir / "Ge.csv").write_text("wl,n,k\n2.0,4.0,0.1\n15.0,4.0,0.1\n", encoding="utf-8")

    config_path = tmp_path / "pso.yaml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 123,
                "paths": {"database_dir": str(database_dir), "output_dir": str(output_dir)},
                "data": {
                    "layer_counts": [1],
                    "thickness_range_nm": {"min": 10, "max": 10, "step": 10},
                    "train_ratio": 1.0,
                    "val_ratio": 0.0,
                },
                "targets": {"include_fixed": True, "include_lorentzian": False, "max_targets": 1},
                "search": {
                    "population_size": 2,
                    "iterations": 1,
                    "batch_size": 2,
                    "acceptance_mse_threshold": 0.5,
                    "max_accepted_per_target_layer": 1,
                    "max_stagnant_iterations": 1,
                    "max_restarts": 1,
                    "device": "cpu",
                },
                "tmm": {
                    "wavelength_range_um": [2.0, 15.0],
                    "num_points": 4,
                    "incident_angle": 0.0,
                    "polarization": 0,
                    "tolerance": 0.001,
                    "complex_dtype": "complex128",
                    "batch_size": 2,
                    "device": "cpu",
                },
                "shards": {"records_per_shard": 10},
                "distributed": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    main(["--config", str(config_path)])

    assert (output_dir / "splits" / "split_manifest.json").exists()
    assert (output_dir / "stats" / "summary.json").exists()
