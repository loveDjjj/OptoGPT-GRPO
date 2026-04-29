from pathlib import Path

import yaml

from our_work.ga.scripts.run_ga_dataset import build_work_items, main, resolve_ga_runtime_config


def test_resolve_ga_runtime_config_augments_seed_thicknesses_when_enabled():
    config = {
        "data": {
            "thickness_range_nm": {"min": 10, "max": 500, "step": 10},
            "include_seed_thickness_values": True,
            "target_sample_count": 100,
        },
        "search": {"population_size": 16, "generations": 4, "batch_size": 8},
    }

    runtime = resolve_ga_runtime_config(config)

    assert 870 in runtime["thickness_values_nm"]
    assert runtime["target_sample_count"] == 100
    assert runtime["population_size"] == 16


def test_build_work_items_splits_targets_by_rank():
    assert build_work_items(["a", "b", "c"], rank=1, world_size=2) == ["b"]


def test_run_ga_dataset_main_writes_dataset_with_tiny_tmm_smoke(tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    for material in ["YbF3", "ZnS", "Si", "Bi", "Ge", "Au", "SiO2", "MgF2"]:
        (database_dir / f"{material}.csv").write_text("wl,n,k\n2.0,2.0,0.1\n15.0,2.0,0.1\n", encoding="utf-8")

    config_path = tmp_path / "ga.yaml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 123,
                "paths": {"database_dir": str(database_dir), "output_dir": str(output_dir)},
                "data": {
                    "thickness_range_nm": {"min": 10, "max": 500, "step": 10},
                    "include_seed_thickness_values": True,
                    "train_ratio": 1.0,
                    "val_ratio": 0.0,
                    "target_sample_count": 1,
                },
                "targets": {"include_ids": ["broad_3_13_high"]},
                "search": {
                    "population_size": 2,
                    "generations": 1,
                    "batch_size": 2,
                    "acceptance_mse_threshold": 0.005,
                    "elite_fraction": 0.5,
                    "tournament_size": 2,
                    "crossover_rate": 0.8,
                    "material_mutation_rate": 0.0,
                    "thickness_mutation_rate": 0.0,
                    "thickness_mutation_steps": 1,
                    "random_injection_rate": 0.0,
                    "max_stagnant_generations": 1,
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
                "visualization": {"enabled": False},
                "distributed": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    main(["--config", str(config_path)])

    assert (output_dir / "splits" / "split_manifest.json").exists()
    assert (output_dir / "stats" / "summary.json").exists()
