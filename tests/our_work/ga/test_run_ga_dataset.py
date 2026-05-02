from pathlib import Path

import yaml
import numpy as np

from our_work.ga.scripts.run_ga_dataset import (
    build_targets_from_config,
    build_work_items,
    main,
    resolve_ga_runtime_config,
    resolve_target_task_specs,
)


def test_resolve_ga_runtime_config_augments_seed_thicknesses_when_enabled():
    config = {
        "data": {
            "thickness_range_nm": {"min": 10, "max": 500, "step": 10},
            "include_seed_thickness_values": True,
            "max_samples_per_target": 100,
        },
        "search": {"population_size": 16, "generations_per_restart": 4, "batch_size": 8},
    }

    runtime = resolve_ga_runtime_config(config)

    assert 440 in runtime["thickness_values_nm"]
    assert max(runtime["thickness_values_nm"]) <= 500
    assert runtime["max_samples_per_target"] == 100
    assert runtime["population_size"] == 16


def test_build_work_items_splits_targets_by_rank():
    assert build_work_items(["a", "b", "c"], rank=1, world_size=2) == ["b"]


def test_resolve_target_task_specs_uses_default_seeded_task_list():
    task_specs = resolve_target_task_specs({})

    assert [task["target_id"] for task in task_specs] == [
        "broad_3_13_high",
        "mid_5_8_high",
        "dual_3_5_8_13_high",
    ]


def test_build_targets_from_config_supports_explicit_task_list_and_filters_ids():
    wavelengths = np.array([2.0, 3.0, 5.0, 8.0, 13.0, 15.0], dtype=np.float32)
    targets = build_targets_from_config(
        wavelengths,
        {
            "tasks": [
                {
                    "target_id": "keep_me",
                    "bands": [{"start_um": 3.0, "end_um": 13.0, "absorption": 1.0}],
                    "random_init": {"layer_count": 3},
                },
                {
                    "target_id": "drop_me",
                    "bands": [{"start_um": 3.0, "end_um": 5.0, "absorption": 0.0}],
                    "random_init": {"layer_count": 3},
                },
            ],
            "include_ids": ["keep_me"],
        },
        material_names=["Si", "Ge", "Au"],
        thickness_values_nm=[10, 20, 30],
        seed=9,
    )

    assert [target.target_id for target in targets] == ["keep_me"]
    assert len(targets[0].seed_tokens) == 3


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
                    "max_samples_per_target": 1,
                },
                "targets": {
                    "tasks": [
                        {
                            "target_id": "broad_3_13_high",
                            "family": "seeded_band",
                            "description": "3-13 um high absorption; wavelengths outside this band are ignored by the loss.",
                            "bands": [{"start_um": 3.0, "end_um": 13.0, "absorption": 1.0}],
                            "seed_tokens": ["YbF3_870", "ZnS_480", "Si_280", "Bi_20", "Ge_130", "Bi_820", "Au_100"],
                        }
                    ]
                },
                "search": {
                    "population_size": 2,
                    "generations_per_restart": 1,
                    "restart_count": 1,
                    "batch_size": 2,
                    "acceptance_floor_mse": 0.005,
                    "elite_fraction": 0.5,
                    "tournament_size": 2,
                    "crossover_rate": 0.8,
                    "material_mutation_rate": 0.0,
                    "thickness_mutation_rate": 0.0,
                    "thickness_mutation_steps": 1,
                    "random_injection_rate": 0.0,
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
