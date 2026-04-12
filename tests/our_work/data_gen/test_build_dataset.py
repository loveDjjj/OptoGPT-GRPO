import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from our_work.data_gen.pipeline.build_dataset import build_small_dataset
from our_work.data_gen.scripts.run_build_dataset import (
    main as run_build_dataset_main,
    _assign_layer_counts,
    resolve_analysis_runtime_config,
    resolve_data_gen_runtime_config,
    resolve_thickness_values_nm,
    merge_rank_split_manifests,
)


def test_build_small_dataset_writes_manifest_and_shard(tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    pd.DataFrame({"wl": [2.0, 15.0], "n": [1.4, 1.4], "k": [0.0, 0.0]}).to_csv(database_dir / "SiO2.csv", index=False)
    pd.DataFrame({"wl": [2.0, 15.0], "n": [4.0, 4.0], "k": [0.1, 0.1]}).to_csv(database_dir / "Ge.csv", index=False)

    output_dir = tmp_path / "outputs"
    build_small_dataset(
        output_dir=output_dir,
        database_path=str(database_dir),
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[10, 20],
        layer_counts=[5],
        samples_per_bucket=2,
        num_points=8,
        wavelength_range_um=(2.0, 15.0),
    )

    assert (output_dir / "shards" / "shard-00000.parquet").exists()
    payload = json.loads((output_dir / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    assert payload["train"] == ["shard-00000.parquet"]


def test_resolve_thickness_values_nm_expands_inclusive_range_config():
    values = resolve_thickness_values_nm(
        {
            "thickness_range_nm": {
                "min": 10,
                "max": 500,
                "step": 10,
            }
        }
    )

    assert values[:3] == [10, 20, 30]
    assert values[-3:] == [480, 490, 500]
    assert len(values) == 50


def test_resolve_thickness_values_nm_rejects_ambiguous_or_invalid_range():
    with pytest.raises(ValueError, match="thickness_range_nm"):
        resolve_thickness_values_nm(
            {
                "thickness_values_nm": [10, 20],
                "thickness_range_nm": {"min": 10, "max": 20, "step": 10},
            }
        )


def test_resolve_data_gen_runtime_config_reads_sampling_and_tmm_batch_sizes():
    runtime = resolve_data_gen_runtime_config(
        {
            "data": {
                "thickness_range_nm": {"min": 10, "max": 30, "step": 10},
                "layer_counts": [5],
                "samples_per_bucket": 4,
            },
            "sampling": {
                "device": "cuda:0",
                "batch_size": 8,
                "max_duplicate_retry": 9,
            },
            "tmm": {
                "num_points": 8,
                "wavelength_range_um": [2.0, 15.0],
                "incident_angle": 0.0,
                "polarization": 0,
                "tolerance": 1e-3,
                "complex_dtype": "complex128",
                "batch_size": 2,
            },
        }
    )

    assert runtime["thickness_values_nm"] == [10, 20, 30]
    assert runtime["sampling_device"] == "cuda:0"
    assert runtime["sampling_batch_size"] == 8
    assert runtime["max_duplicate_retry"] == 9
    assert runtime["tmm_batch_size"] == 2


def test_resolve_analysis_runtime_config_reads_spectrum_settings():
    runtime = resolve_analysis_runtime_config(
        {
            "analysis": {
                "enabled": True,
                "auto_after_build": True,
                "batch_size": 128,
                "scopes": ["all", "train"],
                "structure": {"enabled": True},
                "spectrum": {
                    "enabled": True,
                    "device": "cuda:0",
                    "pca_components": 6,
                    "cluster_count": 5,
                    "cluster_fit_samples": 100,
                    "cluster_iterations": 7,
                    "scatter_max_points": 50,
                },
            }
        }
    )

    assert runtime["enabled"] is True
    assert runtime["auto_after_build"] is True
    assert runtime["batch_size"] == 128
    assert runtime["scopes"] == ["all", "train"]
    assert runtime["spectrum_device"] == "cuda:0"
    assert runtime["pca_components"] == 6
    assert runtime["cluster_count"] == 5


def test_resolve_analysis_runtime_config_defaults_to_all_scope_only():
    runtime = resolve_analysis_runtime_config({"analysis": {}})

    assert runtime["scopes"] == ["all"]
    assert runtime["spectrum_engine"] == "rapids"
    assert runtime["save_split_analysis"] is False


def test_assign_layer_counts_round_robins_layer_buckets():
    layer_counts = [5, 6, 7, 8, 9, 10]

    assert _assign_layer_counts(layer_counts, rank=0, world_size=4, shard_mode="layer_bucket") == [5, 9]
    assert _assign_layer_counts(layer_counts, rank=1, world_size=4, shard_mode="layer_bucket") == [6, 10]
    assert _assign_layer_counts(layer_counts, rank=2, world_size=4, shard_mode="layer_bucket") == [7]
    assert _assign_layer_counts(layer_counts, rank=3, world_size=4, shard_mode="layer_bucket") == [8]
    assert _assign_layer_counts(layer_counts, rank=7, world_size=8, shard_mode="layer_bucket") == []


def test_merge_rank_split_manifests_combines_rank_outputs(tmp_path: Path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "split_manifest.rank00.json").write_text(
        json.dumps({"train": ["rank00-shard-00000.parquet"], "val": ["rank00-shard-00001.parquet"], "test": []}),
        encoding="utf-8",
    )
    (splits_dir / "split_manifest.rank01.json").write_text(
        json.dumps({"train": ["rank01-shard-00000.parquet"], "val": [], "test": ["rank01-shard-00001.parquet"]}),
        encoding="utf-8",
    )

    merged = merge_rank_split_manifests(tmp_path, world_size=2)

    assert merged["train"] == ["rank00-shard-00000.parquet", "rank01-shard-00000.parquet"]
    assert merged["val"] == ["rank00-shard-00001.parquet"]
    assert merged["test"] == ["rank01-shard-00001.parquet"]
    written = json.loads((splits_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert written == merged


def test_build_small_dataset_chunks_tmm_calls(monkeypatch, tmp_path: Path):
    calls: list[int] = []

    def fake_sample_batch(**kwargs):
        return [
            ["Ge_10", "SiO2_20"],
            ["Ge_10", "SiO2_20"],
            ["SiO2_20", "Ge_10"],
            ["Ge_20", "SiO2_10"],
        ]

    def fake_simulate(groups, **kwargs):
        calls.append(len(groups))
        reflections = [np.zeros((8,), dtype=np.float32) for _ in groups]
        transmissions = [np.ones((8,), dtype=np.float32) for _ in groups]
        ok_mask = np.ones((len(groups),), dtype=np.bool_)
        return np.arange(8, dtype=np.float32), reflections, transmissions, ok_mask

    monkeypatch.setattr("our_work.data_gen.pipeline.build_dataset.sample_structure_token_batch", fake_sample_batch)
    monkeypatch.setattr("our_work.data_gen.pipeline.build_dataset.simulate_structure_batch", fake_simulate)

    output_dir = tmp_path / "outputs"
    build_small_dataset(
        output_dir=output_dir,
        database_path="database",
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[10, 20],
        layer_counts=[2],
        samples_per_bucket=3,
        sampling_batch_size=4,
        tmm_batch_size=2,
        max_duplicate_retry=4,
        sampling_device="cpu",
        num_points=8,
        wavelength_range_um=(2.0, 15.0),
        show_progress=False,
    )

    payload = json.loads((output_dir / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    assert payload["train"] == ["shard-00000.parquet"]
    assert calls == [2, 1]

    with pytest.raises(ValueError, match="step"):
        resolve_thickness_values_nm(
            {
                "thickness_range_nm": {
                    "min": 10,
                    "max": 500,
                    "step": 0,
                }
            }
        )

    with pytest.raises(ValueError, match="divisible"):
        resolve_thickness_values_nm(
            {
                "thickness_range_nm": {
                    "min": 10,
                    "max": 495,
                    "step": 10,
                }
            }
        )


def test_run_build_dataset_main_triggers_auto_analysis(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("dummy", encoding="utf-8")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "our_work.data_gen.scripts.run_build_dataset.load_yaml_config",
        lambda *args, **kwargs: {
            "seed": 42,
            "paths": {"database_dir": str(tmp_path / "database"), "output_dir": str(tmp_path / "output")},
            "data": {"layer_counts": [5], "samples_per_bucket": 1, "thickness_range_nm": {"min": 10, "max": 10, "step": 10}},
            "sampling": {"device": "cpu", "batch_size": 1, "max_duplicate_retry": 1},
            "tmm": {"wavelength_range_um": [2.0, 15.0], "num_points": 8, "incident_angle": 0.0, "polarization": 0, "tolerance": 0.001, "complex_dtype": "complex128", "batch_size": 1},
            "shards": {"records_per_shard": 1},
            "splits": {"train_ratio": 1.0, "val_ratio": 0.0},
            "analysis": {"enabled": True, "auto_after_build": True, "scopes": ["all"]},
        },
    )
    monkeypatch.setattr("our_work.data_gen.scripts.run_build_dataset.build_material_registry", lambda *args, **kwargs: type("Registry", (), {"material_names": ["Ge"]})())
    monkeypatch.setattr("our_work.data_gen.scripts.run_build_dataset.build_small_dataset", lambda **kwargs: captured.update({"build": kwargs}))
    monkeypatch.setattr("our_work.data_gen.scripts.run_build_dataset.analyze_dataset", lambda **kwargs: captured.update({"analysis": kwargs}))

    run_build_dataset_main(["--config", str(config_path)])

    assert "build" in captured
    assert "analysis" in captured
    assert captured["analysis"]["scopes"] == ["all"]
