from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from our_work.eval.dataset import select_split_shard_paths
from our_work.eval.pipeline import run_eval_suite
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def _write_tiny_checkpoint(base: Path) -> Path:
    checkpoint_dir = base / "checkpoint-1"
    model = SpectralGPTForCausalLM(
        SpectralGPTConfig(
            vocab_size=5,
            spectrum_dim=8,
            prefix_length=2,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    model.save_pretrained(checkpoint_dir)
    SpectralStructureTokenizer(tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"]).save_pretrained(checkpoint_dir)
    return checkpoint_dir


def _write_tiny_dataset(base: Path) -> Path:
    dataset_dir = base / "dataset"
    (dataset_dir / "shards").mkdir(parents=True)
    (dataset_dir / "splits").mkdir(parents=True)
    records = [
        {
            "sample_id": "train-000",
            "layer_count": 1,
            "structure_tokens": ["Ge_10"],
            "spectrum_rt": [0.1] * 4 + [0.9] * 4,
        },
        {
            "sample_id": "val-000",
            "layer_count": 1,
            "structure_tokens": ["Ge_10"],
            "spectrum_rt": [0.2] * 4 + [0.8] * 4,
        },
    ]
    pd.DataFrame.from_records(records[:1]).to_parquet(dataset_dir / "shards" / "shard-train.parquet", index=False)
    pd.DataFrame.from_records(records[1:]).to_parquet(dataset_dir / "shards" / "shard-val.parquet", index=False)
    (dataset_dir / "splits" / "split_manifest.json").write_text(
        json.dumps({"train": ["shard-train.parquet"], "val": ["shard-val.parquet"], "test": []}),
        encoding="utf-8",
    )
    return dataset_dir


def _write_multi_shard_dataset(base: Path) -> Path:
    dataset_dir = base / "dataset_multi"
    (dataset_dir / "shards").mkdir(parents=True)
    (dataset_dir / "splits").mkdir(parents=True)
    shard_names = []
    for idx in range(4):
        shard_name = f"shard-train-{idx:02d}.parquet"
        shard_names.append(shard_name)
        pd.DataFrame.from_records(
            [
                {
                    "sample_id": f"train-{idx:02d}",
                    "layer_count": 1,
                    "structure_tokens": ["Ge_10"],
                    "spectrum_rt": [0.1] * 4 + [0.9] * 4,
                }
            ]
        ).to_parquet(dataset_dir / "shards" / shard_name, index=False)
    (dataset_dir / "splits" / "split_manifest.json").write_text(
        json.dumps({"train": shard_names, "val": [], "test": []}),
        encoding="utf-8",
    )
    return dataset_dir


def _write_tiny_database(base: Path) -> Path:
    database_dir = base / "database"
    database_dir.mkdir(parents=True)
    pd.DataFrame({"wl": [2.0, 15.0], "n": [4.0, 4.0], "k": [0.1, 0.1]}).to_csv(database_dir / "Ge.csv", index=False)
    return database_dir


def test_run_eval_suite_writes_expected_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = _write_tiny_checkpoint(tmp_path)
    dataset_dir = _write_tiny_dataset(tmp_path)
    database_dir = _write_tiny_database(tmp_path)

    monkeypatch.setattr(
        "our_work.eval.pipeline.generate_structure_tokens",
        lambda **kwargs: [["Ge_10"] for _ in range(kwargs["spectra"].shape[0])],
    )

    def fake_simulate(groups, **kwargs):
        reflections = [[0.1] * 4 for _ in groups]
        transmissions = [[0.9] * 4 for _ in groups]
        return [2.0, 6.0, 10.0, 15.0], reflections, transmissions, [True] * len(groups)

    monkeypatch.setattr("our_work.eval.pipeline.simulate_structure_batch", fake_simulate)

    payload = run_eval_suite(
        {
            "experiment": {"name": "eval_suite_test"},
            "paths": {
                "checkpoint_dir": str(checkpoint_dir),
                "dataset_dir": str(dataset_dir),
                "database_dir": str(database_dir),
                "output_dir": str(tmp_path / "outputs"),
            },
            "data": {
                "splits": ["train", "val"],
                "sample_mode": "random",
                "max_samples_per_split": {"train": 1, "val": 1},
                "seed": 42,
            },
            "inference": {"device": "cpu", "batch_size": 2, "max_new_tokens": 2},
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
            "plots": {
                "enable": True,
                "worst_count": 1,
                "best_count": 1,
                "mean_count": 1,
                "save_histograms": True,
                "save_split_comparison": True,
                "save_sample_plots": True,
            },
            "outputs": {
                "save_jsonl": True,
                "save_summary_json": True,
                "save_split_summary_json": True,
                "save_selected_samples_json": True,
                "save_spectra_in_results": False,
            },
        }
    )

    run_dir = Path(payload["run_dir"])
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "split_summaries.json").exists()
    assert (run_dir / "selected_samples.json").exists()
    assert (run_dir / "results" / "train.jsonl").exists()
    assert (run_dir / "results" / "val.jsonl").exists()
    assert (run_dir / "plots" / "comparison" / "train_vs_val_rmse.png").exists()
    assert any((run_dir / "samples" / "train").rglob("*.png"))
    summary = json.loads((run_dir / "split_summaries.json").read_text(encoding="utf-8"))
    assert summary["train"]["sample_mode"] == "random"


def test_eval_records_buckets_tmm_batches_by_prediction_layer_count(monkeypatch: pytest.MonkeyPatch) -> None:
    from our_work.eval.pipeline import _evaluate_records

    calls: list[list[int]] = []

    def fake_simulate(groups, **kwargs):
        calls.append([len(group) for group in groups])
        reflections = [[0.1] * 4 for _ in groups]
        transmissions = [[0.9] * 4 for _ in groups]
        return [2.0, 6.0, 10.0, 15.0], reflections, transmissions, [True] * len(groups)

    monkeypatch.setattr("our_work.eval.pipeline.simulate_structure_batch", fake_simulate)

    records = [
        {"sample_id": "a", "layer_count": 1, "structure_tokens": ["Ge_10"], "spectrum_rt": [0.1] * 4 + [0.9] * 4},
        {"sample_id": "b", "layer_count": 2, "structure_tokens": ["Ge_10", "Ge_10"], "spectrum_rt": [0.1] * 4 + [0.9] * 4},
        {"sample_id": "c", "layer_count": 1, "structure_tokens": ["Ge_10"], "spectrum_rt": [0.1] * 4 + [0.9] * 4},
    ]
    predicted_groups = [["Ge_10"], ["Ge_10", "Ge_10"], ["Ge_10"]]

    rows = _evaluate_records(
        split_name="train",
        records=records,
        predicted_groups=predicted_groups,
        database_path="database",
        wavelength_range_um=(2.0, 15.0),
        num_points=4,
        incident_angle=0.0,
        polarization=0,
        tolerance=0.001,
        complex_dtype="complex128",
        tmm_batch_size=4,
        tmm_device="cpu",
    )

    assert calls == [[1, 1], [2]]
    assert all(row["generated_valid"] for row in rows)


def test_select_split_shard_paths_supports_head_and_random_subset(tmp_path: Path) -> None:
    dataset_dir = _write_multi_shard_dataset(tmp_path)

    head_paths = select_split_shard_paths(dataset_dir, "train", sample_mode="head_shards", max_shards=2, seed=42)
    assert [path.name for path in head_paths] == ["shard-train-00.parquet", "shard-train-01.parquet"]

    random_paths = select_split_shard_paths(dataset_dir, "train", sample_mode="shard_subset_random", max_shards=2, seed=7)
    assert len(random_paths) == 2
    assert len({path.name for path in random_paths}) == 2
