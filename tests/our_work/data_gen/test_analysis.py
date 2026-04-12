from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from our_work.data_gen.analysis import analyze_dataset
from our_work.data_gen.scripts.run_analyze_dataset import main as analyze_main


def _write_tiny_dataset(base: Path) -> Path:
    dataset_dir = base / "dataset"
    (dataset_dir / "shards").mkdir(parents=True)
    (dataset_dir / "splits").mkdir(parents=True)
    (dataset_dir / "vocab").mkdir(parents=True)

    records = [
        {
            "sample_id": "sample-000",
            "layer_count": 2,
            "structure_tokens": ["Ge_10", "SiO2_20"],
            "token_ids": [1, 4, 5, 2],
            "materials": ["Ge", "SiO2"],
            "thickness_nm": [10, 20],
            "spectrum_rt": [0.1] * 8 + [0.9] * 8,
        },
        {
            "sample_id": "sample-001",
            "layer_count": 2,
            "structure_tokens": ["SiO2_20", "Ge_30"],
            "token_ids": [1, 5, 6, 2],
            "materials": ["SiO2", "Ge"],
            "thickness_nm": [20, 30],
            "spectrum_rt": [0.2] * 8 + [0.8] * 8,
        },
        {
            "sample_id": "sample-002",
            "layer_count": 3,
            "structure_tokens": ["Ge_10", "Ge_30", "SiO2_20"],
            "token_ids": [1, 4, 6, 5, 2],
            "materials": ["Ge", "Ge", "SiO2"],
            "thickness_nm": [10, 30, 20],
            "spectrum_rt": [0.3] * 8 + [0.7] * 8,
        },
    ]
    pd.DataFrame.from_records(records[:2]).to_parquet(dataset_dir / "shards" / "shard-00000.parquet", index=False)
    pd.DataFrame.from_records(records[2:]).to_parquet(dataset_dir / "shards" / "shard-00001.parquet", index=False)
    (dataset_dir / "splits" / "split_manifest.json").write_text(
        json.dumps(
            {
                "train": ["shard-00000.parquet"],
                "val": ["shard-00001.parquet"],
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "vocab" / "vocab.json").write_text(
        json.dumps({"tokens": ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20", "Ge_30"]}),
        encoding="utf-8",
    )
    return dataset_dir


def test_analyze_dataset_writes_structure_and_spectrum_artifacts(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_dataset(tmp_path)

    summaries = analyze_dataset(
        dataset_dir=dataset_dir,
        scopes=["all", "train", "val", "test"],
        batch_size=2,
        wavelength_min=2.0,
        wavelength_max=15.0,
        pca_components=4,
        cluster_count=2,
        cluster_fit_samples=8,
        cluster_iterations=5,
        scatter_max_points=8,
        device="cpu",
    )

    assert set(summaries.keys()) == {"all", "train", "val", "test"}
    assert (dataset_dir / "analysis" / "all" / "structure_material_by_layer.png").exists()
    assert (dataset_dir / "analysis" / "all" / "spectrum_mean_std.png").exists()
    assert (dataset_dir / "analysis" / "all" / "spectrum_cluster_sizes.png").exists()
    manifest = json.loads((dataset_dir / "analysis" / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scopes"] == ["all", "train", "val", "test"]
    assert summaries["test"]["structure"]["skipped_reason"] == "no records"
    assert summaries["test"]["spectrum"]["skipped_reason"] == "no records"


def test_run_analyze_dataset_main_supports_dataset_dir(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_dataset(tmp_path)
    output_dir = tmp_path / "analysis-output"

    analyze_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "all",
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--wavelength-min",
            "2.0",
            "--wavelength-max",
            "15.0",
            "--pca-components",
            "4",
            "--cluster-count",
            "2",
            "--cluster-fit-samples",
            "8",
            "--cluster-iterations",
            "5",
            "--scatter-max-points",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert (output_dir / "all" / "analysis_summary.json").exists()
    assert (output_dir / "all" / "structure_thickness_global.png").exists()
