from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from our_work.data_gen.analysis import analyze_dataset
from our_work.data_gen.scripts.convert_legacy_npy_dataset import convert_legacy_npy_dataset, main


def _fresh_tmp_dir(name: str) -> Path:
    path = Path("tests/.tmp") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _write_legacy_arrays(base: Path) -> dict[str, Path]:
    paths = {
        "spectrum_train": base / "Spectrum_train.npy",
        "structure_train": base / "Structure_train.npy",
        "spectrum_test": base / "Spectrum_test.npy",
        "structure_test": base / "Structure_test.npy",
    }
    np.save(paths["spectrum_train"], np.asarray([[0.1, 0.2, 0.7, 0.8], [0.2, 0.3, 0.6, 0.7], [0.3, 0.4, 0.5, 0.6]], dtype=np.float32))
    np.save(paths["spectrum_test"], np.asarray([[0.4, 0.5, 0.4, 0.5], [0.5, 0.6, 0.3, 0.4]], dtype=np.float32))
    np.save(
        paths["structure_train"],
        np.asarray(
            [
                ["Ge_10", "SiO2_20"],
                ["Si_30", "Ge_10"],
                ["ZnS_40", "Si_30", "Ge_10"],
            ],
            dtype=object,
        ),
    )
    np.save(
        paths["structure_test"],
        np.asarray(
            [
                ["Ge_10", "ZnS_40"],
                ["SiO2_20", "Si_30"],
            ],
            dtype=object,
        ),
    )
    return paths


def test_convert_legacy_npy_dataset_writes_analysis_compatible_parquet() -> None:
    tmp_dir = _fresh_tmp_dir("legacy-convert-analysis")
    try:
        paths = _write_legacy_arrays(tmp_dir)
        output_dir = tmp_dir / "converted"

        summary = convert_legacy_npy_dataset(
            spectrum_train=paths["spectrum_train"],
            structure_train=paths["structure_train"],
            spectrum_test=paths["spectrum_test"],
            structure_test=paths["structure_test"],
            output_dir=output_dir,
            records_per_shard=2,
        )

        assert summary["split_counts"] == {"train": 3, "val": 0, "test": 2}
        manifest = json.loads((output_dir / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
        assert manifest["train"] == ["train-shard-00000.parquet", "train-shard-00001.parquet"]
        assert manifest["test"] == ["test-shard-00000.parquet"]

        frame = pd.read_parquet(output_dir / "shards" / "train-shard-00000.parquet")
        assert list(frame.columns) == [
            "sample_id",
            "layer_count",
            "structure_tokens",
            "token_ids",
            "materials",
            "thickness_nm",
            "spectrum_rt",
        ]
        assert frame.iloc[0]["sample_id"] == "train-000000000"
        assert frame.iloc[0]["materials"].tolist() == ["Ge", "SiO2"]
        assert frame.iloc[0]["thickness_nm"].tolist() == [10, 20]
        assert np.allclose(frame.iloc[0]["spectrum_rt"].tolist(), [0.1, 0.2, 0.7, 0.8])

        vocab = json.loads((output_dir / "vocab" / "vocab.json").read_text(encoding="utf-8"))
        assert vocab["tokens"][:4] == ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        assert "Ge_10" in vocab["tokens"]

        summaries = analyze_dataset(
            dataset_dir=output_dir,
            scopes=["train", "test"],
            batch_size=2,
            wavelength_min=0.4,
            wavelength_max=1.1,
            enable_spectrum_analysis=False,
        )
        assert summaries["train"]["structure"]["sample_count"] == 3
        assert summaries["test"]["structure"]["sample_count"] == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_convert_legacy_npy_dataset_main_supports_max_samples() -> None:
    tmp_dir = _fresh_tmp_dir("legacy-convert-main")
    try:
        paths = _write_legacy_arrays(tmp_dir)
        output_dir = tmp_dir / "converted-main"

        main(
            [
                "--spectrum-train",
                str(paths["spectrum_train"]),
                "--structure-train",
                str(paths["structure_train"]),
                "--spectrum-test",
                str(paths["spectrum_test"]),
                "--structure-test",
                str(paths["structure_test"]),
                "--output-dir",
                str(output_dir),
                "--records-per-shard",
                "10",
                "--max-train-samples",
                "1",
                "--max-test-samples",
                "1",
                "--num-workers",
                "1",
            ]
        )

        summary = json.loads((output_dir / "stats" / "summary.json").read_text(encoding="utf-8"))
        assert summary["num_workers"] == 1
        assert summary["split_counts"] == {"train": 1, "val": 0, "test": 1}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_convert_legacy_npy_dataset_parallel_workers_write_shards() -> None:
    tmp_dir = _fresh_tmp_dir("legacy-convert-parallel")
    try:
        paths = _write_legacy_arrays(tmp_dir)
        output_dir = tmp_dir / "converted-parallel"

        summary = convert_legacy_npy_dataset(
            spectrum_train=paths["spectrum_train"],
            structure_train=paths["structure_train"],
            spectrum_test=paths["spectrum_test"],
            structure_test=paths["structure_test"],
            output_dir=output_dir,
            records_per_shard=1,
            num_workers=2,
        )

        assert summary["num_workers"] == 2
        assert summary["split_counts"] == {"train": 3, "val": 0, "test": 2}
        assert summary["split_manifest"]["train"] == [
            "train-shard-00000.parquet",
            "train-shard-00001.parquet",
            "train-shard-00002.parquet",
        ]

        frame = pd.read_parquet(output_dir / "shards" / "train-shard-00001.parquet")
        assert frame.iloc[0]["sample_id"] == "train-000000001"
        assert frame.iloc[0]["structure_tokens"].tolist() == ["Si_30", "Ge_10"]
        assert frame.iloc[0]["token_ids"].tolist() == [6, 4]

        vocab = json.loads((output_dir / "vocab" / "vocab.json").read_text(encoding="utf-8"))
        assert vocab["tokens"] == ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20", "Si_30", "ZnS_40"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
