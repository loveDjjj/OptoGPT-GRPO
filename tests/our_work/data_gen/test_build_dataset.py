import json
from pathlib import Path

import pandas as pd

from our_work.data_gen.pipeline.build_dataset import build_small_dataset


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
