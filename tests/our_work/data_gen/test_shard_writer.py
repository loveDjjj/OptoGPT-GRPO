import json
from pathlib import Path

from our_work.data_gen.pipeline.shard_writer import write_split_manifest


def test_write_split_manifest_creates_json(tmp_path: Path):
    write_split_manifest(
        tmp_path / "splits" / "split_manifest.json",
        {"train": ["shard-00000.parquet"], "val": ["shard-00001.parquet"], "test": []},
    )
    payload = json.loads((tmp_path / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    assert payload["train"] == ["shard-00000.parquet"]
