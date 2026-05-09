from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_parquet_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        frame = pd.read_parquet(path)
        records.extend(frame.to_dict(orient="records"))
    return records


def load_split_records(output_dir: str | Path, split_name: str) -> list[dict]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "splits" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_paths = [output_dir / "shards" / shard_name for shard_name in manifest[split_name]]
    return load_parquet_records([str(path) for path in shard_paths])
