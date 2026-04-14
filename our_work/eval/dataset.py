from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


def resolve_split_shard_paths(dataset_dir: str | Path, split_name: str) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "splits" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [dataset_dir / "shards" / shard_name for shard_name in manifest.get(split_name, [])]


def iter_split_records(dataset_dir: str | Path, split_name: str) -> Iterator[dict]:
    columns = ["sample_id", "layer_count", "structure_tokens", "spectrum_rt"]
    for shard_path in resolve_split_shard_paths(dataset_dir, split_name):
        parquet_file = pq.ParquetFile(shard_path)
        for record_batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
            payload = record_batch.to_pydict()
            for index in range(len(payload["sample_id"])):
                yield {
                    "sample_id": str(payload["sample_id"][index]),
                    "layer_count": int(payload["layer_count"][index]),
                    "structure_tokens": list(payload["structure_tokens"][index]),
                    "spectrum_rt": list(payload["spectrum_rt"][index]),
                }


def sample_split_records(dataset_dir: str | Path, split_name: str, *, max_samples: int, seed: int) -> tuple[list[dict], int]:
    rng = random.Random(int(seed))
    target_count = max(0, int(max_samples))
    sampled: list[dict] = []
    total_count = 0
    for row in iter_split_records(dataset_dir, split_name):
        total_count += 1
        if target_count == 0:
            continue
        if len(sampled) < target_count:
            sampled.append(row)
            continue
        replacement_index = rng.randrange(total_count)
        if replacement_index < target_count:
            sampled[replacement_index] = row
    return sampled, total_count
