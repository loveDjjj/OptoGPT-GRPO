from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow.parquet as pq


def load_vocab_tokens(dataset_dir: str | Path) -> list[str]:
    dataset_dir = Path(dataset_dir)
    vocab_path = dataset_dir / "vocab" / "vocab.json"
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tokens" in payload:
        return [str(token) for token in payload["tokens"]]
    if isinstance(payload, list):
        return [str(token) for token in payload]
    raise ValueError(f"Unsupported vocab format: {vocab_path}")


def derive_materials_and_thicknesses(tokens: Sequence[str]) -> tuple[list[str], list[int]]:
    material_names: set[str] = set()
    thickness_values_nm: set[int] = set()
    for token in tokens:
        parts = str(token).rsplit("_", 1)
        if len(parts) != 2:
            continue
        material_names.add(parts[0])
        try:
            thickness_values_nm.add(int(parts[1]))
        except ValueError:
            continue
    return sorted(material_names), sorted(thickness_values_nm)


def resolve_analysis_scopes(dataset_dir: str | Path, scopes: Sequence[str]) -> dict[str, list[Path]]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "splits" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    resolved: dict[str, list[Path]] = {}
    requested_scopes = [str(scope) for scope in scopes]
    for scope in requested_scopes:
        if scope == "all":
            shard_names: list[str] = []
            for split_name in ("train", "val", "test"):
                shard_names.extend(manifest.get(split_name, []))
            resolved["all"] = [dataset_dir / "shards" / shard_name for shard_name in sorted(set(shard_names))]
        else:
            resolved[scope] = [dataset_dir / "shards" / shard_name for shard_name in manifest.get(scope, [])]
    return resolved


def resolve_custom_scope(shard_paths: Sequence[str | Path]) -> dict[str, list[Path]]:
    return {"custom": [Path(path) for path in shard_paths]}


def iter_record_batches(
    *,
    shard_paths: Sequence[str | Path],
    batch_size: int,
    columns: Sequence[str] | None = None,
) -> Iterator[list[dict]]:
    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(Path(shard_path))
        for record_batch in parquet_file.iter_batches(batch_size=int(batch_size), columns=list(columns) if columns else None):
            yield record_batch.to_pylist()
