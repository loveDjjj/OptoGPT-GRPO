from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq

try:
    import cudf
except Exception:  # pragma: no cover - optional dependency
    cudf = None

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


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


def count_total_rows(shard_paths: Sequence[str | Path]) -> int:
    total = 0
    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(Path(shard_path))
        total += int(parquet_file.metadata.num_rows)
    return total


def iter_structure_batches(
    *,
    shard_paths: Sequence[str | Path],
    batch_size: int,
) -> Iterator[dict[str, object]]:
    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(Path(shard_path))
        for record_batch in parquet_file.iter_batches(
            batch_size=int(batch_size),
            columns=["sample_id", "layer_count", "materials", "thickness_nm"],
        ):
            table = record_batch.to_pydict()
            yield {
                "sample_id": list(table["sample_id"]),
                "layer_count": [int(value) for value in table["layer_count"]],
                "materials": [list(value) for value in table["materials"]],
                "thickness_nm": [[int(item) for item in values] for values in table["thickness_nm"]],
            }


def iter_spectrum_frames(
    *,
    shard_paths: Sequence[str | Path],
) -> Iterator["cudf.DataFrame"]:
    if cudf is None:
        raise RuntimeError("RAPIDS cudf is not installed")
    for shard_path in shard_paths:
        yield cudf.read_parquet(Path(shard_path), columns=["sample_id", "layer_count", "materials", "spectrum_rt"])


def extract_spectrum_matrix(frame: "cudf.DataFrame") -> "cp.ndarray":
    if cudf is None or cp is None:
        raise RuntimeError("RAPIDS cudf/cupy is not installed")

    spectrum_series = frame["spectrum_rt"].list.astype("float32")
    lengths = spectrum_series.list.len().to_cupy()
    row_count = int(len(frame))
    if row_count == 0:
        return cp.empty((0, 0), dtype=cp.float32)

    row_length = int(lengths[0].item())
    if not bool(cp.all(lengths == row_length)):
        raise ValueError("spectrum_rt rows must all have the same length")

    leaves = spectrum_series.list.leaves.to_cupy(dtype=np.float32)
    return leaves.reshape(row_count, row_length)
