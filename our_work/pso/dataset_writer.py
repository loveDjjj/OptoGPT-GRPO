from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from our_work.data_gen.pipeline.shard_writer import write_records_to_parquet, write_split_manifest
from our_work.pso.search import AcceptedStructure


def _float_list(values) -> list[float]:
    return [round(float(value), 6) for value in values]


def serialize_accepted_structure(
    accepted: AcceptedStructure,
    *,
    sample_id: str,
    token_to_id: dict[str, int],
    acceptance_mse_threshold: float,
) -> dict[str, Any]:
    materials: list[str] = []
    thickness_nm: list[int] = []
    for token in accepted.structure_tokens:
        material, thickness = token.rsplit("_", 1)
        materials.append(material)
        thickness_nm.append(int(thickness))

    return {
        "sample_id": sample_id,
        "layer_count": len(accepted.structure_tokens),
        "structure_tokens": list(accepted.structure_tokens),
        "token_ids": [int(token_to_id[token]) for token in accepted.structure_tokens],
        "materials": materials,
        "thickness_nm": thickness_nm,
        "spectrum_rt": _float_list(accepted.reflection) + _float_list(accepted.transmission),
        "target_id": accepted.target_id,
        "target_family": accepted.target_family,
        "target_center_um": accepted.target_center_um,
        "target_fwhm_um": accepted.target_fwhm_um,
        "target_mse": float(accepted.target_mse),
        "acceptance_mse_threshold": float(acceptance_mse_threshold),
        "pso_seed": int(accepted.pso_seed),
        "pso_restart_index": int(accepted.pso_restart_index),
    }


def _split_records(records: list[dict[str, Any]], *, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * float(train_ratio))
    val_count = int(len(shuffled) * float(val_ratio))
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def _write_split_shards(output_dir: Path, split_records: dict[str, list[dict[str, Any]]], *, records_per_shard: int) -> dict[str, list[str]]:
    manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    shard_index = 0
    for split_name in ("train", "val", "test"):
        records = split_records[split_name]
        for start in range(0, len(records), int(records_per_shard)):
            chunk = records[start : start + int(records_per_shard)]
            if not chunk:
                continue
            shard_name = f"shard-{shard_index:05d}.parquet"
            write_records_to_parquet(output_dir / "shards" / shard_name, chunk)
            manifest[split_name].append(shard_name)
            shard_index += 1
    write_split_manifest(output_dir / "splits" / "split_manifest.json", manifest)
    return manifest


def write_pso_supplement_dataset(
    *,
    output_dir: str | Path,
    accepted: list[AcceptedStructure],
    token_to_id: dict[str, int],
    vocab_tokens: list[str],
    records_per_shard: int,
    acceptance_mse_threshold: float,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if int(records_per_shard) <= 0:
        raise ValueError("records_per_shard must be positive")

    records = [
        serialize_accepted_structure(
            item,
            sample_id=f"pso-{index}",
            token_to_id=token_to_id,
            acceptance_mse_threshold=acceptance_mse_threshold,
        )
        for index, item in enumerate(accepted)
    ]
    split_records = _split_records(records, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    manifest = _write_split_shards(output_path, split_records, records_per_shard=records_per_shard)
    write_split_manifest(output_path / "vocab" / "vocab.json", {"tokens": list(vocab_tokens)})

    target_ids = sorted({record["target_id"] for record in records})
    target_manifest = {
        target_id: {
            "record_count": sum(1 for record in records if record["target_id"] == target_id),
        }
        for target_id in target_ids
    }
    write_split_manifest(output_path / "targets" / "target_manifest.json", target_manifest)
    summary = {
        "accepted_count": len(records),
        "target_count": len(target_ids),
        "records_per_shard": int(records_per_shard),
        "split_counts": {name: len(values) for name, values in split_records.items()},
        "split_manifest": manifest,
    }
    summary_path = output_path / "stats" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
