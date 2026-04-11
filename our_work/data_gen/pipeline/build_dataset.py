from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from our_work.data_gen.pipeline.shard_writer import write_records_to_parquet, write_split_manifest
from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from our_work.data_gen.pipeline.token_vocab import build_token_vocab
from our_work.data_gen.pipeline.sampler import sample_unique_bucket


def _serializable_record(
    *,
    sample_id: str,
    layer_count: int,
    structure_tokens: list[str],
    token_to_id: dict[str, int],
    spectrum_rt,
) -> dict[str, Any]:
    materials: list[str] = []
    thickness_nm: list[int] = []
    for token in structure_tokens:
        material, thickness = token.rsplit("_", 1)
        materials.append(material)
        thickness_nm.append(int(thickness))
    return {
        "sample_id": sample_id,
        "layer_count": layer_count,
        "structure_tokens": list(structure_tokens),
        "token_ids": [int(token_to_id[token]) for token in structure_tokens],
        "materials": materials,
        "thickness_nm": thickness_nm,
        "spectrum_rt": list(spectrum_rt),
    }


def _split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    train_records = shuffled[:train_count]
    val_records = shuffled[train_count : train_count + val_count]
    test_records = shuffled[train_count + val_count :]
    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def _write_split_shards(
    output_dir: str | Path,
    split_records: dict[str, list[dict[str, Any]]],
    *,
    records_per_shard: int,
) -> dict[str, list[str]]:
    base_dir = Path(output_dir)
    shard_dir = base_dir / "shards"
    split_manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    shard_index = 0
    for split_name in ("train", "val", "test"):
        records = split_records[split_name]
        for start in range(0, len(records), records_per_shard):
            chunk = records[start : start + records_per_shard]
            if not chunk:
                continue
            shard_name = f"shard-{shard_index:05d}.parquet"
            write_records_to_parquet(shard_dir / shard_name, chunk)
            split_manifest[split_name].append(shard_name)
            shard_index += 1
    write_split_manifest(base_dir / "splits" / "split_manifest.json", split_manifest)
    return split_manifest


def build_small_dataset(
    *,
    output_dir: str | Path,
    database_path: str,
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_counts: list[int],
    samples_per_bucket: int,
    num_points: int,
    wavelength_range_um: tuple[float, float],
    incident_angle: float = 0.0,
    polarization: int = 0,
    tolerance: float = 1e-3,
    complex_dtype: str = "complex128",
    records_per_shard: int = 50000,
    train_ratio: float = 1.0,
    val_ratio: float = 0.0,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, list[str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = build_token_vocab(material_names, thickness_values_nm)
    all_records: list[dict[str, Any]] = []
    layer_iterable = tqdm(
        layer_counts,
        desc="data_gen buckets",
        unit="bucket",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    for layer_count in layer_iterable:
        structure_token_groups = sample_unique_bucket(
            material_names=material_names,
            thickness_values_nm=thickness_values_nm,
            layer_count=layer_count,
            target_count=samples_per_bucket,
            rng_seed=seed + layer_count,
        )
        _, reflections, transmissions, ok_mask = simulate_structure_batch(
            structure_token_groups,
            database_path=database_path,
            wavelength_range_um=wavelength_range_um,
            num_points=num_points,
            incident_angle=incident_angle,
            polarization=polarization,
            tolerance=tolerance,
            complex_dtype=complex_dtype,
        )
        for index, (tokens, reflection, transmission, ok) in enumerate(
            zip(structure_token_groups, reflections, transmissions, ok_mask)
        ):
            if not bool(ok):
                continue
            spectrum_rt = flatten_rt_spectrum(reflection, transmission).astype("float32").tolist()
            all_records.append(
                _serializable_record(
                    sample_id=f"{layer_count}-{index}",
                    layer_count=layer_count,
                    structure_tokens=tokens,
                    token_to_id=vocab.token_to_id,
                    spectrum_rt=spectrum_rt,
                )
            )
        if hasattr(layer_iterable, "set_postfix"):
            valid_count = int(sum(bool(item) for item in ok_mask))
            layer_iterable.set_postfix(
                {
                    "layer_count": int(layer_count),
                    "generated": int(len(structure_token_groups)),
                    "valid": valid_count,
                    "kept": int(len(all_records)),
                },
                refresh=False,
            )

    split_records = _split_records(
        all_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    split_manifest = _write_split_shards(
        output_path,
        split_records,
        records_per_shard=records_per_shard,
    )
    write_split_manifest(
        output_path / "vocab" / "vocab.json",
        {"tokens": vocab.special_tokens + [token for token in vocab.token_to_id.keys() if token not in vocab.special_tokens]},
    )
    return split_manifest
