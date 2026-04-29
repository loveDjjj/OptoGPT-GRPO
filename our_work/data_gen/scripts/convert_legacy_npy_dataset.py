from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work._shared.io.config import resolve_repo_path
from our_work.data_gen.pipeline.shard_writer import write_records_to_parquet, write_split_manifest

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def parse_structure_tokens(structure_tokens: Any) -> tuple[list[str], list[int], list[str]]:
    tokens = [str(token) for token in list(structure_tokens)]
    materials: list[str] = []
    thickness_nm: list[int] = []
    for token in tokens:
        parts = token.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid structure token: {token!r}")
        material, thickness = parts
        materials.append(material)
        thickness_nm.append(int(thickness))
    return tokens, materials, thickness_nm


def _load_spectrum_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r", allow_pickle=False)


def _load_structure_array(path: str | Path) -> np.ndarray:
    # Object arrays cannot be memory-mapped by NumPy. This intentionally loads
    # one split at a time so train/test are not resident simultaneously.
    return np.load(Path(path), allow_pickle=True)


def _iter_indices(total_count: int, max_samples: int | None) -> range:
    resolved_count = total_count if max_samples is None else min(total_count, int(max_samples))
    return range(resolved_count)


def _progress(iterable, *, total: int, desc: str):
    if tqdm is None:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, unit="sample", dynamic_ncols=True)


def _token_id(token: str, token_to_id: dict[str, int]) -> int:
    if token not in token_to_id:
        token_to_id[token] = len(token_to_id)
    return int(token_to_id[token])


def _serialize_record(
    *,
    split_name: str,
    index: int,
    spectrum: np.ndarray,
    structure_tokens: Any,
    token_to_id: dict[str, int],
) -> dict[str, Any]:
    tokens, materials, thickness_nm = parse_structure_tokens(structure_tokens)
    return {
        "sample_id": f"{split_name}-{index:09d}",
        "layer_count": len(tokens),
        "structure_tokens": tokens,
        "token_ids": [_token_id(token, token_to_id) for token in tokens],
        "materials": materials,
        "thickness_nm": thickness_nm,
        "spectrum_rt": np.asarray(spectrum, dtype=np.float32).reshape(-1).tolist(),
    }


def _write_split(
    *,
    split_name: str,
    spectrum_path: str | Path,
    structure_path: str | Path,
    output_dir: Path,
    token_to_id: dict[str, int],
    records_per_shard: int,
    max_samples: int | None,
) -> tuple[list[str], int]:
    spectra = _load_spectrum_array(spectrum_path)
    structures = _load_structure_array(structure_path)
    if len(spectra) != len(structures):
        raise ValueError(f"{split_name} spectrum/structure sample count mismatch: {len(spectra)} != {len(structures)}")

    shard_names: list[str] = []
    records: list[dict[str, Any]] = []
    shard_index = 0
    written_count = 0
    indices = _iter_indices(len(spectra), max_samples)
    total_count = len(indices)

    for index in _progress(indices, total=total_count, desc=f"convert {split_name}"):
        records.append(
            _serialize_record(
                split_name=split_name,
                index=int(index),
                spectrum=spectra[index],
                structure_tokens=structures[index],
                token_to_id=token_to_id,
            )
        )
        if len(records) >= int(records_per_shard):
            shard_name = f"{split_name}-shard-{shard_index:05d}.parquet"
            write_records_to_parquet(output_dir / "shards" / shard_name, records)
            shard_names.append(shard_name)
            written_count += len(records)
            records = []
            shard_index += 1

    if records:
        shard_name = f"{split_name}-shard-{shard_index:05d}.parquet"
        write_records_to_parquet(output_dir / "shards" / shard_name, records)
        shard_names.append(shard_name)
        written_count += len(records)

    del spectra
    del structures
    gc.collect()
    return shard_names, written_count


def convert_legacy_npy_dataset(
    *,
    spectrum_train: str | Path | None = None,
    structure_train: str | Path | None = None,
    spectrum_test: str | Path | None = None,
    structure_test: str | Path | None = None,
    output_dir: str | Path,
    records_per_shard: int = 50000,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    if int(records_per_shard) <= 0:
        raise ValueError("records_per_shard must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    if spectrum_train and structure_train:
        manifest["train"], split_counts["train"] = _write_split(
            split_name="train",
            spectrum_path=spectrum_train,
            structure_path=structure_train,
            output_dir=output_path,
            token_to_id=token_to_id,
            records_per_shard=int(records_per_shard),
            max_samples=max_train_samples,
        )
    elif spectrum_train or structure_train:
        raise ValueError("spectrum_train and structure_train must be provided together")

    if spectrum_test and structure_test:
        manifest["test"], split_counts["test"] = _write_split(
            split_name="test",
            spectrum_path=spectrum_test,
            structure_path=structure_test,
            output_dir=output_path,
            token_to_id=token_to_id,
            records_per_shard=int(records_per_shard),
            max_samples=max_test_samples,
        )
    elif spectrum_test or structure_test:
        raise ValueError("spectrum_test and structure_test must be provided together")

    if not manifest["train"] and not manifest["test"]:
        raise ValueError("At least one complete train or test split must be provided")

    write_split_manifest(output_path / "splits" / "split_manifest.json", manifest)
    write_split_manifest(output_path / "vocab" / "vocab.json", {"tokens": list(token_to_id.keys())})
    summary = {
        "format": "legacy_npy_converted",
        "records_per_shard": int(records_per_shard),
        "split_counts": split_counts,
        "split_manifest": manifest,
        "vocab_size": len(token_to_id),
        "notes": {
            "spectrum_rt": "Copied from legacy Spectrum_*.npy rows. For the original dataset this is usually [R(71), T(71)].",
            "structure": "Parsed from legacy Structure_*.npy object-array tokens such as Material_ThicknessNm.",
        },
    }
    summary_path = output_path / "stats" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert legacy Spectrum_*.npy and Structure_*.npy files to data_gen parquet format.")
    parser.add_argument("--spectrum-train", default=None)
    parser.add_argument("--structure-train", default=None)
    parser.add_argument("--spectrum-test", default=None)
    parser.add_argument("--structure-test", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records-per-shard", type=int, default=50000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args(argv)

    convert_legacy_npy_dataset(
        spectrum_train=resolve_repo_path(args.spectrum_train, project_root=PROJECT_ROOT) if args.spectrum_train else None,
        structure_train=resolve_repo_path(args.structure_train, project_root=PROJECT_ROOT) if args.structure_train else None,
        spectrum_test=resolve_repo_path(args.spectrum_test, project_root=PROJECT_ROOT) if args.spectrum_test else None,
        structure_test=resolve_repo_path(args.structure_test, project_root=PROJECT_ROOT) if args.structure_test else None,
        output_dir=resolve_repo_path(args.output_dir, project_root=PROJECT_ROOT),
        records_per_shard=args.records_per_shard,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()
