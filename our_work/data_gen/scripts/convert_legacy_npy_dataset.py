from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
_WORKER_SPECTRA: np.ndarray | None = None
_WORKER_STRUCTURES: np.ndarray | None = None
_WORKER_TOKEN_TO_ID: dict[str, int] | None = None

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


def _progress(iterable, *, total: int, desc: str, unit: str = "sample"):
    if tqdm is None:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _token_id(token: str, token_to_id: dict[str, int], *, allow_new_tokens: bool = True) -> int:
    if token not in token_to_id:
        if not allow_new_tokens:
            raise KeyError(f"Token {token!r} was not found in the prebuilt vocabulary")
        token_to_id[token] = len(token_to_id)
    return int(token_to_id[token])


def _scan_structure_tokens(
    *,
    split_name: str,
    structure_path: str | Path,
    token_to_id: dict[str, int],
    max_samples: int | None,
) -> int:
    structures = _load_structure_array(structure_path)
    structure_count = len(structures)
    indices = _iter_indices(len(structures), max_samples)
    for index in _progress(indices, total=len(indices), desc=f"scan vocab {split_name}"):
        tokens, _, _ = parse_structure_tokens(structures[index])
        for token in tokens:
            _token_id(token, token_to_id)
    del structures
    gc.collect()
    return structure_count


def _serialize_record(
    *,
    split_name: str,
    index: int,
    spectrum: np.ndarray,
    structure_tokens: Any,
    token_to_id: dict[str, int],
    allow_new_tokens: bool = True,
) -> dict[str, Any]:
    tokens, materials, thickness_nm = parse_structure_tokens(structure_tokens)
    return {
        "sample_id": f"{split_name}-{index:09d}",
        "layer_count": len(tokens),
        "structure_tokens": tokens,
        "token_ids": [_token_id(token, token_to_id, allow_new_tokens=allow_new_tokens) for token in tokens],
        "materials": materials,
        "thickness_nm": thickness_nm,
        "spectrum_rt": np.asarray(spectrum, dtype=np.float32).reshape(-1).tolist(),
    }


def _init_split_worker(spectrum_path: str, structure_path: str, token_to_id: dict[str, int]) -> None:
    global _WORKER_SPECTRA, _WORKER_STRUCTURES, _WORKER_TOKEN_TO_ID
    _WORKER_SPECTRA = _load_spectrum_array(spectrum_path)
    _WORKER_STRUCTURES = _load_structure_array(structure_path)
    _WORKER_TOKEN_TO_ID = dict(token_to_id)


def _write_chunk_worker(task: dict[str, Any]) -> tuple[int, str, int]:
    if _WORKER_SPECTRA is None or _WORKER_STRUCTURES is None or _WORKER_TOKEN_TO_ID is None:
        raise RuntimeError("legacy npy converter worker was not initialized")

    split_name = str(task["split_name"])
    shard_index = int(task["shard_index"])
    start_index = int(task["start_index"])
    end_index = int(task["end_index"])
    output_dir = Path(task["output_dir"])
    shard_name = f"{split_name}-shard-{shard_index:05d}.parquet"
    records = [
        _serialize_record(
            split_name=split_name,
            index=index,
            spectrum=_WORKER_SPECTRA[index],
            structure_tokens=_WORKER_STRUCTURES[index],
            token_to_id=_WORKER_TOKEN_TO_ID,
            allow_new_tokens=False,
        )
        for index in range(start_index, end_index)
    ]
    write_records_to_parquet(output_dir / "shards" / shard_name, records)
    return shard_index, shard_name, len(records)


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


def _write_split_parallel(
    *,
    split_name: str,
    spectrum_path: str | Path,
    structure_path: str | Path,
    output_dir: Path,
    token_to_id: dict[str, int],
    records_per_shard: int,
    max_samples: int | None,
    num_workers: int,
    structure_count: int,
) -> tuple[list[str], int]:
    spectra = _load_spectrum_array(spectrum_path)
    if len(spectra) != int(structure_count):
        raise ValueError(f"{split_name} spectrum/structure sample count mismatch: {len(spectra)} != {int(structure_count)}")
    total_count = len(_iter_indices(len(spectra), max_samples))
    del spectra
    gc.collect()

    tasks: list[dict[str, Any]] = []
    for shard_index, start_index in enumerate(range(0, total_count, int(records_per_shard))):
        end_index = min(start_index + int(records_per_shard), total_count)
        tasks.append(
            {
                "split_name": split_name,
                "shard_index": shard_index,
                "start_index": start_index,
                "end_index": end_index,
                "output_dir": str(output_dir),
            }
        )
    if not tasks:
        return [], 0

    results: list[tuple[int, str, int]] = []
    with ProcessPoolExecutor(
        max_workers=int(num_workers),
        initializer=_init_split_worker,
        initargs=(str(spectrum_path), str(structure_path), token_to_id),
    ) as executor:
        futures = [executor.submit(_write_chunk_worker, task) for task in tasks]
        for future in _progress(as_completed(futures), total=len(futures), desc=f"convert {split_name}", unit="shard"):
            results.append(future.result())

    results.sort(key=lambda item: item[0])
    return [shard_name for _, shard_name, _ in results], sum(count for _, _, count in results)


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
    num_workers: int = 1,
) -> dict[str, Any]:
    if int(records_per_shard) <= 0:
        raise ValueError("records_per_shard must be positive")
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive")
    resolved_num_workers = int(num_workers)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_specs: list[tuple[str, str | Path, str | Path, int | None]] = []
    structure_counts: dict[str, int] = {}

    if spectrum_train and structure_train:
        split_specs.append(("train", spectrum_train, structure_train, max_train_samples))
    elif spectrum_train or structure_train:
        raise ValueError("spectrum_train and structure_train must be provided together")

    if spectrum_test and structure_test:
        split_specs.append(("test", spectrum_test, structure_test, max_test_samples))
    elif spectrum_test or structure_test:
        raise ValueError("spectrum_test and structure_test must be provided together")

    if not split_specs:
        raise ValueError("At least one complete train or test split must be provided")

    if resolved_num_workers > 1:
        for split_name, _, structure_path, max_samples in split_specs:
            structure_counts[split_name] = _scan_structure_tokens(
                split_name=split_name,
                structure_path=structure_path,
                token_to_id=token_to_id,
                max_samples=max_samples,
            )

    for split_name, spectrum_path, structure_path, max_samples in split_specs:
        if resolved_num_workers > 1:
            manifest[split_name], split_counts[split_name] = _write_split_parallel(
                split_name=split_name,
                spectrum_path=spectrum_path,
                structure_path=structure_path,
                output_dir=output_path,
                token_to_id=token_to_id,
                records_per_shard=int(records_per_shard),
                max_samples=max_samples,
                num_workers=resolved_num_workers,
                structure_count=structure_counts[split_name],
            )
        else:
            manifest[split_name], split_counts[split_name] = _write_split(
                split_name=split_name,
                spectrum_path=spectrum_path,
                structure_path=structure_path,
                output_dir=output_path,
                token_to_id=token_to_id,
                records_per_shard=int(records_per_shard),
                max_samples=max_samples,
            )

    if not manifest["train"] and not manifest["test"]:
        raise ValueError("At least one complete train or test split must be provided")

    write_split_manifest(output_path / "splits" / "split_manifest.json", manifest)
    write_split_manifest(output_path / "vocab" / "vocab.json", {"tokens": list(token_to_id.keys())})
    summary = {
        "format": "legacy_npy_converted",
        "records_per_shard": int(records_per_shard),
        "num_workers": resolved_num_workers,
        "split_counts": split_counts,
        "split_manifest": manifest,
        "vocab_size": len(token_to_id),
        "notes": {
            "spectrum_rt": "Copied from legacy Spectrum_*.npy rows. For the original dataset this is usually [R(71), T(71)].",
            "structure": "Parsed from legacy Structure_*.npy object-array tokens such as Material_ThicknessNm.",
            "parallelism": "When num_workers > 1, each worker process loads the current split Structure_*.npy object array, so host memory use scales with num_workers.",
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
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker processes for shard conversion. Default: 1.")
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
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
