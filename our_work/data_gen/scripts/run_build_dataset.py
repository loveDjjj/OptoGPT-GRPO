from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work._shared.io.config import load_yaml_config, resolve_repo_path
from our_work.data_gen.pipeline.build_dataset import build_small_dataset
from our_work.data_gen.pipeline.material_registry import build_material_registry
from our_work._shared.utils.seed import set_global_seed
from utils.dist import barrier, cleanup_distributed, init_distributed


def resolve_thickness_values_nm(data_config: dict) -> list[int]:
    explicit_values = data_config.get("thickness_values_nm")
    range_config = data_config.get("thickness_range_nm")

    if explicit_values is not None and range_config is not None:
        raise ValueError("data.thickness_values_nm and data.thickness_range_nm cannot be set at the same time")

    if explicit_values is not None:
        return [int(value) for value in explicit_values]

    if range_config is None:
        raise KeyError("data.thickness_range_nm is required when data.thickness_values_nm is not provided")

    min_value = int(range_config["min"])
    max_value = int(range_config["max"])
    step_value = int(range_config["step"])

    if step_value <= 0:
        raise ValueError("data.thickness_range_nm.step must be a positive integer")
    if min_value > max_value:
        raise ValueError("data.thickness_range_nm.min must be less than or equal to max")
    if (max_value - min_value) % step_value != 0:
        raise ValueError("data.thickness_range_nm max-min must be divisible by step")

    return list(range(min_value, max_value + step_value, step_value))


def resolve_data_gen_runtime_config(config: dict) -> dict:
    data_config = config["data"]
    sampling_config = config.get("sampling", {})
    tmm_config = config["tmm"]
    distributed_config = config.get("distributed", {})

    return {
        "thickness_values_nm": resolve_thickness_values_nm(data_config),
        "sampling_device": str(sampling_config.get("device", "auto")),
        "sampling_batch_size": int(sampling_config.get("batch_size", 65536)),
        "max_duplicate_retry": int(sampling_config.get("max_duplicate_retry", 1000)),
        "tmm_batch_size": int(tmm_config.get("batch_size", 2048)),
        "tmm_device": str(tmm_config.get("device", "auto")),
        "tmm_cpu_threads": int(tmm_config.get("cpu_threads", max(1, min(16, os.cpu_count() or 16)))),
        "distributed_enabled": bool(distributed_config.get("enabled", False)),
        "distributed_timeout_minutes": int(distributed_config.get("timeout_minutes", 30)),
        "distributed_shard_mode": str(distributed_config.get("shard_mode", "layer_bucket")),
    }


def _assign_layer_counts(layer_counts: list[int], *, rank: int, world_size: int, shard_mode: str) -> list[int]:
    if world_size <= 1:
        return list(layer_counts)
    resolved_mode = str(shard_mode).strip().lower()
    if resolved_mode != "layer_bucket":
        raise ValueError(f"unsupported distributed shard_mode: {shard_mode}")
    active_world_size = min(int(world_size), len(layer_counts))
    if rank >= active_world_size:
        return []
    return [layer_count for index, layer_count in enumerate(layer_counts) if index % active_world_size == rank]


def _resolve_rank_device(requested_device: str, *, local_rank: int, dist_enabled: bool) -> str:
    resolved = str(requested_device).strip().lower()
    if not dist_enabled:
        return requested_device
    if resolved in {"auto", "cuda"} or resolved.startswith("cuda:"):
        if torch.cuda.is_available():
            return f"cuda:{int(local_rank)}"
    return requested_device


def _resolve_torch_device_name(requested_device: str) -> str:
    resolved = str(requested_device).strip().lower()
    if resolved == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def merge_rank_split_manifests(output_dir: str | Path, *, world_size: int) -> dict[str, list[str]]:
    output_path = Path(output_dir)
    merged: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rank in range(int(world_size)):
        manifest_path = output_path / "splits" / f"split_manifest.rank{rank:02d}.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split_name in merged:
            merged[split_name].extend(payload.get(split_name, []))
    for split_name in merged:
        merged[split_name].sort()
    final_path = output_path / "splits" / "split_manifest.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small our_work spectral dataset.")
    parser.add_argument("--config", required=True, help="Path to the dataset YAML config.")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
    config = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)
    runtime = resolve_data_gen_runtime_config(config)
    dist_requested = runtime["distributed_enabled"] and int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if dist_requested else 0
    dist_device_name = _resolve_rank_device(runtime["tmm_device"], local_rank=local_rank, dist_enabled=dist_requested)
    dist_device = torch.device(_resolve_torch_device_name(dist_device_name))
    dist_ctx = init_distributed(device=dist_device, timeout_minutes=runtime["distributed_timeout_minutes"]) if dist_requested else None
    rank = 0 if dist_ctx is None else int(dist_ctx.rank)
    world_size = 1 if dist_ctx is None else int(dist_ctx.world_size)
    set_global_seed(int(config.get("seed", 42)), rank_offset=rank)
    torch.set_num_threads(int(runtime["tmm_cpu_threads"]))
    registry = build_material_registry(config["paths"]["database_dir"])
    assigned_layer_counts = _assign_layer_counts(
        [int(value) for value in config["data"]["layer_counts"]],
        rank=rank,
        world_size=world_size,
        shard_mode=runtime["distributed_shard_mode"],
    )
    build_small_dataset(
        output_dir=config["paths"]["output_dir"],
        database_path=config["paths"]["database_dir"],
        material_names=registry.material_names,
        thickness_values_nm=runtime["thickness_values_nm"],
        layer_counts=assigned_layer_counts,
        samples_per_bucket=int(config["data"]["samples_per_bucket"]),
        sampling_batch_size=runtime["sampling_batch_size"],
        tmm_batch_size=runtime["tmm_batch_size"],
        max_duplicate_retry=runtime["max_duplicate_retry"],
        sampling_device=_resolve_rank_device(runtime["sampling_device"], local_rank=local_rank, dist_enabled=dist_requested),
        tmm_device=dist_device_name,
        num_points=int(config["tmm"]["num_points"]),
        wavelength_range_um=tuple(config["tmm"]["wavelength_range_um"]),
        incident_angle=float(config["tmm"].get("incident_angle", 0.0)),
        polarization=int(config["tmm"].get("polarization", 0)),
        tolerance=float(config["tmm"].get("tolerance", 1e-3)),
        complex_dtype=str(config["tmm"].get("complex_dtype", "complex128")),
        records_per_shard=int(config["shards"]["records_per_shard"]),
        train_ratio=float(config["splits"]["train_ratio"]),
        val_ratio=float(config["splits"]["val_ratio"]),
        seed=int(config.get("seed", 42)),
        show_progress=bool(config.get("logging", {}).get("show_progress_bar", True)) and rank == 0,
        shard_prefix=f"rank{rank:02d}-" if world_size > 1 else "",
        split_manifest_name=f"split_manifest.rank{rank:02d}.json" if world_size > 1 else "split_manifest.json",
        write_vocab=rank == 0,
    )
    if dist_ctx is not None:
        barrier()
        if dist_ctx.is_main:
            merge_rank_split_manifests(config["paths"]["output_dir"], world_size=world_size)
        barrier()
        cleanup_distributed()


if __name__ == "__main__":
    main()
