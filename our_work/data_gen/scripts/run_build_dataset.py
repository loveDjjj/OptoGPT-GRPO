from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work._shared.io.config import load_yaml_config, resolve_repo_path
from our_work.data_gen.pipeline.build_dataset import build_small_dataset
from our_work.data_gen.pipeline.material_registry import build_material_registry
from our_work._shared.utils.seed import set_global_seed


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

    return {
        "thickness_values_nm": resolve_thickness_values_nm(data_config),
        "sampling_device": str(sampling_config.get("device", "auto")),
        "sampling_batch_size": int(sampling_config.get("batch_size", 65536)),
        "max_duplicate_retry": int(sampling_config.get("max_duplicate_retry", 1000)),
        "tmm_batch_size": int(tmm_config.get("batch_size", 2048)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small our_work spectral dataset.")
    parser.add_argument("--config", required=True, help="Path to the dataset YAML config.")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
    config = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)
    runtime = resolve_data_gen_runtime_config(config)
    set_global_seed(int(config.get("seed", 42)))
    registry = build_material_registry(config["paths"]["database_dir"])
    build_small_dataset(
        output_dir=config["paths"]["output_dir"],
        database_path=config["paths"]["database_dir"],
        material_names=registry.material_names,
        thickness_values_nm=runtime["thickness_values_nm"],
        layer_counts=[int(value) for value in config["data"]["layer_counts"]],
        samples_per_bucket=int(config["data"]["samples_per_bucket"]),
        sampling_batch_size=runtime["sampling_batch_size"],
        tmm_batch_size=runtime["tmm_batch_size"],
        max_duplicate_retry=runtime["max_duplicate_retry"],
        sampling_device=runtime["sampling_device"],
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
        show_progress=bool(config.get("logging", {}).get("show_progress_bar", True)),
    )


if __name__ == "__main__":
    main()
