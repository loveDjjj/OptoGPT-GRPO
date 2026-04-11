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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small our_work spectral dataset.")
    parser.add_argument("--config", required=True, help="Path to the dataset YAML config.")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
    config = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)
    set_global_seed(int(config.get("seed", 42)))
    registry = build_material_registry(config["paths"]["database_dir"])
    build_small_dataset(
        output_dir=config["paths"]["output_dir"],
        database_path=config["paths"]["database_dir"],
        material_names=registry.material_names,
        thickness_values_nm=[int(value) for value in config["data"]["thickness_values_nm"]],
        layer_counts=[int(value) for value in config["data"]["layer_counts"]],
        samples_per_bucket=int(config["data"]["samples_per_bucket"]),
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
