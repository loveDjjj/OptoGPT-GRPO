from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work._shared.io.config import load_yaml_config, resolve_repo_path
from our_work.data_gen.pipeline.material_registry import build_material_registry
from our_work.data_gen.pipeline.token_vocab import build_token_vocab
from our_work.data_gen.scripts.run_build_dataset import resolve_thickness_values_nm
from our_work.pso.dataset_writer import write_pso_supplement_dataset
from our_work.pso.search import PSOSearchConfig, TMMEvaluationConfig, make_tmm_evaluator, run_pso_search
from our_work.pso.targets import TargetProfile, build_default_targets, build_fixed_band_targets, build_lorentzian_targets


def resolve_pso_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    search_cfg = config.get("search", {})
    return {
        "thickness_values_nm": resolve_thickness_values_nm(data_cfg),
        "layer_counts": [int(value) for value in data_cfg.get("layer_counts", [5, 6, 7, 8, 9, 10])],
        "train_ratio": float(data_cfg.get("train_ratio", 1.0)),
        "val_ratio": float(data_cfg.get("val_ratio", 0.0)),
        "population_size": int(search_cfg.get("population_size", 1024)),
        "iterations": int(search_cfg.get("iterations", 10)),
        "batch_size": int(search_cfg.get("batch_size", 1024)),
        "max_accepted_per_target_layer": int(search_cfg.get("max_accepted_per_target_layer", 100)),
        "acceptance_mse_threshold": float(search_cfg.get("acceptance_mse_threshold", 0.01)),
        "max_stagnant_iterations": int(search_cfg.get("max_stagnant_iterations", 5)),
        "max_restarts": int(search_cfg.get("max_restarts", 3)),
        "inertia": float(search_cfg.get("inertia", 0.7)),
        "cognitive": float(search_cfg.get("cognitive", 1.5)),
        "social": float(search_cfg.get("social", 1.5)),
        "device": str(search_cfg.get("device", "auto")),
    }


def build_targets_from_config(wavelengths_um: np.ndarray, target_cfg: dict[str, Any]) -> list[TargetProfile]:
    targets: list[TargetProfile] = []
    if bool(target_cfg.get("include_fixed", True)):
        targets.extend(build_fixed_band_targets(wavelengths_um))
    if bool(target_cfg.get("include_lorentzian", True)):
        lorentz_cfg = target_cfg.get("lorentzian", {})
        targets.extend(
            build_lorentzian_targets(
                wavelengths_um,
                center_min_um=float(lorentz_cfg.get("center_min_um", 2.1)),
                center_max_um=float(lorentz_cfg.get("center_max_um", 14.9)),
                center_step_um=float(lorentz_cfg.get("center_step_um", 0.1)),
                fwhm_um=float(lorentz_cfg.get("fwhm_um", 0.02)),
            )
        )
    if not targets:
        targets = build_default_targets(wavelengths_um)
    include_ids = target_cfg.get("include_ids")
    if include_ids:
        allowed = {str(value) for value in include_ids}
        targets = [target for target in targets if target.target_id in allowed]
    max_targets = target_cfg.get("max_targets")
    if max_targets is not None:
        targets = targets[: int(max_targets)]
    return targets


def build_work_items(
    *,
    target_ids: list[str],
    layer_counts: list[int],
    max_targets: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> list[tuple[str, int]]:
    selected_targets = target_ids[: int(max_targets)] if max_targets is not None else list(target_ids)
    all_items = [(target_id, int(layer_count)) for target_id in selected_targets for layer_count in layer_counts]
    if int(world_size) <= 1:
        return all_items
    return [item for index, item in enumerate(all_items) if index % int(world_size) == int(rank)]


def _resolve_rank_context(config: dict[str, Any]) -> tuple[int, int]:
    distributed_cfg = config.get("distributed", {})
    if not bool(distributed_cfg.get("enabled", False)):
        return 0, 1
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _material_names_from_config(config: dict[str, Any], database_dir: Path) -> list[str]:
    registry = build_material_registry(database_dir)
    configured = config.get("materials")
    if configured:
        names = [str(value) for value in configured]
        missing = sorted(set(names) - set(registry.material_names))
        if missing:
            raise ValueError(f"materials not found in database: {missing}")
        return sorted(names)
    return registry.material_names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a PSO supplement dataset for our_work.")
    parser.add_argument("--config", required=True, help="Path to a PSO supplement YAML config.")
    args = parser.parse_args(argv)

    config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
    config = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)
    seed = int(config.get("seed", 42))
    runtime = resolve_pso_runtime_config(config)

    paths_cfg = config["paths"]
    base_output_dir = Path(paths_cfg["output_dir"])
    database_dir = Path(paths_cfg["database_dir"])
    material_names = _material_names_from_config(config, database_dir)
    vocab = build_token_vocab(material_names, runtime["thickness_values_nm"])
    vocab_tokens = vocab.special_tokens + [token for token in vocab.token_to_id if token not in vocab.special_tokens]

    tmm_cfg = config["tmm"]
    wavelength_range_um = tuple(float(value) for value in tmm_cfg.get("wavelength_range_um", [2.0, 15.0]))
    num_points = int(tmm_cfg.get("num_points", 1024))
    wavelengths = np.linspace(wavelength_range_um[0], wavelength_range_um[1], num_points, dtype=np.float32)
    targets = build_targets_from_config(wavelengths, config.get("targets", {}))
    target_by_id = {target.target_id: target for target in targets}

    rank, world_size = _resolve_rank_context(config)
    work_items = build_work_items(
        target_ids=[target.target_id for target in targets],
        layer_counts=runtime["layer_counts"],
        max_targets=None,
        rank=rank,
        world_size=world_size,
    )

    tmm_config = TMMEvaluationConfig(
        database_path=str(database_dir),
        wavelength_range_um=wavelength_range_um,
        num_points=num_points,
        incident_angle=float(tmm_cfg.get("incident_angle", 0.0)),
        polarization=int(tmm_cfg.get("polarization", 0)),
        tolerance=float(tmm_cfg.get("tolerance", 1.0e-3)),
        complex_dtype=str(tmm_cfg.get("complex_dtype", "complex128")),
        batch_size=int(tmm_cfg.get("batch_size", runtime["batch_size"])),
        device=None if str(tmm_cfg.get("device", "auto")).strip().lower() == "auto" else str(tmm_cfg.get("device")),
    )
    evaluator = make_tmm_evaluator(tmm_config, acceptance_mse_threshold=runtime["acceptance_mse_threshold"])

    output_dir = base_output_dir if world_size == 1 else base_output_dir / f"rank{rank:02d}"
    all_accepted = []
    global_seen: set[tuple[str, ...]] = set()
    search_summaries: list[dict[str, Any]] = []
    for item_index, (target_id, layer_count) in enumerate(work_items):
        target = target_by_id[target_id]
        result = run_pso_search(
            target=target,
            material_names=material_names,
            thickness_values_nm=runtime["thickness_values_nm"],
            layer_count=layer_count,
            config=PSOSearchConfig(
                population_size=runtime["population_size"],
                iterations=runtime["iterations"],
                batch_size=runtime["batch_size"],
                max_accepted=runtime["max_accepted_per_target_layer"],
                acceptance_mse_threshold=runtime["acceptance_mse_threshold"],
                max_stagnant_iterations=runtime["max_stagnant_iterations"],
                max_restarts=runtime["max_restarts"],
                seed=seed + item_index,
                device=runtime["device"],
                inertia=runtime["inertia"],
                cognitive=runtime["cognitive"],
                social=runtime["social"],
            ),
            evaluator=evaluator,
        )
        newly_kept = 0
        global_duplicates = 0
        for accepted in result.accepted:
            key = tuple(accepted.structure_tokens)
            if key in global_seen:
                global_duplicates += 1
                continue
            global_seen.add(key)
            all_accepted.append(accepted)
            newly_kept += 1
        search_summaries.append(
            {
                "target_id": result.target_id,
                "layer_count": result.layer_count,
                "accepted_count": len(result.accepted),
                "globally_kept_count": newly_kept,
                "global_duplicate_count": global_duplicates,
                "shortfall": result.shortfall,
                "total_evaluated": result.total_evaluated,
                "duplicate_accepted": result.duplicate_accepted,
                "restarts_used": result.restarts_used,
            }
        )

    manifest = write_pso_supplement_dataset(
        output_dir=output_dir,
        accepted=all_accepted,
        token_to_id=vocab.token_to_id,
        vocab_tokens=vocab_tokens,
        records_per_shard=int(config.get("shards", {}).get("records_per_shard", 50000)),
        acceptance_mse_threshold=runtime["acceptance_mse_threshold"],
        train_ratio=runtime["train_ratio"],
        val_ratio=runtime["val_ratio"],
        seed=seed,
    )

    summary_path = output_dir / "stats" / "search_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "work_item_count": len(work_items),
                "accepted_count": len(all_accepted),
                "split_manifest": manifest,
                "search": search_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
