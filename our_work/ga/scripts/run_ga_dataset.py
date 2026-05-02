from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional on servers
    tqdm = None

from our_work._shared.io.config import load_yaml_config, resolve_repo_path
from our_work.data_gen.pipeline.material_registry import build_material_registry
from our_work.data_gen.pipeline.token_vocab import build_token_vocab
from our_work.data_gen.scripts.run_build_dataset import resolve_thickness_values_nm
from our_work.ga.dataset_writer import write_ga_supplement_dataset
from our_work.ga.search import GASearchConfig, TMMEvaluationConfig, make_tmm_evaluator, run_seeded_ga_search
from our_work.ga.targets import GATargetProfile, build_default_ga_targets, seed_thickness_values_nm
from our_work.ga.visualization import save_ga_spectrum_plots


def resolve_ga_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    search_cfg = config.get("search", {})
    thickness_values = resolve_thickness_values_nm(data_cfg)
    if bool(data_cfg.get("include_seed_thickness_values", True)):
        thickness_values = sorted(set(thickness_values) | set(seed_thickness_values_nm()))
    return {
        "thickness_values_nm": [int(value) for value in thickness_values],
        "train_ratio": float(data_cfg.get("train_ratio", 1.0)),
        "val_ratio": float(data_cfg.get("val_ratio", 0.0)),
        "max_samples_per_target": int(data_cfg.get("max_samples_per_target", data_cfg.get("target_sample_count", 100))),
        "population_size": int(search_cfg.get("population_size", 4096)),
        "generations_per_restart": int(search_cfg.get("generations_per_restart", search_cfg.get("generations", 80))),
        "restart_count": int(search_cfg.get("restart_count", search_cfg.get("max_restarts", 20))),
        "batch_size": int(search_cfg.get("batch_size", 1024)),
        "acceptance_floor_mse": float(search_cfg.get("acceptance_floor_mse", search_cfg.get("acceptance_mse_threshold", 0.005))),
        "elite_fraction": float(search_cfg.get("elite_fraction", 0.15)),
        "tournament_size": int(search_cfg.get("tournament_size", 4)),
        "crossover_rate": float(search_cfg.get("crossover_rate", 0.8)),
        "material_mutation_rate": float(search_cfg.get("material_mutation_rate", 0.05)),
        "thickness_mutation_rate": float(search_cfg.get("thickness_mutation_rate", 0.35)),
        "thickness_mutation_steps": int(search_cfg.get("thickness_mutation_steps", 6)),
        "random_injection_rate": float(search_cfg.get("random_injection_rate", 0.08)),
        "device": str(search_cfg.get("device", "auto")),
    }


def build_targets_from_config(wavelengths_um: np.ndarray, target_cfg: dict[str, Any]) -> list[GATargetProfile]:
    targets = build_default_ga_targets(wavelengths_um)
    include_ids = target_cfg.get("include_ids")
    if include_ids:
        allowed = {str(value) for value in include_ids}
        targets = [target for target in targets if target.target_id in allowed]
    if not targets:
        raise ValueError("No GA targets selected")
    return targets


def build_work_items(target_ids: list[str], *, rank: int = 0, world_size: int = 1) -> list[str]:
    if int(world_size) <= 1:
        return list(target_ids)
    return [target_id for index, target_id in enumerate(target_ids) if index % int(world_size) == int(rank)]


def progress_work_items(work_items: list[str], *, rank: int, world_size: int):
    if tqdm is None:
        return work_items
    return tqdm(work_items, total=len(work_items), desc=f"ga rank {rank}/{world_size}", unit="target", dynamic_ncols=True)


def _resolve_rank_context(config: dict[str, Any]) -> tuple[int, int, int]:
    distributed_cfg = config.get("distributed", {})
    if not bool(distributed_cfg.get("enabled", False)):
        return 0, 1, 0
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))


def _resolve_rank_device(requested_device: str, *, local_rank: int, dist_enabled: bool) -> str:
    resolved = str(requested_device).strip().lower()
    if not dist_enabled:
        return requested_device
    if resolved in {"auto", "cuda"} or resolved.startswith("cuda:"):
        if torch.cuda.is_available():
            return f"cuda:{int(local_rank)}"
    return requested_device


def _material_names_from_config(config: dict[str, Any], database_dir: Path) -> list[str]:
    registry = build_material_registry(database_dir)
    configured = config.get("materials")
    if configured:
        names = [str(value) for value in configured]
    else:
        names = list(registry.material_names)
    missing = sorted(set(names) - set(registry.material_names))
    if missing:
        raise ValueError(f"materials not found in database: {missing}")
    return sorted(names)


def _vocab_tokens(vocab) -> list[str]:
    return vocab.special_tokens + [token for token in vocab.token_to_id if token not in vocab.special_tokens]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a seeded genetic-algorithm supplement dataset for our_work.")
    parser.add_argument("--config", required=True, help="Path to a GA supplement YAML config.")
    args = parser.parse_args(argv)

    config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
    config = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)
    seed = int(config.get("seed", 42))
    runtime = resolve_ga_runtime_config(config)

    paths_cfg = config["paths"]
    base_output_dir = Path(paths_cfg["output_dir"])
    database_dir = Path(paths_cfg["database_dir"])
    material_names = _material_names_from_config(config, database_dir)
    vocab = build_token_vocab(material_names, runtime["thickness_values_nm"])

    tmm_cfg = config["tmm"]
    wavelength_range_um = tuple(float(value) for value in tmm_cfg.get("wavelength_range_um", [2.0, 15.0]))
    num_points = int(tmm_cfg.get("num_points", 1024))
    wavelengths = np.linspace(wavelength_range_um[0], wavelength_range_um[1], num_points, dtype=np.float32)
    targets = build_targets_from_config(wavelengths, config.get("targets", {}))
    target_by_id = {target.target_id: target for target in targets}

    rank, world_size, local_rank = _resolve_rank_context(config)
    dist_enabled = world_size > 1
    work_items = build_work_items([target.target_id for target in targets], rank=rank, world_size=world_size)
    tmm_device_raw = str(tmm_cfg.get("device", "auto"))
    tmm_device_name = _resolve_rank_device(tmm_device_raw, local_rank=local_rank, dist_enabled=dist_enabled)
    tmm_config = TMMEvaluationConfig(
        database_path=str(database_dir),
        wavelength_range_um=wavelength_range_um,
        num_points=num_points,
        incident_angle=float(tmm_cfg.get("incident_angle", 0.0)),
        polarization=int(tmm_cfg.get("polarization", 0)),
        tolerance=float(tmm_cfg.get("tolerance", 1.0e-3)),
        complex_dtype=str(tmm_cfg.get("complex_dtype", "complex128")),
        batch_size=int(tmm_cfg.get("batch_size", runtime["batch_size"])),
        device=None if tmm_device_name.strip().lower() == "auto" else tmm_device_name,
    )
    evaluator = make_tmm_evaluator(tmm_config)

    output_dir = base_output_dir if world_size == 1 else base_output_dir / f"rank{rank:02d}"
    all_accepted = []
    global_seen: set[tuple[str, ...]] = set()
    search_summaries: list[dict[str, Any]] = []
    progress = progress_work_items(work_items, rank=rank, world_size=world_size)

    def _set_progress(state: dict[str, Any]) -> None:
        if tqdm is None or not hasattr(progress, "set_postfix"):
            return
        progress.set_postfix(
            target=state["target_id"],
            restart=f"{int(state['restart_index']) + 1}/{int(state['restart_count'])}",
            gen=f"{int(state['generation']) + 1}/{int(state['generations_per_restart'])}",
            kept=f"{int(state['kept_count'])}/{int(state['max_samples_per_target'])}",
            best=f"{float(state['best_mse']):.4g}" if np.isfinite(float(state["best_mse"])) else "nan",
            worst=f"{float(state['worst_kept_mse']):.4g}" if np.isfinite(float(state["worst_kept_mse"])) else "nan",
        )

    for item_index, target_id in enumerate(progress):
        target = target_by_id[target_id]
        result = run_seeded_ga_search(
            target=target,
            material_names=material_names,
            thickness_values_nm=runtime["thickness_values_nm"],
            config=GASearchConfig(
                population_size=runtime["population_size"],
                generations_per_restart=runtime["generations_per_restart"],
                restart_count=runtime["restart_count"],
                batch_size=runtime["batch_size"],
                max_samples_per_target=runtime["max_samples_per_target"],
                acceptance_floor_mse=runtime["acceptance_floor_mse"],
                elite_fraction=runtime["elite_fraction"],
                tournament_size=runtime["tournament_size"],
                crossover_rate=runtime["crossover_rate"],
                material_mutation_rate=runtime["material_mutation_rate"],
                thickness_mutation_rate=runtime["thickness_mutation_rate"],
                thickness_mutation_steps=runtime["thickness_mutation_steps"],
                random_injection_rate=runtime["random_injection_rate"],
                seed=seed + item_index,
                device=_resolve_rank_device(runtime["device"], local_rank=local_rank, dist_enabled=dist_enabled),
            ),
            evaluator=evaluator,
            progress_callback=_set_progress,
        )
        newly_kept = 0
        global_duplicates = 0
        for item in result.accepted:
            key = tuple(item.structure_tokens)
            if key in global_seen:
                global_duplicates += 1
                continue
            global_seen.add(key)
            all_accepted.append(item)
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
                "replacement_count": result.replacement_count,
                "restarts_used": result.restarts_used,
            }
        )
        if tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(target=target_id, kept=f"{newly_kept}/{runtime['max_samples_per_target']}", status="done")

    manifest = write_ga_supplement_dataset(
        output_dir=output_dir,
        accepted=all_accepted,
        token_to_id=vocab.token_to_id,
        vocab_tokens=_vocab_tokens(vocab),
        records_per_shard=int(config.get("shards", {}).get("records_per_shard", 50000)),
        acceptance_floor_mse=runtime["acceptance_floor_mse"],
        train_ratio=runtime["train_ratio"],
        val_ratio=runtime["val_ratio"],
        seed=seed,
    )

    visualization_cfg = config.get("visualization", {})
    artifacts = []
    if bool(visualization_cfg.get("enabled", True)):
        artifacts = save_ga_spectrum_plots(
            accepted=all_accepted,
            targets=targets,
            wavelengths_um=wavelengths,
            output_dir=output_dir,
            top_k=int(visualization_cfg.get("top_k", 20)),
        )

    summary_path = output_dir / "stats" / "search_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "algorithm": "seeded_genetic_algorithm",
                "rank": rank,
                "world_size": world_size,
                "work_item_count": len(work_items),
                "accepted_count": len(all_accepted),
                "max_samples_per_target": runtime["max_samples_per_target"],
                "acceptance_floor_mse": runtime["acceptance_floor_mse"],
                "split_manifest": manifest,
                "visualization_artifacts": artifacts,
                "search": search_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
