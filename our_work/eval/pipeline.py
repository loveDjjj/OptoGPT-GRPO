from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from our_work._shared.io.config import resolve_repo_path
from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from our_work.eval.dataset import sample_records_from_shards, select_split_shard_paths
from our_work.eval.metrics import build_overall_summary, select_plot_rows, summarize_rows
from our_work.eval.plots import (
    save_layer_metric_plot,
    save_metric_histogram,
    save_sample_comparison_plot,
    save_split_metric_comparison,
)
from our_work.eval.reports import create_eval_run_dir, write_config_snapshot, write_json, write_jsonl
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.generation import generate_structure_tokens
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _init_distributed_if_needed() -> tuple[int, int]:
    if not _is_distributed():
        return 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        # NCCL requires each process to bind a unique local CUDA device
        # before collectives are initialized.
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _cleanup_distributed_if_needed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _rank_split_shards(shard_paths: list[Path], rank: int, world_size: int) -> list[Path]:
    if world_size <= 1:
        return shard_paths
    return [path for index, path in enumerate(shard_paths) if index % world_size == rank]


def _gather_rows(rows: list[dict], rank: int, world_size: int) -> list[dict]:
    if world_size <= 1:
        return rows
    payload = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(rows, payload, dst=0)
    if rank != 0:
        return []
    merged: list[dict] = []
    for part in payload or []:
        if part:
            merged.extend(part)
    return merged


def resolve_checkpoint_dir(path: str | Path) -> Path:
    path = resolve_repo_path(path)
    if (path / "config.json").exists():
        return path
    checkpoint_dirs = [child for child in path.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directory found under: {path}")
    return max(checkpoint_dirs, key=lambda child: int(child.name.split("-")[-1]))


def load_eval_components(checkpoint_dir: str | Path, *, device: str | None = None) -> tuple[SpectralGPTForCausalLM, SpectralStructureTokenizer, torch.device]:
    resolved_dir = resolve_checkpoint_dir(checkpoint_dir)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SpectralGPTForCausalLM.from_pretrained(resolved_dir)
    model.to(torch_device)
    model.eval()
    tokenizer = SpectralStructureTokenizer.from_pretrained(resolved_dir)
    return model, tokenizer, torch_device


def resolve_num_points(model, requested_num_points: int | None) -> int:
    config = getattr(model, "config", None)
    if config is None:
        if requested_num_points is None:
            raise ValueError("num_points is required when model.config.spectrum_dim is unavailable")
        return int(requested_num_points)
    spectrum_dim = int(getattr(config, "spectrum_dim", 0))
    if spectrum_dim <= 0 or spectrum_dim % 2 != 0:
        raise ValueError(f"model.config.spectrum_dim must be a positive even integer; got {spectrum_dim}")
    inferred_num_points = spectrum_dim // 2
    if requested_num_points is None:
        return inferred_num_points
    if int(requested_num_points) != inferred_num_points:
        raise ValueError(
            f"num_points mismatch: checkpoint expects {inferred_num_points} because spectrum_dim={spectrum_dim}, "
            f"but config provided {int(requested_num_points)}."
        )
    return int(requested_num_points)


def _is_invalid_structure_token(token: str) -> bool:
    parts = str(token).rsplit("_", 1)
    if len(parts) != 2 or not parts[0]:
        return True
    try:
        int(parts[1])
    except ValueError:
        return True
    return False


def _has_missing_material_data(predicted_tokens: list[str], database_path: str | Path) -> bool:
    database_dir = Path(database_path)
    for token in predicted_tokens:
        if _is_invalid_structure_token(token):
            return False
        material = str(token).rsplit("_", 1)[0]
        if material.strip().lower() == "air":
            continue
        if not ((database_dir / f"{material}.csv").exists() or (database_dir / f"{material}.xlsx").exists()):
            return True
    return False


def _predict_token_groups(*, model, tokenizer, records: list[dict], batch_size: int, max_new_tokens: int, device: torch.device) -> list[list[str]]:
    predictions: list[list[str]] = []
    for start in range(0, len(records), max(1, int(batch_size))):
        chunk = records[start : start + max(1, int(batch_size))]
        spectra = torch.from_numpy(np.asarray([row["spectrum_rt"] for row in chunk], dtype=np.float32)).to(device)
        predictions.extend(
            generate_structure_tokens(
                model=model,
                tokenizer=tokenizer,
                spectra=spectra,
                max_new_tokens=max_new_tokens,
            )
        )
    return predictions


def _evaluate_records(
    *,
    split_name: str,
    records: list[dict],
    predicted_groups: list[list[str]],
    database_path: str | Path,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    incident_angle: float,
    polarization: int,
    tolerance: float,
    complex_dtype: str,
    tmm_batch_size: int,
    tmm_device: str | None,
) -> list[dict]:
    rows: list[dict] = []
    valid_buckets: dict[int, list[tuple[int, list[str]]]] = {}
    for index, (record, predicted_tokens) in enumerate(zip(records, predicted_groups)):
        row = {
            "split": split_name,
            "sample_id": record["sample_id"],
            "target_layer_count": int(record["layer_count"]),
            "prediction_layer_count": len(predicted_tokens),
            "target_tokens": list(record["structure_tokens"]),
            "predicted_tokens": list(predicted_tokens),
            "token_exact_match": list(predicted_tokens) == list(record["structure_tokens"]),
            "generated_valid": False,
            "spectrum_rmse": None,
            "spectrum_mae": None,
            "target_spectrum_rt": list(record["spectrum_rt"]),
            "predicted_spectrum_rt": None,
        }
        rows.append(row)
        if not predicted_tokens:
            continue
        if any(_is_invalid_structure_token(token) for token in predicted_tokens):
            continue
        if _has_missing_material_data(predicted_tokens, database_path):
            continue
        valid_buckets.setdefault(len(predicted_tokens), []).append((index, predicted_tokens))

    for _, bucket_items in sorted(valid_buckets.items()):
        for start in range(0, len(bucket_items), max(1, int(tmm_batch_size))):
            chunk_items = bucket_items[start : start + max(1, int(tmm_batch_size))]
            chunk_indices = [item[0] for item in chunk_items]
            chunk_groups = [item[1] for item in chunk_items]
            _, reflections, transmissions, ok_mask = simulate_structure_batch(
                chunk_groups,
                database_path=str(database_path),
                wavelength_range_um=wavelength_range_um,
                num_points=num_points,
                incident_angle=incident_angle,
                polarization=polarization,
                tolerance=tolerance,
                complex_dtype=complex_dtype,
                device=tmm_device,
            )
            for local_index, row_index in enumerate(chunk_indices):
                if not bool(ok_mask[local_index]):
                    continue
                predicted_spectrum = flatten_rt_spectrum(reflections[local_index], transmissions[local_index]).astype(np.float32).reshape(-1)
                target_spectrum = np.asarray(rows[row_index]["target_spectrum_rt"], dtype=np.float32).reshape(-1)
                diff = predicted_spectrum - target_spectrum
                rows[row_index]["generated_valid"] = True
                rows[row_index]["spectrum_rmse"] = float(np.sqrt(np.mean(diff**2)))
                rows[row_index]["spectrum_mae"] = float(np.mean(np.abs(diff)))
                rows[row_index]["predicted_spectrum_rt"] = predicted_spectrum.tolist()
    return rows


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_eval_suite(config: dict) -> dict:
    rank, world_size = _init_distributed_if_needed()
    checkpoint_dir = resolve_checkpoint_dir(config["paths"]["checkpoint_dir"])
    dataset_dir = resolve_repo_path(config["paths"]["dataset_dir"])
    database_dir = resolve_repo_path(config["paths"]["database_dir"])
    output_root = resolve_repo_path(config["paths"]["output_dir"])
    run_name = str(
        config.get("experiment", {}).get(
            "name",
            checkpoint_dir.parent.name if checkpoint_dir.name.startswith("checkpoint-") else checkpoint_dir.name,
        )
    )

    requested_device = str(config.get("inference", {}).get("device", "auto"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if world_size > 1 else 0
    resolved_device = requested_device
    lowered = requested_device.strip().lower()
    if world_size > 1 and torch.cuda.is_available() and (lowered == "auto" or lowered == "cuda" or lowered.startswith("cuda:")):
        resolved_device = f"cuda:{local_rank}"
    model, tokenizer, device = load_eval_components(checkpoint_dir, device=resolved_device)
    num_points = resolve_num_points(model, requested_num_points=config["tmm"].get("num_points"))
    if rank == 0:
        run_dir = create_eval_run_dir(output_root, run_name=run_name)
        write_config_snapshot(run_dir / "config.snapshot.yaml", config)
        run_dir_str = str(run_dir)
    else:
        run_dir_str = ""
    if world_size > 1:
        payload = [run_dir_str]
        dist.broadcast_object_list(payload, src=0)
        run_dir_str = payload[0]
    run_dir = Path(run_dir_str)

    split_summaries: dict[str, dict] = {}
    selected_samples_payload: dict[str, dict] = {}
    comparison_artifacts: dict[str, str] = {}

    splits = list(config["data"].get("splits", ["train", "val"]))
    max_samples_cfg = config["data"].get("max_samples_per_split", 512)
    sample_mode = str(config["data"].get("sample_mode", "random"))
    max_shards_cfg = config["data"].get("max_shards_per_split")
    inference_cfg = config["inference"]
    tmm_cfg = config["tmm"]
    plots_cfg = config["plots"]
    output_cfg = config["outputs"]

    for split_index, split_name in enumerate(splits):
        if isinstance(max_samples_cfg, dict):
            max_samples = int(max_samples_cfg.get(split_name, 0))
        else:
            max_samples = int(max_samples_cfg)
        if isinstance(max_shards_cfg, dict):
            max_shards = max_shards_cfg.get(split_name)
        else:
            max_shards = max_shards_cfg
        shard_paths = select_split_shard_paths(
            dataset_dir,
            split_name,
            sample_mode=sample_mode,
            max_shards=None if max_shards is None else int(max_shards),
            seed=int(config["data"].get("seed", 42)) + split_index,
        )
        split_shards = _rank_split_shards(shard_paths, rank=rank, world_size=world_size)
        local_max_samples = int(max_samples // world_size) + (1 if rank < int(max_samples % world_size) else 0) if world_size > 1 else int(max_samples)
        records, local_total_count = sample_records_from_shards(
            split_shards,
            max_samples=local_max_samples,
            seed=int(config["data"].get("seed", 42)) + split_index + rank * 9973,
        )
        if world_size > 1:
            local_counts = [0 for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(local_total_count, local_counts, dst=0)
            total_count = int(sum(local_counts or [])) if rank == 0 else 0
        else:
            total_count = int(local_total_count)
        scanned_shard_count = len(shard_paths)
        predicted_groups = _predict_token_groups(
            model=model,
            tokenizer=tokenizer,
            records=records,
            batch_size=int(inference_cfg.get("batch_size", 256)),
            max_new_tokens=int(inference_cfg.get("max_new_tokens", 10)),
            device=device,
        )
        local_rows = _evaluate_records(
            split_name=split_name,
            records=records,
            predicted_groups=predicted_groups,
            database_path=database_dir,
            wavelength_range_um=tuple(tmm_cfg["wavelength_range_um"]),
            num_points=num_points,
            incident_angle=float(tmm_cfg.get("incident_angle", 0.0)),
            polarization=int(tmm_cfg.get("polarization", 0)),
            tolerance=float(tmm_cfg.get("tolerance", 1.0e-3)),
            complex_dtype=str(tmm_cfg.get("complex_dtype", "complex128")),
            tmm_batch_size=int(tmm_cfg.get("batch_size", 512)),
            tmm_device=tmm_cfg.get("device"),
        )
        rows = _gather_rows(local_rows, rank=rank, world_size=world_size)
        if rank != 0:
            continue
        summary = summarize_rows(rows)
        summary["sample_mode"] = sample_mode
        summary["sampled_count"] = len(rows)
        summary["available_count"] = int(total_count)
        summary["scanned_shard_count"] = int(scanned_shard_count)
        split_summaries[split_name] = summary

        if output_cfg.get("save_jsonl", True):
            split_rows = []
            for row in rows:
                slim = dict(row)
                if not output_cfg.get("save_spectra_in_results", False):
                    slim.pop("target_spectrum_rt", None)
                    slim.pop("predicted_spectrum_rt", None)
                split_rows.append(_json_safe(slim))
            write_jsonl(run_dir / "results" / f"{split_name}.jsonl", split_rows)

        split_plot_dir = run_dir / "plots" / split_name
        split_sample_dir = run_dir / "samples" / split_name
        split_plot_dir.mkdir(parents=True, exist_ok=True)
        split_sample_dir.mkdir(parents=True, exist_ok=True)

        valid_rmse = [float(row["spectrum_rmse"]) for row in rows if row.get("generated_valid") and row.get("spectrum_rmse") is not None]
        valid_mae = [float(row["spectrum_mae"]) for row in rows if row.get("generated_valid") and row.get("spectrum_mae") is not None]
        if plots_cfg.get("save_histograms", True):
            save_metric_histogram(valid_rmse, title=f"{split_name} RMSE", xlabel="rmse", output_path=split_plot_dir / "rmse_hist.png")
            save_metric_histogram(valid_mae, title=f"{split_name} MAE", xlabel="mae", output_path=split_plot_dir / "mae_hist.png")
            save_layer_metric_plot(
                summary["per_target_layer_count"],
                metric_key="mean_spectrum_rmse",
                title=f"{split_name} Mean RMSE by Target Layer Count",
                ylabel="RMSE",
                output_path=split_plot_dir / "layer_count_rmse.png",
            )

        selection = select_plot_rows(
            rows,
            worst_count=int(plots_cfg.get("worst_count", 8)),
            best_count=int(plots_cfg.get("best_count", 8)),
            mean_count=int(plots_cfg.get("mean_count", 8)),
        )
        selected_samples_payload[split_name] = {}
        if plots_cfg.get("save_sample_plots", True):
            for bucket_name, bucket_rows in selection.items():
                selected_samples_payload[split_name][bucket_name] = []
                bucket_dir = split_sample_dir / bucket_name
                bucket_dir.mkdir(parents=True, exist_ok=True)
                for row in bucket_rows:
                    rel_path = Path("samples") / split_name / bucket_name / f"{row['sample_id']}.png"
                    if row.get("generated_valid") and row.get("predicted_spectrum_rt") is not None:
                        save_sample_comparison_plot(
                            row,
                            wavelength_range_um=tuple(tmm_cfg["wavelength_range_um"]),
                            num_points=num_points,
                            output_path=run_dir / rel_path,
                        )
                    selected_samples_payload[split_name][bucket_name].append(
                        {
                            "sample_id": row["sample_id"],
                            "spectrum_rmse": row["spectrum_rmse"],
                            "spectrum_mae": row["spectrum_mae"],
                            "token_exact_match": row["token_exact_match"],
                            "generated_valid": row["generated_valid"],
                            "figure_path": str(rel_path).replace("\\", "/"),
                        }
                    )

    overall_summary = build_overall_summary(split_summaries)
    if rank != 0:
        _cleanup_distributed_if_needed(world_size)
        return _json_safe({"run_dir": str(run_dir), "summary": {"status": "non_main_rank"}})
    if plots_cfg.get("save_split_comparison", True):
        comparison_dir = run_dir / "plots" / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        save_split_metric_comparison(
            overall_summary["comparison"]["mean_spectrum_rmse"],
            title="Train vs Val Mean RMSE",
            ylabel="RMSE",
            output_path=comparison_dir / "train_vs_val_rmse.png",
        )
        save_split_metric_comparison(
            overall_summary["comparison"]["valid_generation_rate"],
            title="Train vs Val Valid Generation Rate",
            ylabel="Rate",
            output_path=comparison_dir / "train_vs_val_valid_rate.png",
        )
        save_split_metric_comparison(
            overall_summary["comparison"]["exact_match_rate"],
            title="Train vs Val Exact Match Rate",
            ylabel="Rate",
            output_path=comparison_dir / "train_vs_val_exact_match.png",
        )
        comparison_artifacts = {
            "rmse": "plots/comparison/train_vs_val_rmse.png",
            "valid_generation_rate": "plots/comparison/train_vs_val_valid_rate.png",
            "exact_match_rate": "plots/comparison/train_vs_val_exact_match.png",
        }

    payload = {
        "metadata": {
            "checkpoint_dir": str(checkpoint_dir),
            "dataset_dir": str(dataset_dir),
            "database_dir": str(database_dir),
            "splits": splits,
            "inference_batch_size": int(inference_cfg.get("batch_size", 256)),
            "max_new_tokens": int(inference_cfg.get("max_new_tokens", 10)),
            "tmm_batch_size": int(tmm_cfg.get("batch_size", 512)),
            "num_points": int(num_points),
            "wavelength_range_um": list(tmm_cfg["wavelength_range_um"]),
        },
        "overall_summary": overall_summary,
        "comparison_artifacts": comparison_artifacts,
    }
    if output_cfg.get("save_summary_json", True):
        write_json(run_dir / "summary.json", _json_safe(payload))
    if output_cfg.get("save_split_summary_json", True):
        write_json(run_dir / "split_summaries.json", _json_safe(split_summaries))
    if output_cfg.get("save_selected_samples_json", True):
        write_json(run_dir / "selected_samples.json", _json_safe(selected_samples_payload))
    _cleanup_distributed_if_needed(world_size)
    return _json_safe({"run_dir": str(run_dir), "summary": payload})
