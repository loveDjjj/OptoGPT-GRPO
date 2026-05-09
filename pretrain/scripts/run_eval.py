from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from _shared.io.config import resolve_repo_path
from data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from pretrain.dataset.hf_dataset import load_split_records
from pretrain.dataset.tokenizer import SpectralStructureTokenizer
from pretrain.eval_outputs import build_summary_payload, create_eval_run_dir, write_results_jsonl
from pretrain.eval_plots import (
    plot_metric_histogram,
    plot_sample_spectrum,
    select_sample_plot_rows,
)
from pretrain.model.generation import generate_structure_tokens
from pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def _json_safe_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


def resolve_checkpoint_dir(path: str | Path) -> Path:
    path = resolve_repo_path(path, project_root=PROJECT_ROOT)
    if (path / "config.json").exists():
        return path

    checkpoint_dirs = [
        child for child in path.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directory found under: {path}")
    return max(checkpoint_dirs, key=lambda child: int(child.name.split("-")[-1]))


def load_eval_components(
    checkpoint_dir: str | Path,
    *,
    device: str | None = None,
) -> tuple[SpectralGPTForCausalLM, SpectralStructureTokenizer, torch.device]:
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
            f"but CLI provided {int(requested_num_points)}."
        )
    return int(requested_num_points)


@torch.inference_mode()
def run_eval_sample(model, tokenizer, spectra: torch.Tensor, max_new_tokens: int = 10) -> list[list[str]]:
    return generate_structure_tokens(
        model=model,
        tokenizer=tokenizer,
        spectra=spectra,
        max_new_tokens=max_new_tokens,
    )


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


def _validate_eval_config(*, database_path: str | Path, complex_dtype: str | torch.dtype) -> None:
    database_dir = Path(database_path)
    if not database_dir.exists() or not database_dir.is_dir():
        raise ValueError(f"database_path must point to an existing directory: {database_path}")

    if isinstance(complex_dtype, torch.dtype):
        if complex_dtype not in {torch.complex64, torch.complex128}:
            raise ValueError(f"unsupported complex_dtype: {complex_dtype}")
        return

    resolved = str(complex_dtype).strip().lower()
    if resolved not in {"complex64", "torch.complex64", "c64", "complex128", "torch.complex128", "c128"}:
        raise ValueError(f"unsupported complex_dtype: {complex_dtype}")


def _validate_spectrum_length(name: str, spectrum: np.ndarray, num_points: int) -> None:
    expected_length = 2 * int(num_points)
    if spectrum.ndim != 1 or spectrum.size != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} values for num_points={num_points}; "
            f"got {int(spectrum.size)}"
        )


def evaluate_token_prediction(
    *,
    record: dict,
    predicted_tokens: list[str],
    database_path: str,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    incident_angle: float,
    polarization: int,
    tolerance: float,
    complex_dtype: str,
) -> dict:
    target_tokens = list(record["structure_tokens"])
    target_spectrum = np.asarray(record["spectrum_rt"], dtype=np.float32).reshape(-1)
    _validate_spectrum_length("target_spectrum_rt", target_spectrum, num_points)
    _validate_eval_config(database_path=database_path, complex_dtype=complex_dtype)
    result = {
        "sample_id": record["sample_id"],
        "target_layer_count": int(record["layer_count"]),
        "prediction_layer_count": len(predicted_tokens),
        "target_tokens": target_tokens,
        "predicted_tokens": list(predicted_tokens),
        "token_exact_match": predicted_tokens == target_tokens,
        "generated_valid": False,
        "spectrum_rmse": None,
        "spectrum_mae": None,
        "target_spectrum_rt": target_spectrum.tolist(),
        "predicted_spectrum_rt": None,
    }

    if not predicted_tokens:
        return result
    if any(_is_invalid_structure_token(token) for token in predicted_tokens):
        return result
    if _has_missing_material_data(predicted_tokens, database_path):
        return result

    try:
        _, reflections, transmissions, ok_mask = simulate_structure_batch(
            [predicted_tokens],
            database_path=database_path,
            wavelength_range_um=wavelength_range_um,
            num_points=num_points,
            incident_angle=incident_angle,
            polarization=polarization,
            tolerance=tolerance,
            complex_dtype=complex_dtype,
        )
    except ValueError:
        return result

    if not bool(ok_mask[0]):
        return result

    predicted_spectrum = flatten_rt_spectrum(reflections[0], transmissions[0]).astype(np.float32).reshape(-1)
    _validate_spectrum_length("predicted_spectrum_rt", predicted_spectrum, num_points)
    diff = predicted_spectrum - target_spectrum
    result["generated_valid"] = True
    result["spectrum_rmse"] = float(np.sqrt(np.mean(diff**2)))
    result["spectrum_mae"] = float(np.mean(np.abs(diff)))
    result["predicted_spectrum_rt"] = predicted_spectrum.tolist()
    return result


def evaluate_records(
    *,
    model,
    tokenizer,
    records: list[dict],
    database_path: str,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    incident_angle: float,
    polarization: int,
    tolerance: float,
    complex_dtype: str,
    max_new_tokens: int,
    device: torch.device,
    show_progress: bool = True,
) -> list[dict]:
    results: list[dict] = []
    for record in records:
        target_spectrum = np.asarray(record["spectrum_rt"], dtype=np.float32).reshape(-1)
        _validate_spectrum_length("target_spectrum_rt", target_spectrum, num_points)
    _validate_eval_config(database_path=database_path, complex_dtype=complex_dtype)
    progress = tqdm(
        records,
        desc="eval",
        total=len(records),
        unit="sample",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    for record in progress:
        target_spectrum = np.asarray(record["spectrum_rt"], dtype=np.float32).reshape(-1)
        spectra = torch.from_numpy(target_spectrum.reshape(1, -1)).to(device)
        predicted_tokens = run_eval_sample(
            model=model,
            tokenizer=tokenizer,
            spectra=spectra,
            max_new_tokens=max_new_tokens,
        )[0]
        results.append(
            evaluate_token_prediction(
                record=record,
                predicted_tokens=predicted_tokens,
                database_path=database_path,
                wavelength_range_um=wavelength_range_um,
                num_points=num_points,
                incident_angle=incident_angle,
                polarization=polarization,
                tolerance=tolerance,
                complex_dtype=complex_dtype,
            )
        )
        if hasattr(progress, "set_postfix") and results:
            latest = results[-1]
            progress.set_postfix(
                {
                    "valid": int(sum(1 for row in results if row["generated_valid"])),
                    "exact": int(sum(1 for row in results if row["token_exact_match"])),
                    "last_rmse": latest["spectrum_rmse"],
                },
                refresh=False,
            )
    return results


def summarize_eval(results: list[dict]) -> dict:
    total = len(results)
    valid_results = [item for item in results if item["generated_valid"]]
    exact_matches = sum(1 for item in results if item["token_exact_match"])
    rmse_values = [item["spectrum_rmse"] for item in valid_results]
    return {
        "sample_count": total,
        "valid_generation_count": len(valid_results),
        "exact_match_count": exact_matches,
        "exact_match_rate": float(exact_matches / total) if total else 0.0,
        "mean_spectrum_rmse": float(np.mean(rmse_values)) if rmse_values else None,
    }


def _infer_run_name(checkpoint_dir: Path) -> str:
    if checkpoint_dir.name.startswith("checkpoint-"):
        return checkpoint_dir.parent.name
    return checkpoint_dir.name


def _write_eval_artifacts(
    *,
    run_dir: Path,
    rows: list[dict],
    metadata: dict,
    num_points: int,
    worst_sample_plots: int,
    random_sample_plots: int,
    disable_plots: bool,
) -> dict:
    artifacts: dict[str, str] = {}
    skipped_artifacts: dict[str, str] = {}

    if disable_plots:
        skipped_artifacts["plots"] = "disabled by cli"
    else:
        valid_rmse = [
            float(row["spectrum_rmse"])
            for row in rows
            if row.get("generated_valid") and row.get("spectrum_rmse") is not None
        ]
        valid_mae = [
            float(row["spectrum_mae"])
            for row in rows
            if row.get("generated_valid") and row.get("spectrum_mae") is not None
        ]
        if valid_rmse:
            plot_metric_histogram(
                values=valid_rmse,
                title="Spectrum RMSE",
                xlabel="rmse",
                output_path=run_dir / "plots" / "rmse_hist.png",
            )
            artifacts["rmse_hist"] = "plots/rmse_hist.png"
        else:
            skipped_artifacts["rmse_hist"] = "no valid rmse values"
        if valid_mae:
            plot_metric_histogram(
                values=valid_mae,
                title="Spectrum MAE",
                xlabel="mae",
                output_path=run_dir / "plots" / "mae_hist.png",
            )
            artifacts["mae_hist"] = "plots/mae_hist.png"
        else:
            skipped_artifacts["mae_hist"] = "no valid mae values"

        selections = select_sample_plot_rows(
            rows,
            worst_count=worst_sample_plots,
            random_count=random_sample_plots,
        )
        for bucket_name, bucket_rows in selections.items():
            if not bucket_rows:
                skipped_artifacts[f"{bucket_name}_samples"] = "no valid rows selected"
                continue
            for index, row in enumerate(bucket_rows, start=1):
                rel_path = f"samples/{bucket_name}-{index}-{row['sample_id']}.png"
                plot_sample_spectrum(
                    row=row,
                    output_path=run_dir / rel_path,
                    num_points=num_points,
                )
                row["sample_figure_path"] = rel_path
                row["selection_bucket"] = bucket_name
            artifacts[f"{bucket_name}_samples"] = f"samples/{bucket_name}-*.png"

    write_results_jsonl(rows, run_dir / "results.jsonl")
    artifacts["results_jsonl"] = "results.jsonl"
    summary = build_summary_payload(
        rows=rows,
        metadata=metadata,
        artifacts=artifacts,
        skipped_artifacts=skipped_artifacts,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    artifacts["summary_json"] = "summary.json"
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run smoke evaluation for a spectral checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--database-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--wavelength-min", type=float, default=2.0)
    parser.add_argument("--wavelength-max", type=float, default=15.0)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--incident-angle", type=float, default=0.0)
    parser.add_argument("--polarization", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--complex-dtype", default="complex128")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--worst-sample-plots", type=int, default=5)
    parser.add_argument("--random-sample-plots", type=int, default=5)
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    model, tokenizer, device = load_eval_components(args.checkpoint_dir, device=args.device)
    num_points = resolve_num_points(model, requested_num_points=args.num_points)
    dataset_dir = resolve_repo_path(args.dataset_dir, project_root=PROJECT_ROOT)
    database_dir = resolve_repo_path(args.database_dir, project_root=PROJECT_ROOT)
    records = load_split_records(dataset_dir, args.split)[: args.max_samples]
    resolved_checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    run_root = (
        resolve_repo_path(args.output_dir, project_root=PROJECT_ROOT)
        if args.output_dir
        else resolve_repo_path("outputs/our_work/pretrain", project_root=PROJECT_ROOT)
    )
    run_dir = create_eval_run_dir(run_root, run_name=_infer_run_name(resolved_checkpoint_dir))
    results = evaluate_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        database_path=str(database_dir),
        wavelength_range_um=(args.wavelength_min, args.wavelength_max),
        num_points=num_points,
        incident_angle=args.incident_angle,
        polarization=args.polarization,
        tolerance=args.tolerance,
        complex_dtype=args.complex_dtype,
        max_new_tokens=args.max_new_tokens,
        device=device,
        show_progress=not args.disable_progress,
    )
    metadata = {
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "dataset_dir": str(dataset_dir),
        "database_dir": str(database_dir),
        "split": args.split,
        "max_samples": len(records),
        "num_points": num_points,
        "wavelength_range_um": [args.wavelength_min, args.wavelength_max],
        "incident_angle": args.incident_angle,
        "polarization": args.polarization,
        "tolerance": args.tolerance,
        "complex_dtype": args.complex_dtype,
    }
    summary = _write_eval_artifacts(
        run_dir=run_dir,
        rows=results,
        metadata=metadata,
        num_points=num_points,
        worst_sample_plots=args.worst_sample_plots,
        random_sample_plots=args.random_sample_plots,
        disable_plots=args.disable_plots,
    )
    payload = {
        "summary": summary["global_metrics"],
        "results": results,
        "run_dir": str(run_dir),
    }
    payload = _json_safe_value(payload)
    text = json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False)
    print(text)
    if args.output_json:
        output_path = resolve_repo_path(args.output_json, project_root=PROJECT_ROOT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
