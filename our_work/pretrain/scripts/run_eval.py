from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from our_work.pretrain.dataset.hf_dataset import load_split_records
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.eval_outputs import build_summary_payload, create_eval_run_dir, write_results_jsonl
from our_work.pretrain.eval_plots import (
    plot_metric_histogram,
    plot_sample_spectrum,
    select_sample_plot_rows,
)
from our_work.pretrain.model.generation import generate_structure_tokens
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def resolve_repo_path(path: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    search_roots = [project_root, *project_root.parents]
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return project_root / candidate


def resolve_checkpoint_dir(path: str | Path) -> Path:
    path = resolve_repo_path(path)
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


@torch.inference_mode()
def run_eval_sample(model, tokenizer, spectra: torch.Tensor, max_new_tokens: int = 10) -> list[list[str]]:
    return generate_structure_tokens(
        model=model,
        tokenizer=tokenizer,
        spectra=spectra,
        max_new_tokens=max_new_tokens,
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
        "target_spectrum_rt": np.asarray(record["spectrum_rt"], dtype=np.float32).tolist(),
        "predicted_spectrum_rt": None,
    }

    if not predicted_tokens:
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
    except Exception:
        return result

    if not bool(ok_mask[0]):
        return result

    predicted_spectrum = flatten_rt_spectrum(reflections[0], transmissions[0]).astype(np.float32)
    target_spectrum = np.asarray(record["spectrum_rt"], dtype=np.float32)
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
) -> list[dict]:
    results: list[dict] = []
    for record in records:
        spectra = torch.from_numpy(
            np.asarray([record["spectrum_rt"]], dtype=np.float32)
        ).to(device)
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
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--incident-angle", type=float, default=0.0)
    parser.add_argument("--polarization", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--complex-dtype", default="complex128")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--worst-sample-plots", type=int, default=5)
    parser.add_argument("--random-sample-plots", type=int, default=5)
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    model, tokenizer, device = load_eval_components(args.checkpoint_dir, device=args.device)
    dataset_dir = resolve_repo_path(args.dataset_dir)
    database_dir = resolve_repo_path(args.database_dir)
    records = load_split_records(dataset_dir, args.split)[: args.max_samples]
    resolved_checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    run_root = Path(args.output_dir) if args.output_dir else resolve_repo_path("our_work/pretrain/outputs")
    run_dir = create_eval_run_dir(run_root, run_name=_infer_run_name(resolved_checkpoint_dir))
    results = evaluate_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        database_path=str(database_dir),
        wavelength_range_um=(args.wavelength_min, args.wavelength_max),
        num_points=args.num_points,
        incident_angle=args.incident_angle,
        polarization=args.polarization,
        tolerance=args.tolerance,
        complex_dtype=args.complex_dtype,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    metadata = {
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "dataset_dir": str(dataset_dir),
        "database_dir": str(database_dir),
        "split": args.split,
        "max_samples": len(records),
        "num_points": args.num_points,
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
        num_points=args.num_points,
        worst_sample_plots=args.worst_sample_plots,
        random_sample_plots=args.random_sample_plots,
        disable_plots=args.disable_plots,
    )
    payload = {
        "summary": summary["global_metrics"],
        "results": results,
        "run_dir": str(run_dir),
    }
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
