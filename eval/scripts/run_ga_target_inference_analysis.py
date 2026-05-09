from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _shared.io.config import resolve_repo_path
from data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from pretrain.dataset.tokenizer import SpectralStructureTokenizer
from pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def resolve_checkpoint_dir(path: str | Path) -> Path:
    resolved = resolve_repo_path(path, project_root=PROJECT_ROOT)
    if (resolved / "config.json").exists():
        return resolved
    checkpoint_dirs = [child for child in resolved.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directory found under: {resolved}")
    checkpoint_dirs.sort(key=lambda child: int(child.name.split("-")[-1]))
    return checkpoint_dirs[-1]


def build_absorption_target(
    wavelengths_um: np.ndarray,
    bands: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    absorption = np.zeros_like(wavelengths_um, dtype=np.float32)
    mask = np.zeros_like(wavelengths_um, dtype=bool)
    for idx, band in enumerate(bands):
        start_um = float(band["start_um"])
        end_um = float(band["end_um"])
        target_abs = float(band["absorption"])
        if idx == 0:
            band_mask = (wavelengths_um >= start_um) & (wavelengths_um <= end_um)
        else:
            band_mask = (wavelengths_um > start_um) & (wavelengths_um <= end_um)
        absorption[band_mask] = target_abs
        mask |= band_mask
    return absorption, mask


def absorption_to_rt_proxy(absorption: np.ndarray) -> np.ndarray:
    # The GA custom tasks define absorption targets only.
    # For model conditioning we use a simple proxy: T=0, R=1-A.
    target_t = np.zeros_like(absorption, dtype=np.float32)
    target_r = np.clip(1.0 - absorption, 0.0, 1.0).astype(np.float32)
    return np.concatenate([target_r, target_t], axis=0)


@torch.inference_mode()
def sample_structure_tokens(
    model,
    tokenizer,
    spectra: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[list[str]]:
    batch_size = spectra.size(0)
    input_ids = torch.full(
        (batch_size, 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=spectra.device,
    )
    blocked_ids = {
        int(tokenizer.pad_token_id),
        int(tokenizer.bos_token_id),
        int(tokenizer.unk_token_id),
    }
    top_k_value = max(0, int(top_k))
    top_p_value = float(max(0.0, min(1.0, top_p)))
    temperature_value = float(max(1e-6, temperature))

    for _ in range(int(max_new_tokens)):
        attention_mask = torch.ones(
            (batch_size, model.config.prefix_length + input_ids.size(1)),
            dtype=torch.long,
            device=spectra.device,
        )
        outputs = model(spectra=spectra, input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :].clone()
        logits[:, sorted(blocked_ids)] = float("-inf")
        logits = logits / temperature_value

        if top_k_value > 0 and top_k_value < logits.size(-1):
            topk_vals, _ = torch.topk(logits, k=top_k_value, dim=-1)
            kth = topk_vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

        if top_p_value < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative_probs > top_p_value
            cutoff_mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(cutoff_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_ids], dim=1)

    return [tokenizer.decode(row.tolist()) for row in input_ids]


def evaluate_structures_in_batches(
    structure_token_groups: list[list[str]],
    *,
    database_path: str,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    incident_angle: float,
    polarization: int,
    tolerance: float,
    complex_dtype: str,
    tmm_batch_size: int,
    tmm_device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(structure_token_groups)
    reflection = np.full((n, num_points), np.nan, dtype=np.float32)
    transmission = np.full((n, num_points), np.nan, dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)

    buckets: dict[int, list[tuple[int, list[str]]]] = defaultdict(list)
    for idx, tokens in enumerate(structure_token_groups):
        buckets[len(tokens)].append((idx, tokens))

    all_chunks: list[list[tuple[int, list[str]]]] = []
    for _, items in sorted(buckets.items()):
        for start in range(0, len(items), max(1, int(tmm_batch_size))):
            all_chunks.append(items[start : start + max(1, int(tmm_batch_size))])

    for chunk in tqdm(all_chunks, desc="tmm", unit="chunk", dynamic_ncols=True):
        chunk_indices = [item[0] for item in chunk]
        chunk_groups = [item[1] for item in chunk]
        _, reflections, transmissions, ok_mask = simulate_structure_batch(
            chunk_groups,
            database_path=database_path,
            wavelength_range_um=wavelength_range_um,
            num_points=num_points,
            incident_angle=incident_angle,
            polarization=polarization,
            tolerance=tolerance,
            complex_dtype=complex_dtype,
            device=tmm_device,
        )
        for local_idx, global_idx in enumerate(chunk_indices):
            if not bool(ok_mask[local_idx]):
                continue
            reflection[global_idx] = np.asarray(reflections[local_idx], dtype=np.float32)
            transmission[global_idx] = np.asarray(transmissions[local_idx], dtype=np.float32)
            valid[global_idx] = True
    return reflection, transmission, valid


def masked_absorption_mse(
    pred_r: np.ndarray,
    pred_t: np.ndarray,
    target_a: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    pred_a = np.clip(1.0 - pred_r - pred_t, 0.0, 1.0)
    diff = pred_a - target_a.reshape(1, -1)
    if not np.any(mask):
        return np.mean(diff**2, axis=1)
    masked = diff[:, mask]
    return np.mean(masked**2, axis=1)


def plot_best_rt_a(
    *,
    wavelengths_um: np.ndarray,
    target_r: np.ndarray,
    target_t: np.ndarray,
    target_a: np.ndarray,
    pred_r: np.ndarray,
    pred_t: np.ndarray,
    pred_a: np.ndarray,
    output_dir: Path,
) -> None:
    for name, target_curve, pred_curve in (
        ("R", target_r, pred_r),
        ("T", target_t, pred_t),
        ("A", target_a, pred_a),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(wavelengths_um, target_curve, label=f"Target {name}", linewidth=2)
        ax.plot(wavelengths_um, pred_curve, label=f"Pred {name}", linewidth=1.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(name)
        ax.set_title(f"Best Sample {name} Curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"best_{name}.png", dpi=220)
        plt.close(fig)


def plot_error_summary(
    *,
    errors: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(errors, bins=40, alpha=0.85)
    ax.set_xlabel("Masked Absorption MSE")
    ax.set_ylabel("Count")
    ax.set_title("All Sample Error Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "error_hist.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sorted_err = np.sort(errors)
    ax.plot(np.arange(1, len(sorted_err) + 1), sorted_err)
    ax.set_xlabel("Sorted Sample Index")
    ax.set_ylabel("Masked Absorption MSE")
    ax.set_title("Error CDF-like Curve (Sorted Errors)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "error_sorted_curve.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze checkpoint predictions against GA custom target spectra.")
    parser.add_argument("--ga-config", default="ga/configs/ga_custom_tasks.yaml")
    parser.add_argument("--checkpoint-dir", default="outputs/our_work/pretrain/a100_4gpu/checkpoint-980")
    parser.add_argument("--database-dir", default="_shared/database")
    parser.add_argument("--output-root", default="outputs/our_work/eval/ga_target_infer_analysis")
    parser.add_argument("--samples-per-target", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tmm-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ga_config_path = resolve_repo_path(args.ga_config, project_root=PROJECT_ROOT)
    ga_cfg = yaml.safe_load(ga_config_path.read_text(encoding="utf-8"))
    tmm_cfg = ga_cfg.get("tmm", {})
    target_cfg = ga_cfg.get("targets", {})
    task_specs = list(target_cfg.get("tasks", []))
    include_ids = target_cfg.get("include_ids")
    if include_ids:
        allowed = {str(x) for x in include_ids}
        task_specs = [task for task in task_specs if str(task.get("target_id")) in allowed]
    if not task_specs:
        raise ValueError("No tasks found in ga_custom_tasks.yaml")

    wavelength_min = float(tmm_cfg.get("wavelength_range_um", [2.0, 15.0])[0])
    wavelength_max = float(tmm_cfg.get("wavelength_range_um", [2.0, 15.0])[1])
    num_points = int(tmm_cfg.get("num_points", 1024))
    wavelengths = np.linspace(wavelength_min, wavelength_max, num_points, dtype=np.float32)

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    torch_device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SpectralGPTForCausalLM.from_pretrained(checkpoint_dir)
    model.to(torch_device)
    model.eval()
    tokenizer = SpectralStructureTokenizer.from_pretrained(checkpoint_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root_out = resolve_repo_path(args.output_root, project_root=PROJECT_ROOT) / f"{checkpoint_dir.name}_{timestamp}"
    root_out.mkdir(parents=True, exist_ok=False)

    summary_rows: list[dict] = []
    for task in task_specs:
        target_id = str(task["target_id"])
        task_out = root_out / target_id
        task_out.mkdir(parents=True, exist_ok=True)

        bands = list(task.get("bands", []))
        target_a, loss_mask = build_absorption_target(wavelengths, bands)
        target_rt = absorption_to_rt_proxy(target_a)
        target_r = target_rt[:num_points]
        target_t = target_rt[num_points:]

        sample_count = int(args.samples_per_target)
        batch_size = max(1, int(args.batch_size))
        token_groups: list[list[str]] = []
        for start in tqdm(range(0, sample_count, batch_size), desc=f"decode:{target_id}", unit="batch", dynamic_ncols=True):
            cur_bs = min(batch_size, sample_count - start)
            spectra_np = np.repeat(target_rt.reshape(1, -1), cur_bs, axis=0).astype(np.float32)
            spectra_t = torch.from_numpy(spectra_np).to(torch_device)
            token_groups.extend(
                sample_structure_tokens(
                    model=model,
                    tokenizer=tokenizer,
                    spectra=spectra_t,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                )
            )

        pred_r, pred_t, valid_mask = evaluate_structures_in_batches(
            token_groups,
            database_path=str(resolve_repo_path(args.database_dir, project_root=PROJECT_ROOT)),
            wavelength_range_um=(wavelength_min, wavelength_max),
            num_points=num_points,
            incident_angle=float(tmm_cfg.get("incident_angle", 0.0)),
            polarization=int(tmm_cfg.get("polarization", 0)),
            tolerance=float(tmm_cfg.get("tolerance", 1.0e-3)),
            complex_dtype=str(tmm_cfg.get("complex_dtype", "complex128")),
            tmm_batch_size=int(args.tmm_batch_size),
            tmm_device=None if str(args.device).strip().lower() == "auto" else str(args.device),
        )
        if not np.any(valid_mask):
            (task_out / "note.txt").write_text("No valid structure after TMM validation.\n", encoding="utf-8")
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_r = pred_r[valid_indices]
        valid_t = pred_t[valid_indices]
        errors = masked_absorption_mse(valid_r, valid_t, target_a=target_a, mask=loss_mask)
        best_local = int(np.argmin(errors))
        best_global = int(valid_indices[best_local])
        best_r = valid_r[best_local]
        best_t = valid_t[best_local]
        best_a = np.clip(1.0 - best_r - best_t, 0.0, 1.0)

        plot_best_rt_a(
            wavelengths_um=wavelengths,
            target_r=target_r,
            target_t=target_t,
            target_a=target_a,
            pred_r=best_r,
            pred_t=best_t,
            pred_a=best_a,
            output_dir=task_out,
        )
        plot_error_summary(errors=errors, output_dir=task_out)

        np.save(task_out / "best_pred_r.npy", best_r)
        np.save(task_out / "best_pred_t.npy", best_t)
        np.save(task_out / "target_r.npy", target_r)
        np.save(task_out / "target_t.npy", target_t)
        np.save(task_out / "target_a.npy", target_a)
        np.save(task_out / "all_valid_errors.npy", errors)

        metrics = {
            "target_id": target_id,
            "description": str(task.get("description", "")),
            "sample_count": int(sample_count),
            "valid_count": int(valid_indices.size),
            "best_index_in_all_samples": int(best_global),
            "best_tokens": token_groups[best_global],
            "best_masked_abs_mse": float(errors[best_local]),
            "mean_masked_abs_mse": float(np.mean(errors)),
            "median_masked_abs_mse": float(np.median(errors)),
            "p90_masked_abs_mse": float(np.percentile(errors, 90)),
            "target_proxy_note": "Target R/T is a proxy converted from absorption target: T=0, R=1-A.",
        }
        (task_out / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(metrics)

    overall = {
        "ga_config": str(ga_config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "samples_per_target": int(args.samples_per_target),
        "num_targets": len(summary_rows),
        "targets": summary_rows,
    }
    (root_out / "overall_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(root_out), "targets_finished": len(summary_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

