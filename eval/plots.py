from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_metric_histogram(values: list[float], *, title: str, xlabel: str, output_path: str | Path) -> None:
    if not values:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(40, max(10, len(values) // 2)), color="#2a6f97", edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_split_metric_comparison(split_values: dict[str, float | None], *, title: str, ylabel: str, output_path: str | Path) -> None:
    labels = []
    values = []
    for split_name, value in split_values.items():
        if value is None:
            continue
        labels.append(split_name)
        values.append(float(value))
    if not labels:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#2a6f97", "#bc4749"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_layer_metric_plot(per_layer_metrics: dict[str, dict], *, metric_key: str, title: str, ylabel: str, output_path: str | Path) -> None:
    points = [
        (int(layer_count), float(metrics[metric_key]))
        for layer_count, metrics in per_layer_metrics.items()
        if metrics.get(metric_key) is not None
    ]
    if not points:
        return
    points.sort(key=lambda item: item[0])
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("Target Layer Count")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_sample_comparison_plot(
    row: dict,
    *,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    output_path: str | Path,
) -> None:
    target = np.asarray(row["target_spectrum_rt"], dtype=np.float32)
    predicted = np.asarray(row["predicted_spectrum_rt"], dtype=np.float32)
    target_r = target[:num_points]
    target_t = target[num_points : 2 * num_points]
    pred_r = predicted[:num_points]
    pred_t = predicted[num_points : 2 * num_points]
    wavelengths = np.linspace(float(wavelength_range_um[0]), float(wavelength_range_um[1]), int(num_points), dtype=np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.0, 2.0])
    ax_text = fig.add_subplot(grid[0])
    ax_r = fig.add_subplot(grid[1])
    ax_t = fig.add_subplot(grid[2], sharex=ax_r)

    ax_text.axis("off")
    text = (
        f"sample_id: {row['sample_id']}\n"
        f"split: {row['split']} | target_layers: {row['target_layer_count']} | pred_layers: {row['prediction_layer_count']}\n"
        f"exact_match: {row['token_exact_match']} | generated_valid: {row['generated_valid']} | "
        f"rmse: {row['spectrum_rmse']:.6f} | mae: {row['spectrum_mae']:.6f}\n"
        f"target_tokens: {' '.join(row['target_tokens'])}\n"
        f"predicted_tokens: {' '.join(row['predicted_tokens'])}"
    )
    ax_text.text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    ax_r.plot(wavelengths, target_r, label="target R", color="#1f1f1f", linewidth=2.0)
    ax_r.plot(wavelengths, pred_r, label="pred R", color="#d62828", linewidth=1.6, linestyle="--")
    ax_r.set_ylabel("R")
    ax_r.grid(alpha=0.3)
    ax_r.legend()

    ax_t.plot(wavelengths, target_t, label="target T", color="#2a6f97", linewidth=2.0)
    ax_t.plot(wavelengths, pred_t, label="pred T", color="#22a648", linewidth=1.6, linestyle="--")
    ax_t.set_ylabel("T")
    ax_t.set_xlabel("Wavelength (um)")
    ax_t.grid(alpha=0.3)
    ax_t.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
