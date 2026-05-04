from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_material_heatmap(
    counts: np.ndarray,
    *,
    material_names: Sequence[str],
    layer_labels: Sequence[str],
    output_path: str | Path,
) -> None:
    output_path = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, max(4, len(material_names) * 0.3)))
    image = ax.imshow(counts, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=0)
    ax.set_yticks(range(len(material_names)))
    ax.set_yticklabels(material_names)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Material")
    ax.set_title("Material Distribution by Layer")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_thickness_heatmap(
    counts: np.ndarray,
    *,
    thickness_values_nm: Sequence[int],
    layer_labels: Sequence[str],
    output_path: str | Path,
) -> None:
    output_path = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, max(4, len(thickness_values_nm) * 0.15)))
    image = ax.imshow(counts, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels)
    tick_step = max(1, len(thickness_values_nm) // 20)
    yticks = list(range(0, len(thickness_values_nm), tick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(thickness_values_nm[index]) for index in yticks])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Thickness (nm)")
    ax.set_title("Thickness Distribution by Layer")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    output_path: str | Path,
    max_xticks: int | None = None,
) -> None:
    output_path = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values)
    if max_xticks is not None and int(max_xticks) > 0 and len(labels) > int(max_xticks):
        tick_indices = np.linspace(0, len(labels) - 1, num=int(max_xticks), dtype=int)
        ax.set_xticks(tick_indices.tolist())
        ax.set_xticklabels([labels[index] for index in tick_indices.tolist()], rotation=45, ha="right")
    else:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_spectrum_mean_std_plot(
    wavelengths_um: np.ndarray,
    mean_spectrum: np.ndarray,
    std_spectrum: np.ndarray,
    *,
    output_path: str | Path,
) -> None:
    output_path = _ensure_parent(output_path)
    half = mean_spectrum.size // 2
    mean_r = mean_spectrum[:half]
    mean_t = mean_spectrum[half:]
    std_r = std_spectrum[:half]
    std_t = std_spectrum[half:]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, mean_curve, std_curve, title in (
        (axes[0], mean_r, std_r, "Reflection"),
        (axes[1], mean_t, std_t, "Transmission"),
    ):
        axis.plot(wavelengths_um, mean_curve, color="tab:blue")
        axis.fill_between(wavelengths_um, mean_curve - std_curve, mean_curve + std_curve, alpha=0.25, color="tab:blue")
        axis.set_ylabel("Value")
        axis.set_title(f"{title} Mean ± Std")
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("Wavelength (um)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_pca_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: str | Path,
) -> None:
    output_path = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=6, alpha=0.6, cmap="tab20")
    ax.set_title("Spectrum PCA Scatter")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_cluster_representative_plot(
    wavelengths_um: np.ndarray,
    cluster_mean_spectra: np.ndarray,
    representative_spectra: np.ndarray,
    *,
    output_path: str | Path,
) -> list[str]:
    output_path = _ensure_parent(output_path)
    cluster_count = int(cluster_mean_spectra.shape[0])
    if cluster_count <= 0:
        return []

    cols = 4
    max_clusters_per_figure = 16
    half = cluster_mean_spectra.shape[1] // 2
    artifact_names: list[str] = []
    total_pages = int(np.ceil(cluster_count / max_clusters_per_figure))

    for page_index in range(total_pages):
        start = page_index * max_clusters_per_figure
        end = min(cluster_count, (page_index + 1) * max_clusters_per_figure)
        page_count = end - start
        rows = max(1, int(np.ceil(page_count / cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=True, sharey=True)
        for local_index in range(rows * cols):
            axis = axes[local_index // cols][local_index % cols]
            cluster_index = start + local_index
            if cluster_index >= end:
                axis.axis("off")
                continue
            mean_r = cluster_mean_spectra[cluster_index, :half]
            mean_t = cluster_mean_spectra[cluster_index, half:]
            rep_r = representative_spectra[cluster_index, :half]
            rep_t = representative_spectra[cluster_index, half:]
            axis.plot(wavelengths_um, mean_r, color="tab:red", label="mean R")
            axis.plot(wavelengths_um, mean_t, color="tab:blue", label="mean T")
            axis.plot(wavelengths_um, rep_r, color="tab:red", linestyle="--", alpha=0.6, label="rep R")
            axis.plot(wavelengths_um, rep_t, color="tab:blue", linestyle="--", alpha=0.6, label="rep T")
            axis.set_title(f"Cluster {cluster_index}")
            axis.grid(alpha=0.2)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        if total_pages == 1:
            page_path = output_path
            artifact_name = output_path.name
        else:
            page_path = output_path.with_name(f"{output_path.stem}_part{page_index + 1:02d}{output_path.suffix}")
            artifact_name = page_path.name
        fig.savefig(page_path, dpi=220)
        plt.close(fig)
        artifact_names.append(artifact_name)
    return artifact_names
