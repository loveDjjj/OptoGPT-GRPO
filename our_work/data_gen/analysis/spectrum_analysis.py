from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .plots import save_bar_chart, save_cluster_representative_plot, save_pca_scatter, save_spectrum_mean_std_plot


def resolve_analysis_device(device: str = "auto") -> torch.device:
    resolved = str(device).strip().lower()
    if resolved == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _stack_spectra(batch: list[dict], device: torch.device) -> torch.Tensor:
    spectra = np.asarray([record["spectrum_rt"] for record in batch], dtype=np.float32)
    return torch.from_numpy(spectra).to(device=device, dtype=torch.float32)


def _wavelength_axis(num_points: int, wavelength_min: float | None, wavelength_max: float | None) -> np.ndarray:
    if wavelength_min is None or wavelength_max is None:
        return np.arange(int(num_points), dtype=np.float32)
    return np.linspace(float(wavelength_min), float(wavelength_max), int(num_points), dtype=np.float32)


def _update_reservoir(
    reservoir: list[np.ndarray],
    rows: np.ndarray,
    *,
    max_items: int,
    seen_count: int,
    rng: random.Random,
) -> int:
    for row in rows:
        seen_count += 1
        if len(reservoir) < max_items:
            reservoir.append(np.asarray(row, dtype=np.float32))
            continue
        replace_index = rng.randint(0, seen_count - 1)
        if replace_index < max_items:
            reservoir[replace_index] = np.asarray(row, dtype=np.float32)
    return seen_count


def _fit_kmeans_torch(
    embeddings: np.ndarray,
    *,
    cluster_count: int,
    iterations: int,
    device: torch.device,
) -> torch.Tensor:
    data = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
    actual_cluster_count = max(1, min(int(cluster_count), int(data.size(0))))
    initial_indices = torch.randperm(data.size(0), device=device)[:actual_cluster_count]
    centers = data.index_select(0, initial_indices).clone()
    for _ in range(max(1, int(iterations))):
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)
        counts = torch.bincount(labels, minlength=actual_cluster_count).to(dtype=torch.float32)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, labels, data)
        keep_mask = counts > 0
        new_centers[keep_mask] = new_centers[keep_mask] / counts[keep_mask].unsqueeze(-1)
        centers = torch.where(keep_mask.unsqueeze(-1), new_centers, centers)
    return centers


def analyze_spectrum_distribution(
    *,
    scope_name: str,
    batch_factory: Callable[[], Iterable[list[dict]]],
    output_dir: str | Path,
    wavelength_min: float | None,
    wavelength_max: float | None,
    pca_components: int,
    cluster_count: int,
    cluster_fit_samples: int,
    cluster_iterations: int,
    scatter_max_points: int,
    device: str = "auto",
) -> dict:
    output_dir = Path(output_dir)
    analysis_device = resolve_analysis_device(device)
    rng = random.Random(42)

    sample_count = 0
    spectrum_dim = None
    spectrum_sum = None
    spectrum_sum_sq = None
    layer_count_hist = Counter()
    for batch in batch_factory():
        if not batch:
            continue
        spectra = _stack_spectra(batch, analysis_device)
        if spectrum_dim is None:
            spectrum_dim = int(spectra.size(1))
            spectrum_sum = torch.zeros((spectrum_dim,), dtype=torch.float64, device=analysis_device)
            spectrum_sum_sq = torch.zeros((spectrum_dim,), dtype=torch.float64, device=analysis_device)
        spectrum_sum = spectrum_sum + spectra.to(torch.float64).sum(dim=0)
        spectrum_sum_sq = spectrum_sum_sq + spectra.to(torch.float64).square().sum(dim=0)
        sample_count += int(spectra.size(0))
        for record in batch:
            layer_count_hist[int(record.get("layer_count", len(record.get("materials", []))))] += 1

    if sample_count == 0 or spectrum_dim is None or spectrum_sum is None or spectrum_sum_sq is None:
        summary = {"scope": scope_name, "sample_count": 0, "artifacts": {}, "skipped_reason": "no records"}
        (output_dir / "spectrum_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    mean_spectrum = (spectrum_sum / sample_count).to(torch.float32)
    variance = (spectrum_sum_sq / sample_count) - mean_spectrum.to(torch.float64).square()
    std_spectrum = variance.clamp_min(0.0).sqrt().to(torch.float32)
    half = spectrum_dim // 2
    wavelength_axis = _wavelength_axis(half, wavelength_min, wavelength_max)

    covariance = torch.zeros((spectrum_dim, spectrum_dim), dtype=torch.float64, device=analysis_device)
    for batch in batch_factory():
        if not batch:
            continue
        spectra = _stack_spectra(batch, analysis_device).to(torch.float64)
        centered = spectra - mean_spectrum.to(torch.float64)
        covariance = covariance + centered.T @ centered
    covariance = covariance / max(1, sample_count - 1)

    eigvals, eigvecs = torch.linalg.eigh(covariance)
    actual_components = max(2, min(int(pca_components), int(eigvecs.size(1))))
    principal_components = eigvecs[:, -actual_components:]
    explained = eigvals[-actual_components:].clamp_min(0)
    total_explained = eigvals.clamp_min(0).sum().clamp_min(torch.tensor(1e-12, device=analysis_device))
    explained_ratio = (explained / total_explained).detach().cpu().numpy()

    fit_reservoir: list[np.ndarray] = []
    fit_seen = 0
    for batch in batch_factory():
        if not batch:
            continue
        spectra = _stack_spectra(batch, analysis_device)
        projected = (spectra - mean_spectrum) @ principal_components.to(torch.float32)
        fit_seen = _update_reservoir(
            fit_reservoir,
            projected.detach().cpu().numpy(),
            max_items=int(cluster_fit_samples),
            seen_count=fit_seen,
            rng=rng,
        )

    fit_embeddings = np.asarray(fit_reservoir, dtype=np.float32)
    centers = _fit_kmeans_torch(
        fit_embeddings,
        cluster_count=int(cluster_count),
        iterations=int(cluster_iterations),
        device=analysis_device,
    )
    actual_cluster_count = int(centers.size(0))

    cluster_counts = torch.zeros((actual_cluster_count,), dtype=torch.float64, device=analysis_device)
    cluster_sum = torch.zeros((actual_cluster_count, spectrum_dim), dtype=torch.float64, device=analysis_device)
    representative_distances = [float("inf")] * actual_cluster_count
    representative_spectra = [np.zeros((spectrum_dim,), dtype=np.float32) for _ in range(actual_cluster_count)]
    representative_sample_ids = [""] * actual_cluster_count
    cluster_layer_hist = defaultdict(Counter)
    scatter_coords_reservoir: list[np.ndarray] = []
    scatter_labels_reservoir: list[np.ndarray] = []
    scatter_seen = 0

    for batch in batch_factory():
        if not batch:
            continue
        spectra = _stack_spectra(batch, analysis_device)
        projected = (spectra - mean_spectrum) @ principal_components.to(torch.float32)
        distances = torch.cdist(projected, centers.to(torch.float32))
        labels = torch.argmin(distances, dim=1)
        min_distances = distances.gather(1, labels.unsqueeze(1)).squeeze(1)
        cluster_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        cluster_sum.index_add_(0, labels, spectra.to(torch.float64))

        projected_cpu = projected[:, :2].detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        stacked = np.concatenate([projected_cpu, labels_cpu.reshape(-1, 1)], axis=1)
        scatter_seen = _update_reservoir(
            scatter_coords_reservoir,
            stacked,
            max_items=int(scatter_max_points),
            seen_count=scatter_seen,
            rng=rng,
        )

        min_distances_cpu = min_distances.detach().cpu().tolist()
        spectra_cpu = spectra.detach().cpu().numpy()
        for index, record in enumerate(batch):
            cluster_id = int(labels_cpu[index])
            cluster_layer_hist[cluster_id][int(record.get("layer_count", len(record.get("materials", []))))] += 1
            if float(min_distances_cpu[index]) < representative_distances[cluster_id]:
                representative_distances[cluster_id] = float(min_distances_cpu[index])
                representative_spectra[cluster_id] = spectra_cpu[index].astype(np.float32)
                representative_sample_ids[cluster_id] = str(record.get("sample_id", f"{scope_name}-{index}"))

    cluster_mean_spectra = (cluster_sum / cluster_counts.clamp_min(1.0).unsqueeze(1)).to(torch.float32).cpu().numpy()
    representative_spectra_np = np.stack(representative_spectra, axis=0)
    cluster_counts_np = cluster_counts.to(torch.int64).cpu().numpy()
    scatter_matrix = np.asarray(scatter_coords_reservoir, dtype=np.float32)
    scatter_coords = scatter_matrix[:, :2] if scatter_matrix.size > 0 else np.zeros((0, 2), dtype=np.float32)
    scatter_labels = scatter_matrix[:, 2].astype(np.int64) if scatter_matrix.size > 0 else np.zeros((0,), dtype=np.int64)

    save_spectrum_mean_std_plot(
        wavelength_axis,
        mean_spectrum.detach().cpu().numpy(),
        std_spectrum.detach().cpu().numpy(),
        output_path=output_dir / "spectrum_mean_std.png",
    )
    if scatter_coords.size > 0:
        save_pca_scatter(scatter_coords, scatter_labels, output_path=output_dir / "spectrum_pca_scatter.png")
    save_bar_chart(
        [f"cluster_{index}" for index in range(actual_cluster_count)],
        cluster_counts_np.tolist(),
        title=f"Spectrum Cluster Sizes ({scope_name})",
        ylabel="Count",
        output_path=output_dir / "spectrum_cluster_sizes.png",
    )
    save_cluster_representative_plot(
        wavelength_axis,
        cluster_mean_spectra,
        representative_spectra_np,
        output_path=output_dir / "spectrum_cluster_representatives.png",
    )

    summary = {
        "scope": scope_name,
        "sample_count": int(sample_count),
        "spectrum_dim": int(spectrum_dim),
        "explained_variance_ratio": explained_ratio.tolist(),
        "cluster_count": int(actual_cluster_count),
        "cluster_sizes": {f"cluster_{index}": int(cluster_counts_np[index]) for index in range(actual_cluster_count)},
        "cluster_representatives": {
            f"cluster_{index}": {
                "sample_id": representative_sample_ids[index],
                "distance": float(representative_distances[index]),
                "layer_count_hist": {str(layer): int(count) for layer, count in cluster_layer_hist[index].items()},
            }
            for index in range(actual_cluster_count)
        },
        "layer_count_hist": {str(layer): int(count) for layer, count in layer_count_hist.items()},
        "artifacts": {
            "mean_std": "spectrum_mean_std.png",
            "pca_scatter": "spectrum_pca_scatter.png" if scatter_coords.size > 0 else None,
            "cluster_sizes": "spectrum_cluster_sizes.png",
            "cluster_representatives": "spectrum_cluster_representatives.png",
        },
    }
    (output_dir / "spectrum_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
