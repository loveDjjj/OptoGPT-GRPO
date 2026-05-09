from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

_RAPIDS_IMPORT_ERROR: Exception | None = None
try:
    import cupy as cp
    import cudf
    from cuml.cluster import KMeans
    from cuml.decomposition import PCA
except Exception as exc:  # pragma: no cover - optional dependency
    cp = None
    cudf = None
    KMeans = None
    PCA = None
    _RAPIDS_IMPORT_ERROR = exc

from .io import extract_spectrum_matrix
from .plots import save_bar_chart, save_cluster_representative_plot, save_pca_scatter, save_spectrum_mean_std_plot


def _to_numpy(embeddings_cp: "cp.ndarray", sample_size: int, seed: int = 42) -> np.ndarray:
    total = int(embeddings_cp.shape[0])
    if total <= sample_size:
        return cp.asnumpy(embeddings_cp).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(total, size=int(sample_size), replace=False)
    return cp.asnumpy(embeddings_cp[cp.asarray(indices, dtype=cp.int64)]).astype(np.float32)


def _compute_cluster_metrics(embeddings_np: np.ndarray, labels_np: np.ndarray) -> dict[str, float | None]:
    if embeddings_np.shape[0] < 3 or np.unique(labels_np).size < 2:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    try:
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    except Exception:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(embeddings_np, labels_np)),
        "calinski_harabasz": float(calinski_harabasz_score(embeddings_np, labels_np)),
        "davies_bouldin": float(davies_bouldin_score(embeddings_np, labels_np)),
    }


def _select_best_k(
    *,
    embeddings_cp: "cp.ndarray",
    k_candidates: list[int],
    cluster_iterations: int,
    random_state: int,
    n_init: int,
    selection_strategy: str,
    primary_metric: str,
    metric_sample_size: int,
) -> tuple[int, list[dict[str, float | int | None]], dict[str, str]]:
    metrics_log: list[dict[str, float | int | None]] = []
    sampled_np = _to_numpy(embeddings_cp, sample_size=max(2000, int(metric_sample_size)), seed=int(random_state))
    if sampled_np.shape[0] < 3:
        fallback_k = max(1, min(k_candidates))
        return fallback_k, metrics_log, {"reason": "insufficient_samples_for_metric"}

    candidate_iter = (
        tqdm(k_candidates, desc="spectrum:auto_k", unit="k", dynamic_ncols=True)
        if tqdm is not None
        else k_candidates
    )
    for candidate in candidate_iter:
        k = max(1, min(int(candidate), sampled_np.shape[0] - 1))
        if k < 2:
            continue
        kmeans = KMeans(
            n_clusters=int(k),
            max_iter=int(cluster_iterations),
            output_type="cupy",
            random_state=int(random_state),
            n_init=int(n_init),
        )
        sampled_cp = cp.asarray(sampled_np)
        labels_cp = kmeans.fit_predict(sampled_cp)
        labels_np = cp.asnumpy(labels_cp).astype(np.int64)
        metrics = _compute_cluster_metrics(sampled_np, labels_np)
        metrics_log.append({"k": int(k), **metrics})

    if not metrics_log:
        fallback_k = max(1, min(k_candidates))
        return fallback_k, metrics_log, {"reason": "no_valid_k"}

    primary = str(primary_metric).strip().lower()
    strategy = str(selection_strategy).strip().lower()
    valid_rows = [row for row in metrics_log if row.get(primary) is not None]
    if not valid_rows:
        chosen = metrics_log[0]
        return int(chosen["k"]), metrics_log, {"reason": "primary_metric_unavailable", "primary_metric": primary}

    if strategy == "single_metric":
        reverse = primary != "davies_bouldin"
        chosen = sorted(valid_rows, key=lambda row: float(row[primary]), reverse=reverse)[0]
        return int(chosen["k"]), metrics_log, {"reason": "single_metric", "primary_metric": primary}

    # weighted_rank: robust default across metrics with different scales.
    def _rank(metric_name: str, descending: bool) -> dict[int, int]:
        rows = [row for row in metrics_log if row.get(metric_name) is not None]
        ordered = sorted(rows, key=lambda row: float(row[metric_name]), reverse=descending)
        return {int(row["k"]): idx for idx, row in enumerate(ordered)}

    sil_rank = _rank("silhouette", True)
    ch_rank = _rank("calinski_harabasz", True)
    db_rank = _rank("davies_bouldin", False)
    scores: dict[int, int] = {}
    for row in metrics_log:
        k = int(row["k"])
        if k in sil_rank and k in ch_rank and k in db_rank:
            scores[k] = sil_rank[k] + ch_rank[k] + db_rank[k]
    if not scores:
        chosen = sorted(valid_rows, key=lambda row: float(row[primary]), reverse=(primary != "davies_bouldin"))[0]
        return int(chosen["k"]), metrics_log, {"reason": "weighted_rank_fallback_primary", "primary_metric": primary}
    best_k = min(scores.items(), key=lambda item: item[1])[0]
    return int(best_k), metrics_log, {"reason": "weighted_rank"}


def resolve_analysis_device(device: str = "auto") -> str:
    resolved = str(device).strip().lower()
    if resolved == "auto":
        if cp is None:
            return "cpu"
        try:
            return "cuda" if int(cp.cuda.runtime.getDeviceCount()) > 0 else "cpu"
        except Exception:
            return "cpu"
    return resolved


def _wavelength_axis(num_points: int, wavelength_min: float | None, wavelength_max: float | None) -> np.ndarray:
    if wavelength_min is None or wavelength_max is None:
        return np.arange(int(num_points), dtype=np.float32)
    return np.linspace(float(wavelength_min), float(wavelength_max), int(num_points), dtype=np.float32)


def _sample_rows_cp(
    rows: "cp.ndarray",
    *,
    target_count: int,
    estimated_total_rows: int,
) -> tuple["cp.ndarray", "cp.ndarray"]:
    if target_count <= 0 or rows.shape[0] == 0:
        empty = rows[:0]
        return empty, cp.empty((0,), dtype=cp.int64)
    if target_count >= estimated_total_rows:
        indices = cp.arange(rows.shape[0], dtype=cp.int64)
        return rows, indices
    take_count = max(1, int(np.ceil(rows.shape[0] * float(target_count) / float(estimated_total_rows))))
    take_count = min(int(rows.shape[0]), int(take_count))
    indices = cp.random.permutation(rows.shape[0])[:take_count]
    return rows[indices], indices


def analyze_spectrum_distribution(
    *,
    scope_name: str,
    frame_factory: Callable[[], Iterable[cudf.DataFrame]],
    estimated_total_rows: int,
    output_dir: str | Path,
    wavelength_min: float | None,
    wavelength_max: float | None,
    engine: str,
    pca_components: int,
    pca_fit_samples: int,
    cluster_count: int,
    cluster_fit_samples: int,
    cluster_iterations: int,
    scatter_max_points: int,
    cluster_mode: str = "fixed_k",
    k_candidates: list[int] | None = None,
    selection_strategy: str = "weighted_rank",
    primary_metric: str = "silhouette",
    metric_sample_size: int = 15000,
    random_state: int = 42,
    n_init: int = 1,
    device: str = "auto",
) -> dict:
    output_dir = Path(output_dir)
    resolved_device = resolve_analysis_device(device)
    if str(engine).strip().lower() != "rapids":
        raise ValueError(f"unsupported spectrum analysis engine: {engine}")
    if cp is None or cudf is None or PCA is None or KMeans is None:
        detail = f"{type(_RAPIDS_IMPORT_ERROR).__name__}: {_RAPIDS_IMPORT_ERROR}" if _RAPIDS_IMPORT_ERROR else "unknown import error"
        raise RuntimeError(f"RAPIDS import failed for spectrum analysis: {detail}") from _RAPIDS_IMPORT_ERROR
    if resolved_device != "cuda":
        raise RuntimeError("RAPIDS spectrum analysis currently requires CUDA-capable execution")

    sample_count = 0
    spectrum_dim = None
    spectrum_sum = None
    spectrum_sum_sq = None
    layer_count_hist = Counter()
    fit_rows_chunks: list[cp.ndarray] = []

    fit_target_count = max(int(pca_fit_samples), int(cluster_fit_samples))

    # Pass 1: mean/std + PCA/KMeans fit reservoir.
    pass1_iter = (
        tqdm(frame_factory(), desc=f"spectrum:{scope_name}:pass1", unit="shard", dynamic_ncols=True)
        if tqdm is not None
        else frame_factory()
    )
    for frame in pass1_iter:
        if len(frame) == 0:
            continue
        spectra_cp = extract_spectrum_matrix(frame)
        if spectrum_dim is None:
            spectrum_dim = int(spectra_cp.shape[1])
            spectrum_sum = cp.zeros((spectrum_dim,), dtype=cp.float64)
            spectrum_sum_sq = cp.zeros((spectrum_dim,), dtype=cp.float64)
        spectrum_sum += spectra_cp.astype(cp.float64).sum(axis=0)
        spectrum_sum_sq += cp.square(spectra_cp.astype(cp.float64)).sum(axis=0)
        sample_count += int(spectra_cp.shape[0])
        layer_values = frame["layer_count"].to_pandas().tolist()
        for value in layer_values:
            layer_count_hist[int(value)] += 1
        sampled_rows, _ = _sample_rows_cp(
            spectra_cp,
            target_count=int(fit_target_count),
            estimated_total_rows=max(1, int(estimated_total_rows)),
        )
        if sampled_rows.shape[0] > 0:
            fit_rows_chunks.append(sampled_rows)

    if sample_count == 0 or spectrum_dim is None or spectrum_sum is None or spectrum_sum_sq is None:
        summary = {"scope": scope_name, "sample_count": 0, "artifacts": {}, "skipped_reason": "no records"}
        (output_dir / "spectrum_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    mean_cp = (spectrum_sum / sample_count).astype(cp.float32)
    variance_cp = (spectrum_sum_sq / sample_count) - cp.square(mean_cp.astype(cp.float64))
    std_cp = cp.sqrt(cp.maximum(variance_cp, 0.0)).astype(cp.float32)

    if fit_rows_chunks:
        fit_cp = cp.concatenate(fit_rows_chunks, axis=0)
    else:
        fit_cp = cp.empty((0, spectrum_dim), dtype=cp.float32)
    if fit_cp.shape[0] == 0:
        summary = {"scope": scope_name, "sample_count": int(sample_count), "artifacts": {}, "skipped_reason": "no fit rows"}
        (output_dir / "spectrum_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    mean_fit_cp = mean_cp.reshape(1, -1)
    std_fit_cp = cp.maximum(std_cp, cp.float32(1e-6)).reshape(1, -1)
    standardized_fit_cp = (fit_cp - mean_fit_cp) / std_fit_cp

    actual_components = max(2, min(int(pca_components), int(standardized_fit_cp.shape[1]), int(standardized_fit_cp.shape[0])))
    pca = PCA(n_components=int(actual_components), output_type="cupy")
    fit_embeddings_cp = pca.fit_transform(standardized_fit_cp)
    explained_ratio = cp.asnumpy(pca.explained_variance_ratio_)

    if str(cluster_mode).strip().lower() == "auto_k":
        candidate_values = [int(value) for value in (k_candidates or [])]
        filtered_candidates = sorted(
            {
                max(2, min(int(value), int(fit_embeddings_cp.shape[0]) - 1))
                for value in candidate_values
                if int(value) >= 2
            }
        )
        if not filtered_candidates:
            filtered_candidates = [max(2, min(int(cluster_count), int(fit_embeddings_cp.shape[0]) - 1))]
        actual_cluster_count, k_metrics, k_selection = _select_best_k(
            embeddings_cp=fit_embeddings_cp,
            k_candidates=filtered_candidates,
            cluster_iterations=int(cluster_iterations),
            random_state=int(random_state),
            n_init=int(n_init),
            selection_strategy=selection_strategy,
            primary_metric=primary_metric,
            metric_sample_size=int(metric_sample_size),
        )
    else:
        actual_cluster_count = max(1, min(int(cluster_count), int(fit_embeddings_cp.shape[0])))
        k_metrics = []
        k_selection = {"reason": "fixed_k"}
    kmeans = KMeans(
        n_clusters=int(actual_cluster_count),
        max_iter=int(cluster_iterations),
        output_type="cupy",
        random_state=int(random_state),
        n_init=int(n_init),
    )
    fit_labels_cp = kmeans.fit_predict(fit_embeddings_cp)
    centers_cp = kmeans.cluster_centers_

    cluster_counts = np.zeros((actual_cluster_count,), dtype=np.int64)
    cluster_sum = np.zeros((actual_cluster_count, spectrum_dim), dtype=np.float64)
    representative_distances = np.full((actual_cluster_count,), np.inf, dtype=np.float64)
    representative_spectra = np.zeros((actual_cluster_count, spectrum_dim), dtype=np.float32)
    representative_sample_ids = [""] * actual_cluster_count
    cluster_layer_hist = defaultdict(Counter)
    scatter_coords_chunks: list[np.ndarray] = []
    scatter_labels_chunks: list[np.ndarray] = []
    half = spectrum_dim // 2
    wavelength_axis = _wavelength_axis(half, wavelength_min, wavelength_max)

    # Pass 2: project full dataset, assign clusters, aggregate and pick representatives.
    pass2_iter = (
        tqdm(frame_factory(), desc=f"spectrum:{scope_name}:pass2", unit="shard", dynamic_ncols=True)
        if tqdm is not None
        else frame_factory()
    )
    for frame in pass2_iter:
        if len(frame) == 0:
            continue
        spectra_cp = extract_spectrum_matrix(frame)
        spectra_np = cp.asnumpy(spectra_cp)
        standardized_cp = (spectra_cp - mean_fit_cp) / std_fit_cp
        projected_cp = pca.transform(standardized_cp)
        labels_cp = kmeans.predict(projected_cp)
        labels_np = cp.asnumpy(labels_cp).astype(np.int64)

        for cluster_id in range(actual_cluster_count):
            mask = labels_np == cluster_id
            if not np.any(mask):
                continue
            cluster_counts[cluster_id] += int(mask.sum())
            cluster_sum[cluster_id] += spectra_np[mask].astype(np.float64).sum(axis=0)

        projected_sample_cp, projected_indices_cp = _sample_rows_cp(
            projected_cp[:, :2],
            target_count=int(scatter_max_points),
            estimated_total_rows=max(1, int(estimated_total_rows)),
        )
        if projected_sample_cp.shape[0] > 0:
            scatter_coords_chunks.append(cp.asnumpy(projected_sample_cp).astype(np.float32))
            scatter_labels_chunks.append(labels_np[cp.asnumpy(projected_indices_cp)].astype(np.int64))

        distances_cp = cp.linalg.norm(projected_cp - centers_cp[labels_cp], axis=1)
        distances_np = cp.asnumpy(distances_cp)
        sample_ids = frame["sample_id"].to_pandas().tolist()
        layer_values = frame["layer_count"].to_pandas().tolist()
        for index, cluster_id in enumerate(labels_np.tolist()):
            cluster_layer_hist[int(cluster_id)][int(layer_values[index])] += 1
            if float(distances_np[index]) < representative_distances[int(cluster_id)]:
                representative_distances[int(cluster_id)] = float(distances_np[index])
                representative_spectra[int(cluster_id)] = spectra_np[index].astype(np.float32)
                representative_sample_ids[int(cluster_id)] = str(sample_ids[index])

    cluster_mean_spectra = cluster_sum / np.maximum(cluster_counts, 1)[:, None]
    scatter_coords = np.concatenate(scatter_coords_chunks, axis=0) if scatter_coords_chunks else np.zeros((0, 2), dtype=np.float32)
    scatter_labels = np.concatenate(scatter_labels_chunks, axis=0) if scatter_labels_chunks else np.zeros((0,), dtype=np.int64)
    if scatter_coords.shape[0] > int(scatter_max_points):
        keep_indices = np.random.default_rng(42).choice(scatter_coords.shape[0], size=int(scatter_max_points), replace=False)
        scatter_coords = scatter_coords[keep_indices]
        scatter_labels = scatter_labels[keep_indices]

    save_spectrum_mean_std_plot(
        wavelength_axis,
        cp.asnumpy(mean_cp),
        cp.asnumpy(std_cp),
        output_path=output_dir / "spectrum_mean_std.png",
    )
    if scatter_coords.size > 0:
        save_pca_scatter(scatter_coords, scatter_labels, output_path=output_dir / "spectrum_pca_scatter.png")
    save_bar_chart(
        [f"cluster_{index}" for index in range(actual_cluster_count)],
        cluster_counts.tolist(),
        title=f"Spectrum Cluster Sizes ({scope_name})",
        ylabel="Count",
        output_path=output_dir / "spectrum_cluster_sizes.png",
    )
    representative_artifacts = save_cluster_representative_plot(
        wavelength_axis,
        cluster_mean_spectra.astype(np.float32),
        representative_spectra.astype(np.float32),
        output_path=output_dir / "spectrum_cluster_representatives.png",
    )

    summary = {
        "scope": scope_name,
        "sample_count": int(sample_count),
        "spectrum_dim": int(spectrum_dim),
        "engine": "rapids",
        "explained_variance_ratio": explained_ratio.tolist(),
        "cluster_count": int(actual_cluster_count),
        "cluster_mode": str(cluster_mode),
        "k_selection": {
            "strategy": str(selection_strategy),
            "primary_metric": str(primary_metric),
            "metric_sample_size": int(metric_sample_size),
            "details": k_selection,
            "candidates": [int(value) for value in (k_candidates or [])],
            "metrics": k_metrics,
        },
        "cluster_sizes": {f"cluster_{index}": int(cluster_counts[index]) for index in range(actual_cluster_count)},
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
            "cluster_representatives": representative_artifacts,
        },
    }
    (output_dir / "spectrum_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
