from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pso.targets import build_default_targets, lorentzian_profile


def _normalise_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return list(value)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _resolve_shard_paths(dataset_dir: Path, splits: Sequence[str]) -> list[Path]:
    manifest_path = dataset_dir / "splits" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected: dict[str, None] = {}
    for split in splits:
        split_name = str(split)
        if split_name == "all":
            for name in ("train", "val", "test"):
                for shard_name in manifest.get(name, []):
                    selected[str(shard_name)] = None
            continue
        for shard_name in manifest.get(split_name, []):
            selected[str(shard_name)] = None
    return [dataset_dir / "shards" / shard_name for shard_name in selected]


def _load_records(dataset_dir: Path, splits: Sequence[str]) -> pd.DataFrame:
    shard_paths = _resolve_shard_paths(dataset_dir, splits)
    if not shard_paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in shard_paths]
    all_columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    # Some shards may contain PSO metadata columns that are all-null for fixed targets
    # but non-null for Lorentzian targets. Dropping all-null columns per shard before
    # concatenation avoids pandas' pending dtype change warning without losing data.
    merged = pd.concat([frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True)
    for column in all_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[all_columns]


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    prepared = frame.copy()
    prepared["structure_tokens"] = prepared["structure_tokens"].map(lambda value: [str(item) for item in _normalise_sequence(value)])
    prepared["materials"] = prepared["materials"].map(lambda value: [str(item) for item in _normalise_sequence(value)])
    prepared["thickness_nm"] = prepared["thickness_nm"].map(lambda value: [int(item) for item in _normalise_sequence(value)])
    prepared["spectrum_rt"] = prepared["spectrum_rt"].map(lambda value: [float(item) for item in _normalise_sequence(value)])
    prepared["structure_key"] = prepared["structure_tokens"].map(lambda values: "|".join(values))
    prepared["total_thickness_nm"] = prepared["thickness_nm"].map(lambda values: int(sum(values)))
    prepared["target_mse"] = pd.to_numeric(prepared["target_mse"], errors="coerce")
    prepared["layer_count"] = pd.to_numeric(prepared["layer_count"], errors="coerce").astype("Int64")
    return prepared


def _infer_num_points(frame: pd.DataFrame) -> int:
    lengths = sorted({len(values) for values in frame["spectrum_rt"]})
    if not lengths:
        return 0
    if len(lengths) != 1:
        raise ValueError(f"spectrum_rt rows must have one fixed length, got {lengths}")
    if lengths[0] % 2 != 0:
        raise ValueError("spectrum_rt length must be even because it stores [R..., T...]")
    return lengths[0] // 2


def _split_spectrum_rt(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectrum = np.asarray(values, dtype=np.float32)
    if spectrum.size % 2 != 0:
        raise ValueError("spectrum_rt length must be even")
    half = spectrum.size // 2
    reflection = spectrum[:half]
    transmission = spectrum[half:]
    absorption = 1.0 - reflection - transmission
    return reflection, transmission, absorption


def _target_absorption(row: pd.Series, wavelengths_um: np.ndarray) -> np.ndarray:
    target_id = str(row.get("target_id", ""))
    targets = {target.target_id: target.absorption for target in build_default_targets(wavelengths_um)}
    if target_id in targets:
        return targets[target_id]

    center_um = row.get("target_center_um")
    fwhm_um = row.get("target_fwhm_um")
    if center_um is not None and fwhm_um is not None and not (pd.isna(center_um) or pd.isna(fwhm_um)):
        return lorentzian_profile(wavelengths_um, float(center_um), float(fwhm_um))
    return np.zeros_like(wavelengths_um, dtype=np.float32)


def _build_summary(frame: pd.DataFrame, *, splits: Sequence[str], num_points: int, wavelength_min_um: float, wavelength_max_um: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "record_count": 0,
            "splits": list(splits),
            "target_count": 0,
            "target_family_counts": {},
            "layer_count_counts": {},
            "unique_structure_count": 0,
            "duplicate_structure_count": 0,
            "num_points": int(num_points),
            "wavelength_range_um": [float(wavelength_min_um), float(wavelength_max_um)],
        }
    unique_structure_count = int(frame["structure_key"].nunique())
    return {
        "record_count": int(len(frame)),
        "splits": list(splits),
        "target_count": int(frame["target_id"].nunique()),
        "target_family_counts": {str(key): int(value) for key, value in frame["target_family"].value_counts().items()},
        "layer_count_counts": {str(int(key)): int(value) for key, value in frame["layer_count"].value_counts().sort_index().items()},
        "unique_structure_count": unique_structure_count,
        "duplicate_structure_count": int(len(frame) - unique_structure_count),
        "target_mse": {
            "mean": float(frame["target_mse"].mean()),
            "median": float(frame["target_mse"].median()),
            "min": float(frame["target_mse"].min()),
            "max": float(frame["target_mse"].max()),
        },
        "num_points": int(num_points),
        "wavelength_range_um": [float(wavelength_min_um), float(wavelength_max_um)],
    }


def _make_target_layer_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "target_id",
                "target_family",
                "layer_count",
                "record_count",
                "mse_mean",
                "mse_median",
                "mse_min",
                "mse_max",
                "mse_p90",
                "mse_p99",
            ]
        )
    keys = ["target_id", "target_family", "layer_count"]
    grouped = frame.groupby(keys, dropna=False)
    stats = grouped["target_mse"].agg(
        record_count="count",
        mse_mean="mean",
        mse_median="median",
        mse_min="min",
        mse_max="max",
    )
    stats["mse_p90"] = grouped["target_mse"].quantile(0.90)
    stats["mse_p99"] = grouped["target_mse"].quantile(0.99)
    return stats.reset_index().sort_values(["target_id", "layer_count"])


def _make_material_stats(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    material_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for position, material in enumerate(row["materials"], start=1):
            material_rows.append({"material": material})
            position_rows.append({"position": position, "material": material})
    if not material_rows:
        return pd.DataFrame(columns=["material", "count", "fraction"]), pd.DataFrame(columns=["position", "material", "count"])

    material_counts = pd.DataFrame(material_rows).value_counts("material").reset_index(name="count")
    material_counts["fraction"] = material_counts["count"] / float(material_counts["count"].sum())
    position_counts = pd.DataFrame(position_rows).value_counts(["position", "material"]).reset_index(name="count")
    return material_counts.sort_values(["count", "material"], ascending=[False, True]), position_counts.sort_values(["position", "material"])


def _make_diversity_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "target_id",
                "layer_count",
                "record_count",
                "unique_structure_count",
                "unique_ratio",
                "duplicate_structure_count",
                "total_thickness_mean_nm",
                "total_thickness_min_nm",
                "total_thickness_max_nm",
            ]
        )
    grouped = frame.groupby(["target_id", "layer_count"], dropna=False)
    stats = grouped.agg(
        record_count=("sample_id", "count"),
        unique_structure_count=("structure_key", "nunique"),
        total_thickness_mean_nm=("total_thickness_nm", "mean"),
        total_thickness_min_nm=("total_thickness_nm", "min"),
        total_thickness_max_nm=("total_thickness_nm", "max"),
    ).reset_index()
    stats["unique_ratio"] = stats["unique_structure_count"] / stats["record_count"]
    stats["duplicate_structure_count"] = stats["record_count"] - stats["unique_structure_count"]
    return stats.sort_values(["target_id", "layer_count"])


def _make_best_samples(frame: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "target_id",
                "target_family",
                "layer_count",
                "target_mse",
                "total_thickness_nm",
                "structure_tokens",
            ]
        )
    columns = ["sample_id", "target_id", "target_family", "layer_count", "target_mse", "total_thickness_nm", "structure_tokens"]
    best = frame.sort_values("target_mse").groupby(["target_id", "layer_count"], dropna=False).head(int(top_k))[columns].copy()
    best["structure_tokens"] = best["structure_tokens"].map(lambda values: " ".join(values))
    return best.sort_values(["target_id", "layer_count", "target_mse"])


def _make_search_efficiency(dataset_dir: Path) -> pd.DataFrame:
    summary_path = dataset_dir / "stats" / "search_summary.json"
    if not summary_path.exists():
        return pd.DataFrame(
            columns=[
                "target_id",
                "layer_count",
                "accepted_count",
                "globally_kept_count",
                "global_duplicate_count",
                "shortfall",
                "total_evaluated",
                "duplicate_accepted",
                "restarts_used",
            ]
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    search_rows = payload.get("search", [])
    return pd.DataFrame.from_records(search_rows)


def _plot_placeholder(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_mse_by_target(frame: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "mse_by_target.png"
    if frame.empty:
        _plot_placeholder(path, "MSE by Target", "no records")
        return
    summary = frame.groupby("target_id")["target_mse"].median().sort_values()
    fig_width = max(8, min(28, 0.18 * len(summary)))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(np.arange(len(summary)), summary.values)
    ax.set_title("Median MSE by Target")
    ax.set_xlabel("Target")
    ax.set_ylabel("Median MSE")
    if len(summary) <= 40:
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(summary.index, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_mse_by_layer(frame: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "mse_by_layer_count.png"
    if frame.empty:
        _plot_placeholder(path, "MSE by Layer Count", "no records")
        return
    layers = sorted(int(value) for value in frame["layer_count"].dropna().unique())
    values = [frame.loc[frame["layer_count"] == layer, "target_mse"].dropna().values for layer in layers]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, tick_labels=[str(layer) for layer in layers], showfliers=False)
    ax.set_title("MSE Distribution by Layer Count")
    ax.set_xlabel("Layer Count")
    ax.set_ylabel("Target MSE")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_accepted_count_heatmap(target_layer_stats: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "accepted_count_heatmap.png"
    if target_layer_stats.empty:
        _plot_placeholder(path, "Accepted Count Heatmap", "no records")
        return
    pivot = target_layer_stats.pivot_table(index="target_id", columns="layer_count", values="record_count", fill_value=0)
    fig_height = max(4, min(40, 0.22 * len(pivot)))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title("Accepted Count by Target and Layer Count")
    ax.set_xlabel("Layer Count")
    ax.set_ylabel("Target")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    if len(pivot.index) <= 50:
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
    else:
        ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="records")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_material_frequency(material_stats: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "structures" / "material_frequency.png"
    if material_stats.empty:
        _plot_placeholder(path, "Material Frequency", "no records")
        return
    stats = material_stats.sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(stats)), 5))
    ax.bar(stats["material"], stats["count"])
    ax.set_title("Material Frequency")
    ax.set_xlabel("Material")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_material_by_position(position_stats: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "structures" / "material_by_position_heatmap.png"
    if position_stats.empty:
        _plot_placeholder(path, "Material by Position", "no records")
        return
    pivot = position_stats.pivot_table(index="material", columns="position", values="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(pivot.columns)), max(4, 0.3 * len(pivot.index))))
    image = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_title("Material Count by Layer Position")
    ax.set_xlabel("Layer Position")
    ax.set_ylabel("Material")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, label="count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_thickness_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "structures" / "thickness_distribution.png"
    if frame.empty:
        _plot_placeholder(path, "Thickness Distribution", "no records")
        return
    values = [thickness for row in frame["thickness_nm"] for thickness in row]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(60, max(10, int(math.sqrt(max(1, len(values)))))), color="#2f6f8f", alpha=0.85)
    ax.set_title("Layer Thickness Distribution")
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_total_thickness(frame: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "structures" / "total_thickness_by_target.png"
    if frame.empty:
        _plot_placeholder(path, "Total Thickness by Target", "no records")
        return
    grouped = frame.groupby("target_id")["total_thickness_nm"].median().sort_values()
    fig, ax = plt.subplots(figsize=(max(8, min(28, 0.18 * len(grouped))), 5))
    ax.bar(np.arange(len(grouped)), grouped.values, color="#7a5c27")
    ax.set_title("Median Total Thickness by Target")
    ax.set_xlabel("Target")
    ax.set_ylabel("Median Total Thickness (nm)")
    if len(grouped) <= 40:
        ax.set_xticks(np.arange(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_lorentzian_center_mse(frame: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "figures" / "lorentzian" / "center_vs_best_mse.png"
    lorentz = frame.loc[frame["target_family"] == "lorentzian"].copy()
    if lorentz.empty:
        _plot_placeholder(path, "Lorentzian Center vs Best MSE", "no lorentzian records")
        return
    lorentz["target_center_um"] = pd.to_numeric(lorentz["target_center_um"], errors="coerce")
    stats = lorentz.groupby("target_center_um")["target_mse"].min().dropna().sort_index()
    if stats.empty:
        _plot_placeholder(path, "Lorentzian Center vs Best MSE", "no valid center values")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(stats.index, stats.values, marker="o", linewidth=1.4)
    ax.set_title("Lorentzian Center vs Best MSE")
    ax.set_xlabel("Center Wavelength (um)")
    ax.set_ylabel("Best MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_spectrum_group(
    group: pd.DataFrame,
    *,
    output_dir: Path,
    wavelengths_um: np.ndarray,
    top_k: int,
) -> None:
    sorted_group = group.sort_values("target_mse").head(int(top_k))
    if sorted_group.empty:
        return
    first = sorted_group.iloc[0]
    target_id = str(first["target_id"])
    layer_count = int(first["layer_count"])
    target = _target_absorption(first, wavelengths_um)
    spectra = []
    for _, row in sorted_group.iterrows():
        _, _, absorption = _split_spectrum_rt(row["spectrum_rt"])
        spectra.append(absorption)

    target_dir = output_dir / "figures" / "spectra" / _safe_name(target_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    topk_path = target_dir / f"layer_{layer_count:02d}_topk.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wavelengths_um, target, linestyle="--", color="black", linewidth=2, label="target A")
    for index, absorption in enumerate(spectra, start=1):
        ax.plot(wavelengths_um, absorption, linewidth=1.2, alpha=0.72, label=f"top {index}")
    ax.set_title(f"{target_id} layer {layer_count} top-{len(spectra)} spectra")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Absorption")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(topk_path, dpi=220)
    plt.close(fig)

    spectra_arr = np.stack(spectra, axis=0)
    mean = np.mean(spectra_arr, axis=0)
    std = np.std(spectra_arr, axis=0)
    mean_path = target_dir / f"layer_{layer_count:02d}_mean_band.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wavelengths_um, target, linestyle="--", color="black", linewidth=2, label="target A")
    ax.plot(wavelengths_um, mean, color="#1f6f8b", linewidth=2, label="top-k mean A")
    ax.fill_between(wavelengths_um, mean - std, mean + std, color="#1f6f8b", alpha=0.18, label="top-k std")
    ax.set_title(f"{target_id} layer {layer_count} top-{len(spectra)} mean")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Absorption")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(mean_path, dpi=220)
    plt.close(fig)


def _plot_spectra(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    wavelengths_um: np.ndarray,
    top_k: int,
    max_spectrum_groups: int | None,
) -> None:
    if frame.empty:
        _plot_placeholder(output_dir / "figures" / "spectra" / "empty.png", "Spectra", "no records")
        return
    ranked_groups = (
        frame.groupby(["target_id", "layer_count"], dropna=False)["target_mse"]
        .min()
        .reset_index(name="best_mse")
        .sort_values("best_mse")
    )
    if max_spectrum_groups is not None:
        ranked_groups = ranked_groups.head(max(0, int(max_spectrum_groups)))
    for _, group_row in ranked_groups.iterrows():
        mask = (frame["target_id"] == group_row["target_id"]) & (frame["layer_count"] == group_row["layer_count"])
        _plot_spectrum_group(frame.loc[mask], output_dir=output_dir, wavelengths_um=wavelengths_um, top_k=top_k)


def _write_tables_and_plots(
    frame: pd.DataFrame,
    *,
    dataset_dir: Path,
    output_dir: Path,
    wavelengths_um: np.ndarray,
    top_k: int,
    max_spectrum_groups: int | None,
) -> dict[str, Any]:
    target_layer_stats = _make_target_layer_stats(frame)
    material_stats, position_stats = _make_material_stats(frame)
    diversity_stats = _make_diversity_stats(frame)
    best_samples = _make_best_samples(frame, top_k=top_k)
    search_efficiency = _make_search_efficiency(dataset_dir)

    _write_csv(target_layer_stats, output_dir / "tables" / "target_layer_stats.csv")
    _write_csv(material_stats, output_dir / "tables" / "material_stats.csv")
    _write_csv(position_stats, output_dir / "tables" / "material_position_stats.csv")
    _write_csv(diversity_stats, output_dir / "tables" / "diversity_stats.csv")
    _write_csv(best_samples, output_dir / "tables" / "best_samples.csv")
    _write_csv(search_efficiency, output_dir / "tables" / "search_efficiency.csv")

    _plot_mse_by_target(frame, output_dir)
    _plot_mse_by_layer(frame, output_dir)
    _plot_accepted_count_heatmap(target_layer_stats, output_dir)
    _plot_material_frequency(material_stats, output_dir)
    _plot_material_by_position(position_stats, output_dir)
    _plot_thickness_distribution(frame, output_dir)
    _plot_total_thickness(frame, output_dir)
    _plot_lorentzian_center_mse(frame, output_dir)
    _plot_spectra(frame, output_dir=output_dir, wavelengths_um=wavelengths_um, top_k=top_k, max_spectrum_groups=max_spectrum_groups)

    return {
        "target_layer_rows": int(len(target_layer_stats)),
        "material_rows": int(len(material_stats)),
        "diversity_rows": int(len(diversity_stats)),
        "best_sample_rows": int(len(best_samples)),
        "search_efficiency_rows": int(len(search_efficiency)),
    }


def analyze_pso_dataset(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    splits: Sequence[str] | None = None,
    wavelength_min_um: float = 2.0,
    wavelength_max_um: float = 15.0,
    top_k: int = 8,
    max_spectrum_groups: int | None = 100,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    analysis_dir = Path(output_dir) if output_dir is not None else dataset_path / "analysis" / "pso"
    requested_splits = list(splits or ["all"])
    frame = _prepare_frame(_load_records(dataset_path, requested_splits))
    num_points = _infer_num_points(frame) if not frame.empty else 0
    wavelengths_um = np.linspace(float(wavelength_min_um), float(wavelength_max_um), max(1, num_points), dtype=np.float32)

    summary = _build_summary(
        frame,
        splits=requested_splits,
        num_points=num_points,
        wavelength_min_um=wavelength_min_um,
        wavelength_max_um=wavelength_max_um,
    )
    artifact_counts = _write_tables_and_plots(
        frame,
        dataset_dir=dataset_path,
        output_dir=analysis_dir,
        wavelengths_um=wavelengths_um,
        top_k=max(1, int(top_k)),
        max_spectrum_groups=max_spectrum_groups,
    )
    summary["artifacts"] = artifact_counts

    _write_json(analysis_dir / "summary.json", summary)
    _write_json(
        analysis_dir / "analysis_manifest.json",
        {
            "dataset_dir": str(dataset_path),
            "output_dir": str(analysis_dir),
            "splits": requested_splits,
            "wavelength_range_um": [float(wavelength_min_um), float(wavelength_max_um)],
            "top_k": int(top_k),
            "max_spectrum_groups": max_spectrum_groups,
            "tables": [
                "tables/target_layer_stats.csv",
                "tables/search_efficiency.csv",
                "tables/material_stats.csv",
                "tables/material_position_stats.csv",
                "tables/diversity_stats.csv",
                "tables/best_samples.csv",
            ],
            "figures": [
                "figures/mse_by_target.png",
                "figures/mse_by_layer_count.png",
                "figures/accepted_count_heatmap.png",
                "figures/structures/material_frequency.png",
                "figures/structures/material_by_position_heatmap.png",
                "figures/structures/thickness_distribution.png",
                "figures/structures/total_thickness_by_target.png",
                "figures/lorentzian/center_vs_best_mse.png",
                "figures/spectra/<target_id>/layer_<layer_count>_topk.png",
                "figures/spectra/<target_id>/layer_<layer_count>_mean_band.png",
            ],
        },
    )
    return summary
