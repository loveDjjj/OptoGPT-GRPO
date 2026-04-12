from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .io import (
    derive_materials_and_thicknesses,
    iter_record_batches,
    load_vocab_tokens,
    resolve_analysis_scopes,
    resolve_custom_scope,
)
from .structure_analysis import analyze_structure_distribution
from .spectrum_analysis import analyze_spectrum_distribution


def analyze_dataset(
    *,
    dataset_dir: str | Path | None = None,
    shard_paths: Sequence[str | Path] | None = None,
    split: str = "all",
    scopes: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 4096,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    pca_components: int = 8,
    cluster_count: int = 16,
    cluster_fit_samples: int = 50000,
    cluster_iterations: int = 20,
    scatter_max_points: int = 20000,
    device: str = "auto",
    enable_structure_analysis: bool = True,
    enable_spectrum_analysis: bool = True,
) -> dict:
    if dataset_dir is None and not shard_paths:
        raise ValueError("Either dataset_dir or shard_paths must be provided")
    if dataset_dir is not None and shard_paths:
        raise ValueError("dataset_dir and shard_paths cannot be used together")

    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)
        resolved_scopes = resolve_analysis_scopes(dataset_dir, scopes or [split])
        try:
            tokens = load_vocab_tokens(dataset_dir)
            material_names, thickness_values_nm = derive_materials_and_thicknesses(tokens)
        except Exception:
            material_names, thickness_values_nm = [], []
        analysis_root = Path(output_dir) if output_dir is not None else dataset_dir / "analysis"
    else:
        resolved_scopes = resolve_custom_scope(shard_paths or [])
        material_names, thickness_values_nm = [], []
        analysis_root = Path(output_dir or "analysis")

    analysis_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    for scope_name, scope_shards in resolved_scopes.items():
        scope_output_dir = analysis_root / scope_name
        scope_output_dir.mkdir(parents=True, exist_ok=True)
        scope_summary: dict[str, dict] = {}

        if enable_structure_analysis:
            scope_summary["structure"] = analyze_structure_distribution(
                scope_name=scope_name,
                batches=iter_record_batches(
                    shard_paths=scope_shards,
                    batch_size=int(batch_size),
                    columns=["sample_id", "layer_count", "materials", "thickness_nm"],
                ),
                material_names=material_names,
                thickness_values_nm=thickness_values_nm,
                output_dir=scope_output_dir,
            )
        if enable_spectrum_analysis:
            batch_factory = lambda: iter_record_batches(
                shard_paths=scope_shards,
                batch_size=int(batch_size),
                columns=["sample_id", "layer_count", "materials", "spectrum_rt"],
            )
            scope_summary["spectrum"] = analyze_spectrum_distribution(
                scope_name=scope_name,
                batch_factory=batch_factory,
                output_dir=scope_output_dir,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                pca_components=int(pca_components),
                cluster_count=int(cluster_count),
                cluster_fit_samples=int(cluster_fit_samples),
                cluster_iterations=int(cluster_iterations),
                scatter_max_points=int(scatter_max_points),
                device=device,
            )
        (scope_output_dir / "analysis_summary.json").write_text(
            json.dumps(scope_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries[scope_name] = scope_summary

    (analysis_root / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "scopes": list(resolved_scopes.keys()),
                "output_dir": str(analysis_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summaries
