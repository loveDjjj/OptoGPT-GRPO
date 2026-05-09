from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _shared.io.config import load_yaml_config
from _shared.io.config import resolve_repo_path
from data_gen.analysis import analyze_dataset


def _resolve_value(cli_value, yaml_value, default_value):
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default_value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze generated dataset structure and spectrum distributions.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir")
    source.add_argument("--shard-path", action="append", default=[])
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="all")
    parser.add_argument("--scope", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--wavelength-min", type=float, default=None)
    parser.add_argument("--wavelength-max", type=float, default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--pca-fit-samples", type=int, default=None)
    parser.add_argument("--cluster-count", type=int, default=None)
    parser.add_argument("--cluster-fit-samples", type=int, default=None)
    parser.add_argument("--cluster-iterations", type=int, default=None)
    parser.add_argument("--scatter-max-points", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cluster-mode", default=None, choices=["fixed_k", "auto_k"])
    parser.add_argument("--k-candidate", action="append", type=int, default=None)
    parser.add_argument("--selection-strategy", default=None, choices=["weighted_rank", "single_metric"])
    parser.add_argument("--primary-metric", default=None, choices=["silhouette", "calinski_harabasz", "davies_bouldin"])
    parser.add_argument("--metric-sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--top-material-count", type=int, default=None)
    parser.add_argument("--max-thickness-ticks", type=int, default=None)
    parser.add_argument("--disable-structure-analysis", action="store_true")
    parser.add_argument("--disable-spectrum-analysis", action="store_true")
    args = parser.parse_args(argv)

    yaml_payload = {}
    if args.config:
        config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
        yaml_payload = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=False)
    analysis_cfg = yaml_payload.get("analysis", {})
    structure_cfg = analysis_cfg.get("structure", {})
    spectrum_cfg = analysis_cfg.get("spectrum", {})
    pca_cfg = spectrum_cfg.get("pca", {})
    clustering_cfg = spectrum_cfg.get("clustering", {})
    selection_cfg = clustering_cfg.get("selection", {})
    visualization_cfg = spectrum_cfg.get("visualization", {})

    dataset_dir = resolve_repo_path(args.dataset_dir, project_root=PROJECT_ROOT) if args.dataset_dir else None
    shard_paths = [resolve_repo_path(path, project_root=PROJECT_ROOT) for path in (args.shard_path or [])]
    output_dir = resolve_repo_path(args.output_dir, project_root=PROJECT_ROOT) if args.output_dir else None
    analyze_dataset(
        dataset_dir=dataset_dir,
        shard_paths=shard_paths,
        split=args.split,
        scopes=args.scope or None,
        output_dir=output_dir,
        batch_size=_resolve_value(args.batch_size, analysis_cfg.get("batch_size"), 4096),
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        engine=_resolve_value(args.engine, spectrum_cfg.get("engine"), "rapids"),
        pca_components=_resolve_value(args.pca_components, pca_cfg.get("components"), 8),
        pca_fit_samples=_resolve_value(args.pca_fit_samples, pca_cfg.get("fit_samples"), 50000),
        cluster_count=_resolve_value(args.cluster_count, clustering_cfg.get("fixed_k"), 16),
        cluster_fit_samples=_resolve_value(args.cluster_fit_samples, clustering_cfg.get("fit_samples"), 50000),
        cluster_iterations=_resolve_value(args.cluster_iterations, clustering_cfg.get("max_iter"), 20),
        scatter_max_points=_resolve_value(args.scatter_max_points, visualization_cfg.get("scatter_max_points"), 20000),
        device=_resolve_value(args.device, spectrum_cfg.get("device"), "auto"),
        enable_structure_analysis=(not args.disable_structure_analysis)
        and bool(_resolve_value(None, structure_cfg.get("enabled"), True)),
        enable_spectrum_analysis=(not args.disable_spectrum_analysis)
        and bool(_resolve_value(None, spectrum_cfg.get("enabled"), True)),
        structure_top_material_count=_resolve_value(args.top_material_count, structure_cfg.get("top_material_count"), 20),
        structure_max_thickness_ticks=_resolve_value(args.max_thickness_ticks, structure_cfg.get("max_thickness_ticks"), 20),
        cluster_mode=_resolve_value(args.cluster_mode, clustering_cfg.get("mode"), "fixed_k"),
        k_candidates=_resolve_value(args.k_candidate, clustering_cfg.get("k_candidates"), []),
        selection_strategy=_resolve_value(args.selection_strategy, selection_cfg.get("strategy"), "weighted_rank"),
        primary_metric=_resolve_value(args.primary_metric, selection_cfg.get("primary_metric"), "silhouette"),
        metric_sample_size=_resolve_value(args.metric_sample_size, selection_cfg.get("metric_sample_size"), 15000),
        random_state=_resolve_value(args.random_state, clustering_cfg.get("random_state"), 42),
        n_init=_resolve_value(args.n_init, clustering_cfg.get("n_init"), 1),
    )


if __name__ == "__main__":
    main()
