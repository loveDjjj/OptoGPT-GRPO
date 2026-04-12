from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work._shared.io.config import resolve_repo_path
from our_work.data_gen.analysis import analyze_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze generated our_work dataset structure and spectrum distributions.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir")
    source.add_argument("--shard-path", action="append", default=[])
    parser.add_argument("--split", default="all")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--wavelength-min", type=float, default=None)
    parser.add_argument("--wavelength-max", type=float, default=None)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--cluster-count", type=int, default=16)
    parser.add_argument("--cluster-fit-samples", type=int, default=50000)
    parser.add_argument("--cluster-iterations", type=int, default=20)
    parser.add_argument("--scatter-max-points", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-structure-analysis", action="store_true")
    parser.add_argument("--disable-spectrum-analysis", action="store_true")
    args = parser.parse_args(argv)

    dataset_dir = resolve_repo_path(args.dataset_dir, project_root=PROJECT_ROOT) if args.dataset_dir else None
    shard_paths = [resolve_repo_path(path, project_root=PROJECT_ROOT) for path in (args.shard_path or [])]
    output_dir = resolve_repo_path(args.output_dir, project_root=PROJECT_ROOT) if args.output_dir else None
    analyze_dataset(
        dataset_dir=dataset_dir,
        shard_paths=shard_paths,
        split=args.split,
        output_dir=output_dir,
        batch_size=args.batch_size,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        pca_components=args.pca_components,
        cluster_count=args.cluster_count,
        cluster_fit_samples=args.cluster_fit_samples,
        cluster_iterations=args.cluster_iterations,
        scatter_max_points=args.scatter_max_points,
        device=args.device,
        enable_structure_analysis=not args.disable_structure_analysis,
        enable_spectrum_analysis=not args.disable_spectrum_analysis,
    )


if __name__ == "__main__":
    main()
