from __future__ import annotations

import argparse

from .pipeline import analyze_pso_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze and visualize a PSO supplement dataset.")
    parser.add_argument("--dataset-dir", required=True, help="PSO dataset directory containing shards/ and splits/.")
    parser.add_argument("--output-dir", default=None, help="Analysis output directory. Defaults to <dataset-dir>/analysis/pso.")
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        default=None,
        help="Split to analyze. Repeat for multiple splits. Use 'all' to combine train/val/test.",
    )
    parser.add_argument("--wavelength-min-um", type=float, default=2.0, help="Minimum wavelength used to reconstruct plots.")
    parser.add_argument("--wavelength-max-um", type=float, default=15.0, help="Maximum wavelength used to reconstruct plots.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of best samples drawn per target/layer spectrum group.")
    parser.add_argument(
        "--max-spectrum-groups",
        type=int,
        default=100,
        help="Maximum target/layer groups to plot for spectra. Use -1 to plot every group.",
    )
    args = parser.parse_args(argv)

    max_spectrum_groups = None if int(args.max_spectrum_groups) < 0 else int(args.max_spectrum_groups)
    analyze_pso_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        splits=args.splits or ["all"],
        wavelength_min_um=args.wavelength_min_um,
        wavelength_max_um=args.wavelength_max_um,
        top_k=args.top_k,
        max_spectrum_groups=max_spectrum_groups,
    )


if __name__ == "__main__":
    main()
