from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from our_work.pretrain.eval_outputs import create_eval_run_dir
from our_work.pretrain.scripts.run_eval import (
    _write_eval_artifacts,
    evaluate_records,
    load_eval_components,
    resolve_num_points,
)


def load_records_from_shard(shard_path: Path) -> list[dict]:
    table = pq.read_table(
        shard_path,
        columns=["sample_id", "layer_count", "structure_tokens", "spectrum_rt"],
    )
    rows = table.to_pylist()
    records: list[dict] = []
    for row in rows:
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "layer_count": int(row["layer_count"]),
                "structure_tokens": list(row["structure_tokens"]),
                "spectrum_rt": list(row["spectrum_rt"]),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on each shard separately with separate visualization outputs.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--database-dir", default="our_work/_shared/database")
    parser.add_argument("--output-root", default="outputs/our_work/eval/ga_custom_tasks_shards")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--wavelength-min", type=float, default=2.0)
    parser.add_argument("--wavelength-max", type=float, default=15.0)
    parser.add_argument("--incident-angle", type=float, default=0.0)
    parser.add_argument("--polarization", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--complex-dtype", default="complex128")
    parser.add_argument("--device", default=None)
    parser.add_argument("--worst-sample-plots", type=int, default=8)
    parser.add_argument("--random-sample-plots", type=int, default=8)
    parser.add_argument("--disable-progress", action="store_true")
    args = parser.parse_args()

    model, tokenizer, device = load_eval_components(args.checkpoint_dir, device=args.device)
    num_points = resolve_num_points(model, requested_num_points=None)

    shards_dir = Path(args.shards_dir)
    shard_paths = sorted(shards_dir.glob("shard-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard-*.parquet found under: {shards_dir}")

    output_root = Path(args.output_root)
    summary_rows: list[dict] = []

    for shard_path in shard_paths:
        records = load_records_from_shard(shard_path)
        run_dir = create_eval_run_dir(output_root, run_name=f"{Path(args.checkpoint_dir).name}_{shard_path.stem}")

        results = evaluate_records(
            model=model,
            tokenizer=tokenizer,
            records=records,
            database_path=args.database_dir,
            wavelength_range_um=(args.wavelength_min, args.wavelength_max),
            num_points=num_points,
            incident_angle=args.incident_angle,
            polarization=args.polarization,
            tolerance=args.tolerance,
            complex_dtype=args.complex_dtype,
            max_new_tokens=args.max_new_tokens,
            device=device,
            show_progress=not args.disable_progress,
        )

        metadata = {
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_shard": str(shard_path),
            "database_dir": str(args.database_dir),
            "sample_count": len(records),
            "num_points": num_points,
            "wavelength_range_um": [args.wavelength_min, args.wavelength_max],
            "incident_angle": args.incident_angle,
            "polarization": args.polarization,
            "tolerance": args.tolerance,
            "complex_dtype": args.complex_dtype,
        }

        summary = _write_eval_artifacts(
            run_dir=run_dir,
            rows=results,
            metadata=metadata,
            num_points=num_points,
            worst_sample_plots=args.worst_sample_plots,
            random_sample_plots=args.random_sample_plots,
            disable_plots=False,
        )

        summary_rows.append(
            {
                "shard": shard_path.name,
                "run_dir": str(run_dir),
                "global_metrics": summary.get("global_metrics", {}),
            }
        )
        print(f"[done] {shard_path.name} -> {run_dir}")

    combined = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "shards_dir": str(shards_dir),
        "runs": summary_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    combined_path = output_root / "combined_summary.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {combined_path}")


if __name__ == "__main__":
    main()
