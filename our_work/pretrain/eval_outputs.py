from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def create_eval_run_dir(output_root: str | Path, run_name: str, timestamp: str | None = None) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / run_name / "eval_runs" / stamp
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_results_jsonl(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _mean_or_none(values: list[float | int | None]) -> float | None:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return float(np.mean(clean_values))


def _build_metrics(rows: list[dict]) -> dict[str, float | int | None]:
    sample_count = len(rows)
    valid_rows = [row for row in rows if row.get("generated_valid")]
    valid_count = len(valid_rows)
    exact_match_count = sum(1 for row in rows if row.get("token_exact_match"))
    return {
        "sample_count": sample_count,
        "valid_generation_count": valid_count,
        "valid_generation_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "exact_match_count": exact_match_count,
        "exact_match_rate": float(exact_match_count / sample_count) if sample_count else 0.0,
        "mean_spectrum_rmse": _mean_or_none([row.get("spectrum_rmse") for row in valid_rows]),
        "mean_spectrum_mae": _mean_or_none([row.get("spectrum_mae") for row in valid_rows]),
    }


def build_summary_payload(
    rows: list[dict],
    metadata: dict,
    artifacts: dict[str, str],
    skipped_artifacts: dict[str, str],
) -> dict:
    per_target_layer_count: dict[str, dict[str, float | int | None]] = {}
    layer_counts = sorted({int(row["target_layer_count"]) for row in rows})
    for layer_count in layer_counts:
        layer_rows = [row for row in rows if int(row["target_layer_count"]) == layer_count]
        per_target_layer_count[str(layer_count)] = _build_metrics(layer_rows)

    return {
        "metadata": metadata,
        "global_metrics": _build_metrics(rows),
        "per_target_layer_count": per_target_layer_count,
        "artifacts": artifacts,
        "skipped_artifacts": skipped_artifacts,
    }
