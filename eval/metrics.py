from __future__ import annotations

import math
from statistics import mean, median


def _finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    alpha = position - lower
    return float((1 - alpha) * ordered[lower] + alpha * ordered[upper])


def summarize_rows(rows: list[dict]) -> dict:
    sample_count = len(rows)
    valid_rows = [row for row in rows if row.get("generated_valid")]
    exact_match_count = sum(1 for row in rows if row.get("token_exact_match"))
    rmse_values = _finite([row.get("spectrum_rmse") for row in valid_rows])
    mae_values = _finite([row.get("spectrum_mae") for row in valid_rows])

    per_target_layer_count: dict[str, dict] = {}
    for layer_count in sorted({int(row["target_layer_count"]) for row in rows}):
        layer_rows = [row for row in rows if int(row["target_layer_count"]) == layer_count]
        layer_valid_rows = [row for row in layer_rows if row.get("generated_valid")]
        layer_rmse = _finite([row.get("spectrum_rmse") for row in layer_valid_rows])
        layer_mae = _finite([row.get("spectrum_mae") for row in layer_valid_rows])
        layer_exact = sum(1 for row in layer_rows if row.get("token_exact_match"))
        per_target_layer_count[str(layer_count)] = {
            "sample_count": len(layer_rows),
            "valid_generation_count": len(layer_valid_rows),
            "valid_generation_rate": float(len(layer_valid_rows) / len(layer_rows)) if layer_rows else 0.0,
            "exact_match_count": layer_exact,
            "exact_match_rate": float(layer_exact / len(layer_rows)) if layer_rows else 0.0,
            "mean_spectrum_rmse": float(mean(layer_rmse)) if layer_rmse else None,
            "mean_spectrum_mae": float(mean(layer_mae)) if layer_mae else None,
        }

    return {
        "sample_count": sample_count,
        "valid_generation_count": len(valid_rows),
        "valid_generation_rate": float(len(valid_rows) / sample_count) if sample_count else 0.0,
        "exact_match_count": exact_match_count,
        "exact_match_rate": float(exact_match_count / sample_count) if sample_count else 0.0,
        "mean_spectrum_rmse": float(mean(rmse_values)) if rmse_values else None,
        "median_spectrum_rmse": float(median(rmse_values)) if rmse_values else None,
        "p90_spectrum_rmse": _percentile(rmse_values, 0.9),
        "mean_spectrum_mae": float(mean(mae_values)) if mae_values else None,
        "median_spectrum_mae": float(median(mae_values)) if mae_values else None,
        "p90_spectrum_mae": _percentile(mae_values, 0.9),
        "per_target_layer_count": per_target_layer_count,
    }


def select_plot_rows(rows: list[dict], *, worst_count: int, best_count: int, mean_count: int) -> dict[str, list[dict]]:
    valid_rows = [
        row
        for row in rows
        if row.get("generated_valid") and row.get("spectrum_rmse") is not None and math.isfinite(float(row["spectrum_rmse"]))
    ]
    ordered = sorted(valid_rows, key=lambda row: float(row["spectrum_rmse"]))
    used_ids: set[str] = set()

    def _take(sequence: list[dict], count: int) -> list[dict]:
        selected: list[dict] = []
        for row in sequence:
            sample_id = str(row["sample_id"])
            if sample_id in used_ids:
                continue
            selected.append(row)
            used_ids.add(sample_id)
            if len(selected) >= max(0, int(count)):
                break
        return selected

    best_rows = _take(ordered, best_count)
    worst_rows = _take(list(reversed(ordered)), worst_count)
    mean_rows: list[dict] = []
    if ordered and int(mean_count) > 0:
        target_mean = float(mean(float(row["spectrum_rmse"]) for row in ordered))
        mean_candidates = sorted(ordered, key=lambda row: abs(float(row["spectrum_rmse"]) - target_mean))
        mean_rows = _take(mean_candidates, mean_count)
    return {"best": best_rows, "worst": worst_rows, "mean": mean_rows}


def build_overall_summary(split_summaries: dict[str, dict]) -> dict:
    comparison = {}
    for metric_name in (
        "sample_count",
        "valid_generation_rate",
        "exact_match_rate",
        "mean_spectrum_rmse",
        "mean_spectrum_mae",
    ):
        comparison[metric_name] = {split_name: summary.get(metric_name) for split_name, summary in split_summaries.items()}
    return {"split_summaries": split_summaries, "comparison": comparison}
