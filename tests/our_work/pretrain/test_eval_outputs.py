from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work.pretrain.eval_outputs import (
    build_summary_payload,
    create_eval_run_dir,
    write_results_jsonl,
)


def test_write_results_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    rows = [
        {"sample_id": "a", "target_layer_count": 5, "generated_valid": True},
        {"sample_id": "b", "target_layer_count": 6, "generated_valid": False},
    ]
    output_path = tmp_path / "results.jsonl"

    write_results_jsonl(rows, output_path)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sample_id"] == "a"
    assert json.loads(lines[1])["sample_id"] == "b"


def test_build_summary_payload_includes_global_and_per_layer_metrics() -> None:
    rows = [
        {
            "sample_id": "a",
            "target_layer_count": 5,
            "generated_valid": True,
            "token_exact_match": True,
            "spectrum_rmse": 0.1,
            "spectrum_mae": 0.05,
        },
        {
            "sample_id": "b",
            "target_layer_count": 5,
            "generated_valid": False,
            "token_exact_match": False,
            "spectrum_rmse": None,
            "spectrum_mae": None,
        },
        {
            "sample_id": "c",
            "target_layer_count": 6,
            "generated_valid": True,
            "token_exact_match": False,
            "spectrum_rmse": 0.3,
            "spectrum_mae": 0.2,
        },
    ]

    payload = build_summary_payload(
        rows,
        {"split": "val"},
        {"summary": "summary.json"},
        {"rmse_hist": "not enough valid rows"},
    )

    assert payload["metadata"]["split"] == "val"
    assert payload["global_metrics"]["sample_count"] == 3
    assert payload["global_metrics"]["valid_generation_count"] == 2
    assert payload["global_metrics"]["valid_generation_rate"] == 2 / 3
    assert payload["global_metrics"]["exact_match_count"] == 1
    assert payload["global_metrics"]["exact_match_rate"] == 1 / 3
    assert payload["global_metrics"]["mean_spectrum_rmse"] == 0.2
    assert payload["global_metrics"]["mean_spectrum_mae"] == 0.125
    assert payload["per_target_layer_count"]["5"]["sample_count"] == 2
    assert payload["per_target_layer_count"]["5"]["valid_generation_count"] == 1
    assert payload["per_target_layer_count"]["5"]["mean_spectrum_rmse"] == 0.1
    assert payload["per_target_layer_count"]["6"]["mean_spectrum_rmse"] == 0.3
    assert payload["artifacts"]["summary"] == "summary.json"
    assert payload["skipped_artifacts"]["rmse_hist"] == "not enough valid rows"


def test_create_eval_run_dir_creates_timestamped_subdirectories(tmp_path: Path) -> None:
    run_dir = create_eval_run_dir(tmp_path, run_name="base_run", timestamp="20260411-120000")

    assert run_dir.name == "20260411-120000"
    assert run_dir.parent.name == "eval_runs"
    assert (run_dir / "plots").exists()
    assert (run_dir / "samples").exists()
