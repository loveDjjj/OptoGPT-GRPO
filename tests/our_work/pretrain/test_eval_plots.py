from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work.pretrain.eval_plots import (
    plot_metric_histogram,
    plot_sample_spectrum,
    select_sample_plot_rows,
)


def test_select_sample_plot_rows_returns_worst_rows_plus_random_rows_without_overlap() -> None:
    rows = [
        {"sample_id": "a", "generated_valid": True, "spectrum_rmse": 0.1},
        {"sample_id": "b", "generated_valid": True, "spectrum_rmse": 0.4},
        {"sample_id": "c", "generated_valid": True, "spectrum_rmse": 0.3},
        {"sample_id": "d", "generated_valid": True, "spectrum_rmse": 0.2},
        {"sample_id": "e", "generated_valid": False, "spectrum_rmse": None},
    ]

    selected = select_sample_plot_rows(rows, worst_count=2, random_count=2, seed=7)

    worst_ids = [row["sample_id"] for row in selected["worst"]]
    random_ids = [row["sample_id"] for row in selected["random"]]
    assert worst_ids == ["b", "c"]
    assert len(selected["worst"]) == 2
    assert len(selected["random"]) == 2
    assert set(worst_ids).isdisjoint(random_ids)
    assert len(set(worst_ids + random_ids)) == 4
    assert "e" not in worst_ids + random_ids


def test_plot_metric_histogram_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "rmse_hist.png"

    plot_metric_histogram(
        values=[0.1, 0.2, 0.4],
        title="RMSE",
        xlabel="rmse",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_sample_spectrum_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "sample.png"
    row = {
        "sample_id": "sample-1",
        "target_layer_count": 5,
        "prediction_layer_count": 6,
        "token_exact_match": False,
        "spectrum_rmse": 0.2,
        "target_spectrum_rt": np.linspace(0.1, 0.9, 8).tolist(),
        "predicted_spectrum_rt": np.linspace(0.2, 0.8, 8).tolist(),
    }

    plot_sample_spectrum(row=row, output_path=output_path, num_points=4)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
