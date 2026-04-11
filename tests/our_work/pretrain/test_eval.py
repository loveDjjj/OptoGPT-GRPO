from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from our_work.pretrain.scripts.run_eval import (
    evaluate_token_prediction,
    main,
    resolve_repo_path,
)


def test_evaluate_token_prediction_returns_zero_rmse_for_matching_tokens(tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    (database_dir / "SiO2.csv").write_text("wl,n,k\n2.0,1.4,0.0\n15.0,1.4,0.0\n", encoding="utf-8")
    (database_dir / "Ge.csv").write_text("wl,n,k\n2.0,4.0,0.1\n15.0,4.0,0.1\n", encoding="utf-8")

    _, reflections, transmissions, ok_mask = simulate_structure_batch(
        [["Ge_10", "SiO2_20"]],
        database_path=str(database_dir),
        wavelength_range_um=(2.0, 15.0),
        num_points=16,
        incident_angle=0.0,
        polarization=0,
        tolerance=1e-3,
        complex_dtype="complex128",
    )
    assert ok_mask.tolist() == [True]
    record = {
        "sample_id": "sample-000",
        "layer_count": 2,
        "structure_tokens": ["Ge_10", "SiO2_20"],
        "spectrum_rt": flatten_rt_spectrum(reflections[0], transmissions[0]).tolist(),
    }

    result = evaluate_token_prediction(
        record=record,
        predicted_tokens=["Ge_10", "SiO2_20"],
        database_path=str(database_dir),
        wavelength_range_um=(2.0, 15.0),
        num_points=16,
        incident_angle=0.0,
        polarization=0,
        tolerance=1e-3,
        complex_dtype="complex128",
    )

    assert result["generated_valid"] is True
    assert result["token_exact_match"] is True
    assert result["prediction_layer_count"] == 2
    assert np.isclose(result["spectrum_rmse"], 0.0)


def test_evaluate_token_prediction_marks_invalid_generated_tokens(tmp_path: Path):
    record = {
        "sample_id": "sample-001",
        "layer_count": 1,
        "structure_tokens": ["Ge_10"],
        "spectrum_rt": [0.1] * 32,
    }

    result = evaluate_token_prediction(
        record=record,
        predicted_tokens=["[UNK]"],
        database_path=str(tmp_path),
        wavelength_range_um=(2.0, 15.0),
        num_points=16,
        incident_angle=0.0,
        polarization=0,
        tolerance=1e-3,
        complex_dtype="complex128",
    )

    assert result["generated_valid"] is False
    assert result["token_exact_match"] is False
    assert result["spectrum_rmse"] is None


def test_resolve_repo_path_finds_parent_relative_path(tmp_path: Path):
    repo_root = tmp_path / "repo"
    worktree_root = repo_root / ".worktrees" / "feat-x"
    database_dir = repo_root / "database"
    database_dir.mkdir(parents=True)
    worktree_root.mkdir(parents=True)

    resolved = resolve_repo_path("database", project_root=worktree_root)
    assert resolved == database_dir


def test_run_eval_main_writes_summary_results_and_plots(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "outputs" / "base_run" / "checkpoint-7"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")

    rows = [
        {
            "sample_id": "sample-000",
            "target_layer_count": 1,
            "prediction_layer_count": 1,
            "target_tokens": ["Ge_10"],
            "predicted_tokens": ["Ge_10"],
            "token_exact_match": True,
            "generated_valid": True,
            "spectrum_rmse": 0.0,
            "spectrum_mae": 0.0,
            "target_spectrum_rt": [0.1] * 32,
            "predicted_spectrum_rt": [0.1] * 32,
        },
        {
            "sample_id": "sample-001",
            "target_layer_count": 2,
            "prediction_layer_count": 1,
            "target_tokens": ["Ge_10", "SiO2_20"],
            "predicted_tokens": ["[UNK]"],
            "token_exact_match": False,
            "generated_valid": False,
            "spectrum_rmse": None,
            "spectrum_mae": None,
            "target_spectrum_rt": [0.2] * 32,
            "predicted_spectrum_rt": None,
        },
    ]
    records = [
        {
            "sample_id": "sample-000",
            "layer_count": 1,
            "structure_tokens": ["Ge_10"],
            "spectrum_rt": [0.1] * 32,
        },
        {
            "sample_id": "sample-001",
            "layer_count": 2,
            "structure_tokens": ["Ge_10", "SiO2_20"],
            "spectrum_rt": [0.2] * 32,
        },
    ]
    output_dir = tmp_path / "eval-output"

    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.load_eval_components",
        lambda *args, **kwargs: (object(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.load_split_records",
        lambda *args, **kwargs: records,
    )
    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.evaluate_records",
        lambda **kwargs: [dict(row) for row in rows],
    )

    payload = main(
        [
            "--checkpoint-dir",
            str(checkpoint_dir.parent),
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--database-dir",
            str(tmp_path / "database"),
            "--split",
            "val",
            "--max-samples",
            "2",
            "--num-points",
            "16",
            "--output-dir",
            str(output_dir),
            "--worst-sample-plots",
            "1",
            "--random-sample-plots",
            "1",
        ]
    )

    run_dirs = list((output_dir / "base_run" / "eval_runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert payload["run_dir"] == str(run_dir)
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "plots" / "rmse_hist.png").exists()
    assert (run_dir / "plots" / "mae_hist.png").exists()
    assert list((run_dir / "samples").glob("worst-*.png"))
    assert not list((run_dir / "samples").glob("random-*.png"))

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["global_metrics"]["sample_count"] == 2
    assert summary["global_metrics"]["valid_generation_count"] == 1
    assert summary["global_metrics"]["mean_spectrum_rmse"] == 0.0
    assert "random_samples" in summary["skipped_artifacts"]

    result_lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(result_lines) == 2
    parsed_rows = [json.loads(line) for line in result_lines]
    assert parsed_rows[0]["target_spectrum_rt"] == [0.1] * 32
    assert parsed_rows[0]["predicted_spectrum_rt"] == [0.1] * 32
    assert parsed_rows[1]["predicted_spectrum_rt"] is None
