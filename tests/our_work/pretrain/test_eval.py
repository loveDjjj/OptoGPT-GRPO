from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from our_work.pretrain.scripts.run_eval import (
    evaluate_token_prediction,
    evaluate_records,
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


def test_evaluate_token_prediction_propagates_simulator_system_exceptions(tmp_path: Path, monkeypatch):
    record = {
        "sample_id": "sample-err",
        "layer_count": 1,
        "structure_tokens": ["Ge_10"],
        "spectrum_rt": [0.1] * 32,
    }
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    (database_dir / "Ge.csv").write_text("wl,n,k\n2.0,4.0,0.1\n15.0,4.0,0.1\n", encoding="utf-8")

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("simulator backend unavailable")

    monkeypatch.setattr("our_work.pretrain.scripts.run_eval.simulate_structure_batch", raise_runtime_error)

    try:
        evaluate_token_prediction(
            record=record,
            predicted_tokens=["Ge_10"],
            database_path=str(database_dir),
            wavelength_range_um=(2.0, 15.0),
            num_points=16,
            incident_angle=0.0,
            polarization=0,
            tolerance=1e-3,
            complex_dtype="complex128",
        )
    except RuntimeError as exc:
        assert "backend unavailable" in str(exc)
    else:
        raise AssertionError("simulator/config failures should be propagated")


def test_evaluate_token_prediction_rejects_bad_database_path_as_config_error(tmp_path: Path):
    record = {
        "sample_id": "sample-bad-db",
        "layer_count": 1,
        "structure_tokens": ["Ge_10"],
        "spectrum_rt": [0.1] * 32,
    }

    try:
        evaluate_token_prediction(
            record=record,
            predicted_tokens=["Ge_10"],
            database_path=str(tmp_path / "missing-db"),
            wavelength_range_um=(2.0, 15.0),
            num_points=16,
            incident_angle=0.0,
            polarization=0,
            tolerance=1e-3,
            complex_dtype="complex128",
        )
    except ValueError as exc:
        assert "database_path" in str(exc)
    else:
        raise AssertionError("bad database_path should raise a configuration error")


def test_evaluate_token_prediction_rejects_unsupported_complex_dtype(tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    (database_dir / "Ge.csv").write_text("wl,n,k\n2.0,4.0,0.1\n15.0,4.0,0.1\n", encoding="utf-8")
    record = {
        "sample_id": "sample-bad-dtype",
        "layer_count": 1,
        "structure_tokens": ["Ge_10"],
        "spectrum_rt": [0.1] * 32,
    }

    try:
        evaluate_token_prediction(
            record=record,
            predicted_tokens=["Ge_10"],
            database_path=str(database_dir),
            wavelength_range_um=(2.0, 15.0),
            num_points=16,
            incident_angle=0.0,
            polarization=0,
            tolerance=1e-3,
            complex_dtype="not-a-dtype",
        )
    except ValueError as exc:
        assert "complex_dtype" in str(exc)
    else:
        raise AssertionError("unsupported complex_dtype should raise a configuration error")


def test_evaluate_token_prediction_rejects_malformed_target_spectrum_length(tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    (database_dir / "Ge.csv").write_text("wl,n,k\n2.0,4.0,0.1\n15.0,4.0,0.1\n", encoding="utf-8")
    record = {
        "sample_id": "sample-bad-target",
        "layer_count": 1,
        "structure_tokens": ["Ge_10"],
        "spectrum_rt": [0.1] * 31,
    }

    try:
        evaluate_token_prediction(
            record=record,
            predicted_tokens=["Ge_10"],
            database_path=str(database_dir),
            wavelength_range_um=(2.0, 15.0),
            num_points=16,
            incident_angle=0.0,
            polarization=0,
            tolerance=1e-3,
            complex_dtype="complex128",
        )
    except ValueError as exc:
        assert "target_spectrum_rt" in str(exc)
        assert "32" in str(exc)
    else:
        raise AssertionError("malformed target spectrum lengths should raise ValueError")


def test_evaluate_records_rejects_malformed_dataset_spectrum_before_model_forward(monkeypatch):
    records = [
        {
            "sample_id": "sample-bad-row",
            "layer_count": 1,
            "structure_tokens": ["Ge_10"],
            "spectrum_rt": [0.1] * 31,
        }
    ]

    class UnexpectedModelCall(RuntimeError):
        pass

    def fail_if_called(*args, **kwargs):
        raise UnexpectedModelCall("model forward should not run for malformed dataset rows")

    monkeypatch.setattr("our_work.pretrain.scripts.run_eval.run_eval_sample", fail_if_called)

    try:
        evaluate_records(
            model=object(),
            tokenizer=object(),
            records=records,
            database_path="unused",
            wavelength_range_um=(2.0, 15.0),
            num_points=16,
            incident_angle=0.0,
            polarization=0,
            tolerance=1e-3,
            complex_dtype="complex128",
            max_new_tokens=10,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "target_spectrum_rt" in str(exc)
    except UnexpectedModelCall as exc:
        raise AssertionError(str(exc))
    else:
        raise AssertionError("malformed dataset rows should fail before model inference")


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


def test_run_eval_main_output_json_normalizes_non_finite_metrics(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "outputs" / "base_run" / "checkpoint-9"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    output_json = tmp_path / "payload.json"

    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.load_eval_components",
        lambda *args, **kwargs: (object(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.load_split_records",
        lambda *args, **kwargs: [
            {
                "sample_id": "sample-000",
                "layer_count": 1,
                "structure_tokens": ["Ge_10"],
                "spectrum_rt": [0.1] * 32,
            }
        ],
    )
    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.evaluate_records",
        lambda **kwargs: [
            {
                "sample_id": "sample-000",
                "target_layer_count": 1,
                "prediction_layer_count": 1,
                "target_tokens": ["Ge_10"],
                "predicted_tokens": ["Ge_10"],
                "token_exact_match": True,
                "generated_valid": True,
                "spectrum_rmse": float("nan"),
                "spectrum_mae": float("inf"),
                "target_spectrum_rt": [0.1] * 32,
                "predicted_spectrum_rt": [0.1] * 32,
            }
        ],
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
            "1",
            "--num-points",
            "16",
            "--output-dir",
            str(tmp_path / "eval-output"),
            "--disable-plots",
            "--output-json",
            str(output_json),
        ]
    )

    assert payload["results"][0]["spectrum_rmse"] is None
    assert payload["results"][0]["spectrum_mae"] is None
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["results"][0]["spectrum_rmse"] is None
    assert written["results"][0]["spectrum_mae"] is None
