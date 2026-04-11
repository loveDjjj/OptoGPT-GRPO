from pathlib import Path

import numpy as np
import pandas as pd

from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch
from our_work.pretrain.scripts.run_eval import evaluate_token_prediction, resolve_repo_path


def test_evaluate_token_prediction_returns_zero_rmse_for_matching_tokens(tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    pd.DataFrame({"wl": [2.0, 15.0], "n": [1.4, 1.4], "k": [0.0, 0.0]}).to_csv(
        database_dir / "SiO2.csv",
        index=False,
    )
    pd.DataFrame({"wl": [2.0, 15.0], "n": [4.0, 4.0], "k": [0.1, 0.1]}).to_csv(
        database_dir / "Ge.csv",
        index=False,
    )

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
