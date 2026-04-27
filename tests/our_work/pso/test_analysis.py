from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from our_work.pso.analysis import analyze_pso_dataset
from our_work.pso.analysis.run_analyze_pso import main as analyze_main


def _write_tiny_pso_dataset(base: Path) -> Path:
    dataset_dir = base / "pso-dataset"
    (dataset_dir / "shards").mkdir(parents=True)
    (dataset_dir / "splits").mkdir(parents=True)
    (dataset_dir / "stats").mkdir(parents=True)

    records = [
        {
            "sample_id": "pso-000",
            "layer_count": 5,
            "structure_tokens": ["Ge_10", "SiO2_20", "Ge_30", "Si_40", "ZnS_50"],
            "token_ids": [1, 2, 3, 4, 5],
            "materials": ["Ge", "SiO2", "Ge", "Si", "ZnS"],
            "thickness_nm": [10, 20, 30, 40, 50],
            "spectrum_rt": [0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2],
            "target_id": "broad_3_13",
            "target_family": "fixed",
            "target_center_um": None,
            "target_fwhm_um": None,
            "target_mse": 0.004,
            "acceptance_mse_threshold": 0.01,
            "pso_seed": 42,
            "pso_restart_index": 0,
        },
        {
            "sample_id": "pso-001",
            "layer_count": 5,
            "structure_tokens": ["Ge_10", "SiO2_20", "Ge_30", "Si_40", "ZnS_50"],
            "token_ids": [1, 2, 3, 4, 5],
            "materials": ["Ge", "SiO2", "Ge", "Si", "ZnS"],
            "thickness_nm": [10, 20, 30, 40, 50],
            "spectrum_rt": [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
            "target_id": "broad_3_13",
            "target_family": "fixed",
            "target_center_um": None,
            "target_fwhm_um": None,
            "target_mse": 0.008,
            "acceptance_mse_threshold": 0.01,
            "pso_seed": 43,
            "pso_restart_index": 1,
        },
        {
            "sample_id": "pso-002",
            "layer_count": 6,
            "structure_tokens": ["Si_10", "Si_20", "Ge_30", "Ge_40", "ZnS_50", "SiO2_60"],
            "token_ids": [6, 7, 3, 8, 5, 9],
            "materials": ["Si", "Si", "Ge", "Ge", "ZnS", "SiO2"],
            "thickness_nm": [10, 20, 30, 40, 50, 60],
            "spectrum_rt": [0.0, 0.2, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
            "target_id": "lorentz_fwhm_0p02_center_3p0",
            "target_family": "lorentzian",
            "target_center_um": 3.0,
            "target_fwhm_um": 0.02,
            "target_mse": 0.002,
            "acceptance_mse_threshold": 0.01,
            "pso_seed": 44,
            "pso_restart_index": 0,
        },
    ]
    pd.DataFrame.from_records(records[:2]).to_parquet(dataset_dir / "shards" / "shard-00000.parquet", index=False)
    pd.DataFrame.from_records(records[2:]).to_parquet(dataset_dir / "shards" / "shard-00001.parquet", index=False)
    (dataset_dir / "splits" / "split_manifest.json").write_text(
        json.dumps({"train": ["shard-00000.parquet"], "val": ["shard-00001.parquet"], "test": []}),
        encoding="utf-8",
    )
    (dataset_dir / "stats" / "search_summary.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "world_size": 1,
                "work_item_count": 2,
                "accepted_count": 3,
                "search": [
                    {
                        "target_id": "broad_3_13",
                        "layer_count": 5,
                        "accepted_count": 2,
                        "globally_kept_count": 2,
                        "global_duplicate_count": 1,
                        "shortfall": 0,
                        "total_evaluated": 128,
                        "duplicate_accepted": 1,
                        "restarts_used": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def test_analyze_pso_dataset_writes_quality_structure_and_spectrum_artifacts(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_pso_dataset(tmp_path)
    output_dir = tmp_path / "analysis"

    summary = analyze_pso_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        splits=["all", "train", "val"],
        wavelength_min_um=2.0,
        wavelength_max_um=15.0,
        top_k=2,
    )

    assert summary["record_count"] == 3
    assert summary["unique_structure_count"] == 2
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "tables" / "target_layer_stats.csv").exists()
    assert (output_dir / "tables" / "material_stats.csv").exists()
    assert (output_dir / "tables" / "diversity_stats.csv").exists()
    assert (output_dir / "tables" / "search_efficiency.csv").exists()
    assert (output_dir / "figures" / "mse_by_target.png").exists()
    assert (output_dir / "figures" / "structures" / "material_frequency.png").exists()
    assert (output_dir / "figures" / "spectra" / "broad_3_13" / "layer_05_topk.png").exists()
    assert (output_dir / "figures" / "lorentzian" / "center_vs_best_mse.png").exists()

    stats = pd.read_csv(output_dir / "tables" / "target_layer_stats.csv")
    broad_stats = stats[(stats["target_id"] == "broad_3_13") & (stats["layer_count"] == 5)].iloc[0]
    assert broad_stats["record_count"] == 2
    assert broad_stats["mse_min"] == 0.004


def test_run_analyze_pso_main_supports_dataset_and_output_dirs(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_pso_dataset(tmp_path)
    output_dir = tmp_path / "cli-analysis"

    analyze_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--split",
            "all",
            "--wavelength-min-um",
            "2.0",
            "--wavelength-max-um",
            "15.0",
            "--top-k",
            "1",
        ]
    )

    manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_dir"] == str(dataset_dir)
    assert manifest["splits"] == ["all"]
    assert (output_dir / "tables" / "best_samples.csv").exists()
