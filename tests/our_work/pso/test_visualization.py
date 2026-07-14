import json
from types import SimpleNamespace

import numpy as np

from our_work.pso.targets import TargetProfile
from our_work.pso.visualization import save_best_target_plots


def test_save_best_target_plots_writes_spectrum_structure_and_manifest(monkeypatch, tmp_path):
    wavelengths = np.linspace(2.0, 25.0, 16, dtype=np.float32)
    target = TargetProfile(
        target_id="bands_111111",
        family="binary_band_2_25",
        absorption=np.ones_like(wavelengths),
    )
    candidate = SimpleNamespace(
        structure_tokens=["Au_100", "SiO2_250"],
        target_mse=0.01,
        pso_seed=42,
        pso_restart_index=1,
    )
    reflection = np.full_like(wavelengths, 0.05)
    transmission = np.full_like(wavelengths, 0.05)
    monkeypatch.setattr(
        "our_work.pso.visualization._simulate_best_candidates",
        lambda candidates, tmm_config: {"bands_111111": (reflection, transmission)},
    )

    manifest = save_best_target_plots(
        output_dir=tmp_path,
        wavelengths_um=wavelengths,
        targets={target.target_id: target},
        candidates={target.target_id: candidate},
        tmm_config=SimpleNamespace(batch_size=8),
        dpi=100,
        include_rt=True,
    )

    assert manifest["target_count"] == 1
    assert manifest["plotted_count"] == 1
    assert (tmp_path / "bands_111111.png").stat().st_size > 1000
    payload = json.loads((tmp_path / "bands_111111.json").read_text(encoding="utf-8"))
    assert payload["structure_tokens"] == ["Au_100", "SiO2_250"]
    assert payload["materials"] == ["Au", "SiO2"]
    assert payload["thickness_nm"] == [100, 250]
    combined = json.loads((tmp_path / "best_structures.json").read_text(encoding="utf-8"))
    assert combined["targets"]["bands_111111"]["status"] == "ok"


def test_save_best_target_plots_records_missing_candidate(monkeypatch, tmp_path):
    wavelengths = np.linspace(2.0, 25.0, 8, dtype=np.float32)
    target = TargetProfile("bands_111111", "binary_band_2_25", np.ones_like(wavelengths))
    monkeypatch.setattr(
        "our_work.pso.visualization._simulate_best_candidates",
        lambda candidates, tmm_config: {},
    )

    manifest = save_best_target_plots(
        output_dir=tmp_path,
        wavelengths_um=wavelengths,
        targets={target.target_id: target},
        candidates={},
        tmm_config=SimpleNamespace(batch_size=8),
    )

    assert manifest["plotted_count"] == 0
    assert manifest["targets"]["bands_111111"]["status"] == "no_valid_candidate"
