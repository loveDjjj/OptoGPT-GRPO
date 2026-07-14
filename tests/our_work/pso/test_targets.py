import numpy as np
import pytest

from our_work.pso.targets import (
    build_binary_band_targets,
    build_default_targets,
    build_fixed_band_targets,
    build_lorentzian_targets,
)


def test_fixed_band_targets_match_requested_masks():
    wavelengths = np.array([2.0, 3.0, 4.0, 5.0, 8.0, 13.0, 15.0], dtype=np.float32)

    targets = {target.target_id: target for target in build_fixed_band_targets(wavelengths)}

    assert targets["broad_3_13"].absorption.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert targets["band_5_8"].absorption.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert targets["dual_3_5_8_13"].absorption.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert targets["notch_3_5"].absorption.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_lorentzian_targets_use_internal_centers_and_unit_peak():
    wavelengths = np.linspace(2.0, 15.0, 131, dtype=np.float32)

    targets = build_lorentzian_targets(wavelengths, center_step_um=0.1, fwhm_um=0.02)

    assert len(targets) == 129
    assert targets[0].center_um == 2.1
    assert targets[-1].center_um == 14.9
    assert targets[0].target_id == "lorentz_fwhm_0p02_center_2p1"
    assert np.isclose(float(targets[0].absorption.max()), 1.0)


def test_default_targets_include_four_fixed_and_all_lorentzian_profiles():
    wavelengths = np.linspace(2.0, 15.0, 1024, dtype=np.float32)

    targets = build_default_targets(wavelengths)

    assert len(targets) == 133
    assert [target.target_id for target in targets[:4]] == [
        "broad_3_13",
        "band_5_8",
        "dual_3_5_8_13",
        "notch_3_5",
    ]


def test_binary_band_targets_keep_31_low_transition_patterns():
    wavelengths = np.asarray([2.0, 3.0, 4.0, 6.0, 10.0, 14.0, 20.0, 25.0], dtype=np.float32)
    targets = build_binary_band_targets(
        wavelengths,
        band_edges_um=[2.0, 3.0, 5.0, 8.0, 13.0, 16.0, 25.0],
        max_transitions=2,
        exclude_all_low=True,
        family="binary_band_2_25",
    )

    assert len(targets) == 31
    by_id = {target.target_id: target for target in targets}
    assert "bands_000000" not in by_id
    assert "bands_010101" not in by_id
    assert "bands_010100" not in by_id
    assert by_id["bands_011100"].family == "binary_band_2_25"
    assert by_id["bands_011100"].absorption.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_binary_band_targets_validate_edges():
    wavelengths = np.linspace(2.0, 25.0, 16, dtype=np.float32)

    with pytest.raises(ValueError, match="strictly increasing"):
        build_binary_band_targets(wavelengths, band_edges_um=[2.0, 5.0, 5.0, 25.0])
