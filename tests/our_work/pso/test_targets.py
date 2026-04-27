import numpy as np

from our_work.pso.targets import build_default_targets, build_fixed_band_targets, build_lorentzian_targets


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
