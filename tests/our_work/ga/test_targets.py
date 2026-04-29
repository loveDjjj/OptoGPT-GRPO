import numpy as np

from our_work.ga.targets import build_default_ga_targets


def test_default_ga_targets_include_three_seeded_profiles_with_masks():
    wavelengths = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 15.0], dtype=np.float32)

    targets = {target.target_id: target for target in build_default_ga_targets(wavelengths)}

    assert sorted(targets) == ["broad_3_13_high", "dual_3_5_8_13_high", "mid_5_8_high"]
    assert targets["broad_3_13_high"].loss_mask.tolist() == [False, True, True, True, True, True, True, True, False]
    assert targets["broad_3_13_high"].absorption.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

    assert targets["mid_5_8_high"].absorption.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert targets["mid_5_8_high"].loss_mask.tolist() == [False, True, True, True, True, True, True, True, False]

    assert targets["dual_3_5_8_13_high"].absorption.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert targets["dual_3_5_8_13_high"].seed_tokens[-1] == "Au_100"
