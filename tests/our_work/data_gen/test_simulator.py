import numpy as np

from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, validate_rt_spectrum


def test_flatten_rt_spectrum_concatenates_r_and_t():
    flat = flatten_rt_spectrum(np.array([0.1, 0.2]), np.array([0.7, 0.6]))
    assert flat.tolist() == [0.1, 0.2, 0.7, 0.6]


def test_validate_rt_spectrum_rejects_energy_overflow():
    ok = validate_rt_spectrum(
        reflection=np.array([0.7, 0.8], dtype=np.float32),
        transmission=np.array([0.5, 0.4], dtype=np.float32),
        tolerance=1e-3,
    )
    assert ok is False
