from __future__ import annotations

import numpy as np

from our_work._shared.physics import calculate_optical_properties_batch
from our_work._shared.physics.structure import tokens_to_tmm_config


def flatten_rt_spectrum(reflection: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(reflection), np.asarray(transmission)], axis=0)


def validate_rt_spectrum(reflection: np.ndarray, transmission: np.ndarray, tolerance: float) -> bool:
    if not np.all(np.isfinite(reflection)) or not np.all(np.isfinite(transmission)):
        return False
    if float(reflection.min()) < -tolerance or float(reflection.max()) > 1.0 + tolerance:
        return False
    if float(transmission.min()) < -tolerance or float(transmission.max()) > 1.0 + tolerance:
        return False
    if float((reflection + transmission).max()) > 1.0 + tolerance:
        return False
    return True


def simulate_structure_batch(
    structure_token_groups: list[list[str]],
    *,
    database_path: str,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    incident_angle: float,
    polarization: int,
    tolerance: float,
    complex_dtype: str,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    configs = [
        tokens_to_tmm_config(tokens, database_path=database_path)
        for tokens in structure_token_groups
    ]
    wavelengths, reflections, transmissions = calculate_optical_properties_batch(
        structure_configs=configs,
        wavelength_range=wavelength_range_um,
        num_points=num_points,
        incident_angle=incident_angle,
        polarization=polarization,
        complex_dtype=complex_dtype,
    )
    if wavelengths is None:
        raise RuntimeError("TMM batch simulation failed.")
    ok_mask = np.asarray(
        [
            validate_rt_spectrum(reflection, transmission, tolerance)
            for reflection, transmission in zip(reflections, transmissions)
        ],
        dtype=np.bool_,
    )
    return np.asarray(wavelengths, dtype=np.float32), list(reflections), list(transmissions), ok_mask
