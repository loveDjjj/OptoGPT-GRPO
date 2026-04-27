from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetProfile:
    target_id: str
    family: str
    absorption: np.ndarray
    center_um: float | None = None
    fwhm_um: float | None = None


def _band_profile(wavelengths_um: np.ndarray, bands: list[tuple[float, float, float]]) -> np.ndarray:
    profile = np.zeros_like(wavelengths_um, dtype=np.float32)
    for start_um, end_um, value in bands:
        mask = (wavelengths_um >= float(start_um)) & (wavelengths_um <= float(end_um))
        profile[mask] = float(value)
    return profile


def build_fixed_band_targets(wavelengths_um: np.ndarray) -> list[TargetProfile]:
    wavelengths = np.asarray(wavelengths_um, dtype=np.float32)
    return [
        TargetProfile("broad_3_13", "fixed", _band_profile(wavelengths, [(3.0, 13.0, 1.0)])),
        TargetProfile("band_5_8", "fixed", _band_profile(wavelengths, [(5.0, 8.0, 1.0)])),
        TargetProfile(
            "dual_3_5_8_13",
            "fixed",
            _band_profile(wavelengths, [(3.0, 5.0, 1.0), (8.0, 13.0, 1.0)]),
        ),
        TargetProfile(
            "notch_3_5",
            "fixed",
            _band_profile(wavelengths, [(2.0, 15.0, 1.0), (3.0, 5.0, 0.0)]),
        ),
    ]


def lorentzian_profile(wavelengths_um: np.ndarray, center_um: float, fwhm_um: float) -> np.ndarray:
    if fwhm_um <= 0:
        raise ValueError("fwhm_um must be positive")
    gamma = float(fwhm_um) / 2.0
    x = (np.asarray(wavelengths_um, dtype=np.float32) - float(center_um)) / gamma
    profile = 1.0 / (1.0 + x * x)
    peak = float(np.max(profile))
    if peak > 0:
        profile = profile / peak
    return profile.astype(np.float32)


def _format_center_id(center_um: float) -> str:
    return f"{center_um:.1f}".replace(".", "p")


def build_lorentzian_targets(
    wavelengths_um: np.ndarray,
    *,
    center_min_um: float = 2.1,
    center_max_um: float = 14.9,
    center_step_um: float = 0.1,
    fwhm_um: float = 0.02,
) -> list[TargetProfile]:
    if center_step_um <= 0:
        raise ValueError("center_step_um must be positive")
    centers = np.round(
        np.arange(float(center_min_um), float(center_max_um) + center_step_um * 0.5, float(center_step_um)),
        10,
    )
    targets: list[TargetProfile] = []
    for center in centers:
        center_value = round(float(center), 1)
        targets.append(
            TargetProfile(
                target_id=f"lorentz_fwhm_0p02_center_{_format_center_id(center_value)}",
                family="lorentzian",
                absorption=lorentzian_profile(wavelengths_um, center_value, fwhm_um),
                center_um=center_value,
                fwhm_um=float(fwhm_um),
            )
        )
    return targets


def build_default_targets(wavelengths_um: np.ndarray) -> list[TargetProfile]:
    return [
        *build_fixed_band_targets(wavelengths_um),
        *build_lorentzian_targets(wavelengths_um),
    ]
