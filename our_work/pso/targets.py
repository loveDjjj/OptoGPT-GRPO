from __future__ import annotations

from dataclasses import dataclass
from itertools import product

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


def _binary_band_profile(
    wavelengths_um: np.ndarray,
    band_edges_um: list[float],
    pattern: tuple[int, ...],
) -> np.ndarray:
    wavelengths = np.asarray(wavelengths_um, dtype=np.float32)
    profile = np.zeros_like(wavelengths, dtype=np.float32)
    for band_index, value in enumerate(pattern):
        start_um = float(band_edges_um[band_index])
        end_um = float(band_edges_um[band_index + 1])
        # Shared boundaries belong to the preceding interval, matching the
        # piecewise target convention used by the GA task builder.
        if band_index == 0:
            mask = (wavelengths >= start_um) & (wavelengths <= end_um)
        else:
            mask = (wavelengths > start_um) & (wavelengths <= end_um)
        profile[mask] = float(value)
    return profile


def build_binary_band_targets(
    wavelengths_um: np.ndarray,
    *,
    band_edges_um: list[float],
    max_transitions: int | None = None,
    exclude_all_low: bool = False,
    family: str = "binary_band",
) -> list[TargetProfile]:
    """Build high/low absorption targets over adjacent wavelength bands."""

    edges = [float(value) for value in band_edges_um]
    if len(edges) < 2:
        raise ValueError("band_edges_um must contain at least two values")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("band_edges_um must be strictly increasing")
    if max_transitions is not None and int(max_transitions) < 0:
        raise ValueError("max_transitions must be non-negative or null")

    band_count = len(edges) - 1
    targets: list[TargetProfile] = []
    for pattern in product((0, 1), repeat=band_count):
        if exclude_all_low and not any(pattern):
            continue
        transition_count = sum(left != right for left, right in zip(pattern, pattern[1:]))
        if max_transitions is not None and transition_count > int(max_transitions):
            continue
        pattern_id = "".join(str(value) for value in pattern)
        targets.append(
            TargetProfile(
                target_id=f"bands_{pattern_id}",
                family=str(family),
                absorption=_binary_band_profile(wavelengths_um, edges, pattern),
            )
        )
    return targets


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
