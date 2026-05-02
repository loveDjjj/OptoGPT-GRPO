from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GATargetProfile:
    target_id: str
    family: str
    absorption: np.ndarray
    loss_mask: np.ndarray
    seed_tokens: list[str]
    description: str = ""


SEEDED_SOLUTIONS: dict[str, list[str]] = {
    "broad_3_13_high": ["YbF3_870", "ZnS_480", "Si_280", "Bi_20", "Ge_130", "Bi_820", "Au_100"],
    "mid_5_8_high": ["Si_250", "SiO2_120", "Ge_500", "MgF2_850", "Ge_110", "MgF2_500", "Bi_130", "Au_100"],
    "dual_3_5_8_13_high": [
        "SiO2_150",
        "MgF2_500",
        "Si_500",
        "ZnS_450",
        "Ge_490",
        "MgF2_280",
        "Si_320",
        "Bi_250",
        "Au_100",
    ],
}


def preprocess_seed_tokens(seed_tokens: list[str], *, max_thickness_nm: int = 500, step_nm: int = 10) -> list[str]:
    if int(max_thickness_nm) <= 0:
        raise ValueError("max_thickness_nm must be positive")
    if int(step_nm) <= 0:
        raise ValueError("step_nm must be positive")

    processed: list[str] = []
    pending = list(seed_tokens)
    while pending:
        token = pending.pop(0)
        material, thickness_text = str(token).rsplit("_", 1)
        thickness = int(thickness_text)
        if thickness <= int(max_thickness_nm):
            processed.append(f"{material}_{thickness}")
            continue
        lower = max(int(step_nm), int((thickness // 2) // int(step_nm)) * int(step_nm))
        upper = max(int(step_nm), thickness - lower)
        pending = [f"{material}_{lower}", f"{material}_{upper}", *pending]
    return processed


def _piecewise_profile(wavelengths_um: np.ndarray, bands: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(wavelengths_um, dtype=np.float32)
    absorption = np.zeros_like(wavelengths, dtype=np.float32)
    mask = np.zeros_like(wavelengths, dtype=bool)
    for band_index, (start_um, end_um, value) in enumerate(bands):
        if band_index == 0:
            band_mask = (wavelengths >= float(start_um)) & (wavelengths <= float(end_um))
        else:
            # Shared boundaries belong to the previous interval, matching the
            # 3-5 / 5-8 / 8-13 band convention used by the seed targets.
            band_mask = (wavelengths > float(start_um)) & (wavelengths <= float(end_um))
        absorption[band_mask] = float(value)
        mask |= band_mask
    return absorption, mask


def build_default_ga_targets(wavelengths_um: np.ndarray) -> list[GATargetProfile]:
    broad_abs, broad_mask = _piecewise_profile(np.asarray(wavelengths_um), [(3.0, 13.0, 1.0)])
    mid_abs, mid_mask = _piecewise_profile(
        np.asarray(wavelengths_um),
        [(3.0, 5.0, 0.0), (5.0, 8.0, 1.0), (8.0, 13.0, 0.0)],
    )
    dual_abs, dual_mask = _piecewise_profile(
        np.asarray(wavelengths_um),
        [(3.0, 5.0, 1.0), (5.0, 8.0, 0.0), (8.0, 13.0, 1.0)],
    )
    return [
        GATargetProfile(
            target_id="broad_3_13_high",
            family="seeded_band",
            absorption=broad_abs,
            loss_mask=broad_mask,
            seed_tokens=preprocess_seed_tokens(list(SEEDED_SOLUTIONS["broad_3_13_high"])),
            description="3-13 um high absorption; wavelengths outside this band are ignored by the loss.",
        ),
        GATargetProfile(
            target_id="mid_5_8_high",
            family="seeded_band",
            absorption=mid_abs,
            loss_mask=mid_mask,
            seed_tokens=preprocess_seed_tokens(list(SEEDED_SOLUTIONS["mid_5_8_high"])),
            description="3-5 low, 5-8 high, 8-13 low absorption; outside bands ignored.",
        ),
        GATargetProfile(
            target_id="dual_3_5_8_13_high",
            family="seeded_band",
            absorption=dual_abs,
            loss_mask=dual_mask,
            seed_tokens=preprocess_seed_tokens(list(SEEDED_SOLUTIONS["dual_3_5_8_13_high"])),
            description="3-5 high, 5-8 low, 8-13 high absorption; outside bands ignored.",
        ),
    ]


def seed_material_names() -> list[str]:
    names: set[str] = set()
    for tokens in SEEDED_SOLUTIONS.values():
        for token in tokens:
            names.add(token.rsplit("_", 1)[0])
    return sorted(names)


def seed_thickness_values_nm() -> list[int]:
    values: set[int] = set()
    for tokens in SEEDED_SOLUTIONS.values():
        for token in preprocess_seed_tokens(list(tokens)):
            values.add(int(token.rsplit("_", 1)[1]))
    return sorted(values)
