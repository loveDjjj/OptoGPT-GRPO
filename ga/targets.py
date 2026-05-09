from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import copy

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

DEFAULT_GA_TASK_SPECS: list[dict[str, Any]] = [
    {
        "target_id": "broad_3_13_high",
        "family": "seeded_band",
        "description": "3-13 um high absorption; wavelengths outside this band are ignored by the loss.",
        "bands": [{"start_um": 3.0, "end_um": 13.0, "absorption": 1.0}],
        "seed_tokens": list(SEEDED_SOLUTIONS["broad_3_13_high"]),
    },
    {
        "target_id": "mid_5_8_high",
        "family": "seeded_band",
        "description": "3-5 low, 5-8 high, 8-13 low absorption; outside bands ignored.",
        "bands": [
            {"start_um": 3.0, "end_um": 5.0, "absorption": 0.0},
            {"start_um": 5.0, "end_um": 8.0, "absorption": 1.0},
            {"start_um": 8.0, "end_um": 13.0, "absorption": 0.0},
        ],
        "seed_tokens": list(SEEDED_SOLUTIONS["mid_5_8_high"]),
    },
    {
        "target_id": "dual_3_5_8_13_high",
        "family": "seeded_band",
        "description": "3-5 high, 5-8 low, 8-13 high absorption; outside bands ignored.",
        "bands": [
            {"start_um": 3.0, "end_um": 5.0, "absorption": 1.0},
            {"start_um": 5.0, "end_um": 8.0, "absorption": 0.0},
            {"start_um": 8.0, "end_um": 13.0, "absorption": 1.0},
        ],
        "seed_tokens": list(SEEDED_SOLUTIONS["dual_3_5_8_13_high"]),
    },
]


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


def default_ga_task_specs() -> list[dict[str, Any]]:
    return copy.deepcopy(DEFAULT_GA_TASK_SPECS)


def _normalize_band_specs(task_id: str, band_specs: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    normalized: list[tuple[float, float, float]] = []
    for index, band in enumerate(list(band_specs or [])):
        if not isinstance(band, dict):
            raise ValueError(f"Task '{task_id}' band #{index} must be a mapping")
        start_um = float(band["start_um"])
        end_um = float(band["end_um"])
        absorption = float(band["absorption"])
        if end_um < start_um:
            raise ValueError(f"Task '{task_id}' band #{index} has end_um < start_um")
        normalized.append((start_um, end_um, absorption))
    if not normalized:
        raise ValueError(f"Task '{task_id}' must define at least one band")
    return normalized


def _sample_random_seed_tokens(
    *,
    task_id: str,
    random_init_cfg: dict[str, Any],
    material_names: list[str],
    thickness_values_nm: list[int],
    rng: np.random.Generator,
) -> list[str]:
    if not material_names:
        raise ValueError(f"Task '{task_id}' cannot build a random seed without material_names")
    if not thickness_values_nm:
        raise ValueError(f"Task '{task_id}' cannot build a random seed without thickness_values_nm")

    layer_count = random_init_cfg.get("layer_count")
    if layer_count is None:
        min_layers = int(random_init_cfg.get("min_layers", random_init_cfg.get("min_layer_count", 6)))
        max_layers = int(random_init_cfg.get("max_layers", random_init_cfg.get("max_layer_count", 10)))
        if max_layers < min_layers:
            raise ValueError(f"Task '{task_id}' random_init max_layers must be >= min_layers")
        layer_count = int(rng.integers(min_layers, max_layers + 1))
    else:
        layer_count = int(layer_count)
    if layer_count <= 0:
        raise ValueError(f"Task '{task_id}' random_init layer_count must be positive")

    allowed_materials = [str(value) for value in random_init_cfg.get("materials", material_names)]
    missing = sorted(set(allowed_materials) - set(material_names))
    if missing:
        raise ValueError(f"Task '{task_id}' random_init materials not found in runtime materials: {missing}")

    sampled_materials = rng.choice(np.asarray(allowed_materials, dtype=object), size=layer_count, replace=True).tolist()
    sampled_thickness = rng.choice(np.asarray(thickness_values_nm, dtype=np.int32), size=layer_count, replace=True).tolist()
    return [f"{material}_{int(thickness)}" for material, thickness in zip(sampled_materials, sampled_thickness, strict=True)]


def validate_seed_tokens(
    seed_tokens: list[str],
    *,
    task_id: str,
    material_names: list[str],
    thickness_values_nm: list[int],
) -> list[str]:
    valid_materials = set(str(value) for value in material_names)
    valid_thickness = set(int(value) for value in thickness_values_nm)
    validated: list[str] = []
    for token in seed_tokens:
        material, thickness_text = str(token).rsplit("_", 1)
        thickness = int(thickness_text)
        if material not in valid_materials:
            raise ValueError(f"Task '{task_id}' seed material '{material}' is not in the configured material set")
        if thickness not in valid_thickness:
            raise ValueError(
                f"Task '{task_id}' seed thickness {thickness} nm is outside the configured thickness grid"
            )
        validated.append(f"{material}_{thickness}")
    return validated


def build_ga_targets_from_task_specs(
    wavelengths_um: np.ndarray,
    task_specs: list[dict[str, Any]],
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
    seed: int = 42,
    max_seed_thickness_nm: int = 500,
    thickness_step_nm: int = 10,
) -> list[GATargetProfile]:
    rng = np.random.default_rng(int(seed))
    targets: list[GATargetProfile] = []
    for task_index, task_spec in enumerate(list(task_specs or [])):
        if not isinstance(task_spec, dict):
            raise ValueError(f"Task #{task_index} must be a mapping")
        target_id = str(task_spec["target_id"])
        family = str(task_spec.get("family", "custom_band"))
        bands = _normalize_band_specs(target_id, list(task_spec.get("bands", [])))
        absorption, mask = _piecewise_profile(np.asarray(wavelengths_um), bands)

        raw_seed_tokens = task_spec.get("seed_tokens")
        if raw_seed_tokens:
            seed_tokens = preprocess_seed_tokens(
                [str(token) for token in raw_seed_tokens],
                max_thickness_nm=max_seed_thickness_nm,
                step_nm=thickness_step_nm,
            )
        else:
            random_init_cfg = dict(task_spec.get("random_init", {}))
            seed_tokens = _sample_random_seed_tokens(
                task_id=target_id,
                random_init_cfg=random_init_cfg,
                material_names=material_names,
                thickness_values_nm=thickness_values_nm,
                rng=rng,
            )
        seed_tokens = validate_seed_tokens(
            seed_tokens,
            task_id=target_id,
            material_names=material_names,
            thickness_values_nm=thickness_values_nm,
        )
        targets.append(
            GATargetProfile(
                target_id=target_id,
                family=family,
                absorption=absorption,
                loss_mask=mask,
                seed_tokens=seed_tokens,
                description=str(task_spec.get("description", "")),
            )
        )
    if not targets:
        raise ValueError("No GA target tasks were defined")
    return targets


def collect_seed_thickness_values(
    task_specs: list[dict[str, Any]],
    *,
    max_seed_thickness_nm: int = 500,
    thickness_step_nm: int = 10,
) -> list[int]:
    values: set[int] = set()
    for task_spec in list(task_specs or []):
        raw_seed_tokens = list(task_spec.get("seed_tokens", []) or [])
        if not raw_seed_tokens:
            continue
        for token in preprocess_seed_tokens(
            [str(value) for value in raw_seed_tokens],
            max_thickness_nm=max_seed_thickness_nm,
            step_nm=thickness_step_nm,
        ):
            values.add(int(token.rsplit("_", 1)[1]))
    return sorted(values)


def build_default_ga_targets(wavelengths_um: np.ndarray) -> list[GATargetProfile]:
    return build_ga_targets_from_task_specs(
        np.asarray(wavelengths_um),
        default_ga_task_specs(),
        material_names=seed_material_names(),
        thickness_values_nm=seed_thickness_values_nm(),
        seed=42,
    )


def seed_material_names() -> list[str]:
    names: set[str] = set()
    for tokens in SEEDED_SOLUTIONS.values():
        for token in tokens:
            names.add(token.rsplit("_", 1)[0])
    return sorted(names)


def seed_thickness_values_nm() -> list[int]:
    return collect_seed_thickness_values(default_ga_task_specs())
