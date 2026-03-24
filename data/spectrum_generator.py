"""Synthetic target-spectrum generators used for dataset-level GRPO training.

The current project trains against families of gate-like absorption targets.
Each generated target includes:

- absorption curve A(lambda),
- reflection/transmission curves encoded in the legacy 142-D input layout,
- metadata describing the sampled spectral task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_WAVELENGTHS_UM = np.linspace(0.4, 1.1, 71, dtype=np.float32)


@dataclass
class GateDatasetConfig:
    """Configuration for a synthetic gate-spectrum dataset split."""

    count: int
    seed: int
    wavelength_min_um: float = 0.4
    wavelength_max_um: float = 1.1
    num_points: int = 71
    width_min_nm: float = 80.0
    width_max_nm: float = 400.0
    edge_smooth_nm: float = 20.0
    families: Tuple[str, ...] = ("pass", "stop")


def wavelength_grid(
    wavelength_min_um: float = 0.4,
    wavelength_max_um: float = 1.1,
    num_points: int = 71,
) -> np.ndarray:
    """Return the wavelength grid shared by targets and the TMM solver."""

    return np.linspace(wavelength_min_um, wavelength_max_um, num_points, dtype=np.float32)


def flatten_spectrum(reflection: Sequence[float], transmission: Sequence[float]) -> np.ndarray:
    """Flatten reflection/transmission into the layout expected by OptoGPT."""

    return np.concatenate(
        [
            np.asarray(reflection, dtype=np.float32),
            np.asarray(transmission, dtype=np.float32),
        ]
    )


def _gate_window(
    wavelengths_nm: np.ndarray,
    left_nm: float,
    right_nm: float,
    edge_smooth_nm: float,
) -> np.ndarray:
    """Construct a hard or softly smoothed spectral gate window."""

    if edge_smooth_nm <= 0:
        return ((wavelengths_nm >= left_nm) & (wavelengths_nm <= right_nm)).astype(np.float32)

    scale = max(float(edge_smooth_nm), 1e-6)
    left_transition = 0.5 * (1.0 + np.tanh((wavelengths_nm - left_nm) / scale))
    right_transition = 0.5 * (1.0 + np.tanh((right_nm - wavelengths_nm) / scale))
    return np.clip(left_transition * right_transition, 0.0, 1.0).astype(np.float32)


def build_gate_absorption_target(
    wavelengths_um: np.ndarray,
    left_nm: float,
    right_nm: float,
    family: str = "pass",
    edge_smooth_nm: float = 20.0,
) -> Dict[str, np.ndarray | float | str]:
    """Build one gate-like absorption target and its derived R/T conditioning."""

    wavelengths_um = np.asarray(wavelengths_um, dtype=np.float32)
    wavelengths_nm = wavelengths_um * 1000.0
    window = _gate_window(
        wavelengths_nm=wavelengths_nm,
        left_nm=float(left_nm),
        right_nm=float(right_nm),
        edge_smooth_nm=float(edge_smooth_nm),
    )

    family_normalized = str(family).strip().lower()
    if family_normalized == "pass":
        absorption = window
    elif family_normalized == "stop":
        absorption = 1.0 - window
    else:
        raise ValueError(f"Unsupported gate family: {family}")

    # Standard opaque-absorber target assumption used throughout the current
    # project: T_target = 0 and R_target = 1 - A_target.
    transmission = np.zeros_like(absorption, dtype=np.float32)
    reflection = np.clip(1.0 - absorption, 0.0, 1.0).astype(np.float32)
    absorption = np.clip(absorption, 0.0, 1.0).astype(np.float32)

    return {
        "wavelengths_um": wavelengths_um,
        "reflection": reflection,
        "transmission": transmission,
        "absorption": absorption,
        "spectrum": flatten_spectrum(reflection, transmission),
        "family": family_normalized,
        "left_nm": float(left_nm),
        "right_nm": float(right_nm),
        "width_nm": float(max(0.0, right_nm - left_nm)),
        "edge_smooth_nm": float(edge_smooth_nm),
    }


def _sample_gate_interval(
    rng: np.random.Generator,
    wavelengths_um: np.ndarray,
    width_min_nm: float,
    width_max_nm: float,
) -> Tuple[float, float]:
    """Sample one valid wavelength interval on the discretized grid."""

    wavelengths_nm = np.asarray(wavelengths_um, dtype=np.float32) * 1000.0
    if wavelengths_nm.size < 2:
        raise ValueError("At least two wavelength points are required to sample a gate interval.")

    step_nm = float(np.mean(np.diff(wavelengths_nm)))
    min_width_points = max(1, int(round(float(width_min_nm) / step_nm)))
    max_width_points = max(min_width_points, int(round(float(width_max_nm) / step_nm)))
    max_width_points = min(max_width_points, int(wavelengths_nm.size))
    width_points = int(rng.integers(min_width_points, max_width_points + 1))
    start_max = int(wavelengths_nm.size - width_points)
    start_idx = int(rng.integers(0, start_max + 1))
    end_idx = int(start_idx + width_points - 1)
    return float(wavelengths_nm[start_idx]), float(wavelengths_nm[end_idx])


def generate_gate_target_batch(config: GateDatasetConfig) -> List[Dict[str, np.ndarray | float | str | int]]:
    """Generate one synthetic target dataset split."""

    rng = np.random.default_rng(config.seed)
    wavelengths_um = wavelength_grid(
        wavelength_min_um=config.wavelength_min_um,
        wavelength_max_um=config.wavelength_max_um,
        num_points=config.num_points,
    )
    families = tuple(str(family).strip().lower() for family in config.families)
    if not families:
        raise ValueError("GateDatasetConfig.families must contain at least one family.")

    targets: List[Dict[str, np.ndarray | float | str | int]] = []
    for target_id in range(config.count):
        left_nm, right_nm = _sample_gate_interval(
            rng=rng,
            wavelengths_um=wavelengths_um,
            width_min_nm=config.width_min_nm,
            width_max_nm=config.width_max_nm,
        )
        family = str(rng.choice(families))
        target = build_gate_absorption_target(
            wavelengths_um=wavelengths_um,
            left_nm=left_nm,
            right_nm=right_nm,
            family=family,
            edge_smooth_nm=config.edge_smooth_nm,
        )
        target["target_id"] = int(target_id)
        targets.append(target)
    return targets
