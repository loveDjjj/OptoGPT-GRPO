"""TMM-based reward evaluation for OptoGPT-generated multilayer structures.

The policy predicts discrete structure tokens. This module converts those
tokens into TMM configurations, evaluates their optical response in batches, and
maps the result back to scalar rewards. The current default reward is the
negative absorption RMSE against the target spectrum.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence

import numpy as np

from TMM import calculate_optical_properties_batch
from utils.structure import pad_tmm_configs_to_max_layers, tokens_to_tmm_config


def flatten_rt(reflection: Sequence[float], transmission: Sequence[float]) -> np.ndarray:
    """Pack reflection and transmission curves into the 142-D model layout."""

    return np.concatenate([np.asarray(reflection, dtype=np.float32), np.asarray(transmission, dtype=np.float32)])


def absorption_curve(reflection: Sequence[float], transmission: Sequence[float]) -> np.ndarray:
    """Recover absorption from reflection/transmission using A = 1 - R - T."""

    reflection_np = np.asarray(reflection, dtype=np.float32)
    transmission_np = np.asarray(transmission, dtype=np.float32)
    return 1.0 - reflection_np - transmission_np


def _split_rt_spectrum(spectrum: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    spectrum_np = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    if spectrum_np.size % 2 != 0:
        raise ValueError(f"Spectrum length must be even, got {spectrum_np.size}.")
    half = spectrum_np.size // 2
    return spectrum_np[:half], spectrum_np[half:]


def spectrum_error(
    predicted_spectrum: Sequence[float],
    target_spectrum: Sequence[float],
    metric: str = "absorption_rmse",
) -> float:
    """Compute the configured spectrum matching error."""

    predicted = np.asarray(predicted_spectrum, dtype=np.float32)
    target = np.asarray(target_spectrum, dtype=np.float32)

    if metric == "absorption_rmse":
        pred_r, pred_t = _split_rt_spectrum(predicted)
        target_r, target_t = _split_rt_spectrum(target)
        pred_a = absorption_curve(pred_r, pred_t)
        target_a = absorption_curve(target_r, target_t)
        return float(np.sqrt(np.mean(np.square(pred_a - target_a))))
    if metric == "mae":
        return float(np.mean(np.abs(predicted - target)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(predicted - target))))
    raise ValueError(f"Unsupported reward metric: {metric}")


def is_physical_spectrum(
    reflection: Sequence[float],
    transmission: Sequence[float],
    tolerance: float = 0.01,
) -> bool:
    """Reject spectra that violate basic passive-optics constraints."""

    reflection_np = np.asarray(reflection, dtype=np.float32)
    transmission_np = np.asarray(transmission, dtype=np.float32)
    if not np.all(np.isfinite(reflection_np)) or not np.all(np.isfinite(transmission_np)):
        return False
    if np.min(reflection_np) < -tolerance or np.max(reflection_np) > 1.0 + tolerance:
        return False
    if np.min(transmission_np) < -tolerance or np.max(transmission_np) > 1.0 + tolerance:
        return False
    if np.max(reflection_np + transmission_np) > 1.0 + tolerance:
        return False
    return True


def _normalize_targets(target_spectra: Sequence[Sequence[float]] | Sequence[float], count: int) -> List[np.ndarray]:
    arr = np.asarray(target_spectra, dtype=np.float32)
    if arr.ndim == 1:
        return [arr for _ in range(count)]
    if arr.ndim == 2 and arr.shape[0] == count:
        return [arr[i] for i in range(count)]
    raise ValueError("target_spectra must be shape (142,) or (batch, 142).")


def evaluate_structures_with_tmm(
    structure_token_groups: Sequence[Sequence[str]],
    target_spectra: Sequence[Sequence[float]] | Sequence[float],
    wavelength_range_um: Sequence[float] = (0.4, 1.1),
    num_points: int = 71,
    incident_angle: float = 0.0,
    polarization: int = 0,
    metric: str = "absorption_rmse",
    invalid_structure_penalty: float = 1.0,
    nonphysical_spectrum_penalty: float | None = None,
    physical_tolerance: float = 0.01,
    database_path: str = "nk",
    material_aliases: Mapping[str, str] | None = None,
    return_spectra: bool = True,
    pad_to_max_layers: bool = False,
    pad_material: str = "Air",
    batch_size: int | None = None,
    tmm_debug: bool = False,
) -> List[dict]:
    """Evaluate a batch of decoded structures with the TMM solver.

    The function keeps reward evaluation separate from policy code:
    token decoding belongs to the policy, while physics-based scoring belongs
    here. This makes the reward path easier to test and reuse.
    """

    results: List[dict] = [None] * len(structure_token_groups)
    normalized_targets = _normalize_targets(target_spectra, len(structure_token_groups))
    nonphysical_penalty = float(
        invalid_structure_penalty if nonphysical_spectrum_penalty is None else nonphysical_spectrum_penalty
    )

    valid_items = []
    for idx, tokens in enumerate(structure_token_groups):
        try:
            config = tokens_to_tmm_config(
                tokens=tokens,
                database_path=database_path,
                material_aliases=material_aliases,
            )
        except Exception as exc:
            results[idx] = {
                "index": idx,
                "structure_tokens": list(tokens),
                "error": float(invalid_structure_penalty),
                "reward": float(-invalid_structure_penalty),
                "status": f"invalid_structure: {exc}",
            }
            continue

        valid_items.append(
            {
                "index": idx,
                "structure_tokens": list(structure_token_groups[idx]),
                "layer_count": len(config["materials"]),
                "config": config,
            }
        )

    if not valid_items:
        return results

    eval_items = list(valid_items)
    if pad_to_max_layers:
        # Padding uses zero-thickness dummy layers so one mixed batch can be
        # evaluated at the same maximum layer count without changing physics.
        padded_configs, padded_layers = pad_tmm_configs_to_max_layers(
            [item["config"] for item in valid_items],
            pad_material=pad_material,
        )
        eval_items = [dict(item, padded_layer_count=padded_layers, config=config) for item, config in zip(valid_items, padded_configs)]

    effective_batch_size = int(batch_size or len(eval_items))
    effective_batch_size = max(1, effective_batch_size)

    for start in range(0, len(eval_items), effective_batch_size):
        batch_items = eval_items[start : start + effective_batch_size]
        batch_indices = [item["index"] for item in batch_items]
        batch_configs = [item["config"] for item in batch_items]
        wavelengths, reflections, transmissions = calculate_optical_properties_batch(
            structure_configs=batch_configs,
            wavelength_range=tuple(wavelength_range_um),
            num_points=num_points,
            incident_angle=incident_angle,
            polarization=polarization,
            plot_results=False,
            debug=tmm_debug,
        )

        if wavelengths is None:
            for item in batch_items:
                results[item["index"]] = {
                    "index": item["index"],
                    "structure_tokens": item["structure_tokens"],
                    "layer_count": item["layer_count"],
                    "padded_layer_count": int(item.get("padded_layer_count", item["layer_count"])),
                    "error": float(invalid_structure_penalty),
                    "reward": float(-invalid_structure_penalty),
                    "status": "tmm_failed",
                }
            continue

        for local_idx, item in enumerate(batch_items):
            original_idx = item["index"]
            reflection = np.asarray(reflections[local_idx], dtype=np.float32)
            transmission = np.asarray(transmissions[local_idx], dtype=np.float32)
            base_result = {
                "index": original_idx,
                "structure_tokens": item["structure_tokens"],
                "layer_count": item["layer_count"],
                "padded_layer_count": int(item.get("padded_layer_count", item["layer_count"])),
            }

            if not is_physical_spectrum(reflection, transmission, tolerance=physical_tolerance):
                result = {
                    **base_result,
                    "error": float(nonphysical_penalty),
                    "reward": float(-nonphysical_penalty),
                    "status": "nonphysical_spectrum",
                }
                if return_spectra:
                    result["wavelengths_um"] = np.asarray(wavelengths, dtype=np.float32)
                    result["reflection"] = reflection
                    result["transmission"] = transmission
                    result["predicted_spectrum"] = flatten_rt(reflection, transmission)
                results[original_idx] = result
                continue

            predicted_spectrum = flatten_rt(reflection, transmission)
            error = spectrum_error(predicted_spectrum, normalized_targets[original_idx], metric=metric)
            result = {
                **base_result,
                "error": float(error),
                "reward": float(-error),
                "status": "ok",
            }
            if return_spectra:
                result["wavelengths_um"] = np.asarray(wavelengths, dtype=np.float32)
                result["reflection"] = reflection
                result["transmission"] = transmission
                result["predicted_spectrum"] = predicted_spectrum
            results[original_idx] = result

    return results
