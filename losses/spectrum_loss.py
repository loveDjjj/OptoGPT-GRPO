"""基于 TMM 的光谱损失计算。"""

from __future__ import annotations

from typing import List, Mapping, Sequence

import numpy as np

from physics import calculate_optical_properties_batch
from physics.spectrum import flatten_rt, is_physical_spectrum, spectrum_error
from physics.structure import bucket_indices_by_layer_count, pad_tmm_configs_to_max_layers, tokens_to_tmm_config


def _normalize_targets(target_spectra: Sequence[Sequence[float]] | Sequence[float], count: int) -> List[np.ndarray]:
    """把单条目标光谱或批量目标光谱统一成列表形式。"""

    arr = np.asarray(target_spectra, dtype=np.float32)
    if arr.ndim == 1:
        return [arr for _ in range(count)]
    if arr.ndim == 2 and arr.shape[0] == count:
        return [arr[i] for i in range(count)]
    raise ValueError("target_spectra 的形状必须是 (142,) 或 (batch, 142)。")


def evaluate_generated_structures(
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
    database_path: str = "data/materials",
    material_aliases: Mapping[str, str] | None = None,
    return_spectra: bool = True,
    pad_to_max_layers: bool = False,
    bucket_by_layer_count: bool = False,
    pad_material: str = "Air",
    batch_size: int | None = None,
    tmm_debug: bool = False,
) -> List[dict]:
    """评估生成结构对应的光谱损失。

    输出字段中显式使用 `spectrum_loss`，避免继续沿用 RL 阶段的 `reward` 概念。
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
                "spectrum_loss": float(invalid_structure_penalty),
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

    grouped_eval_items: List[List[dict]] = []
    if bucket_by_layer_count:
        index_buckets = bucket_indices_by_layer_count([item["structure_tokens"] for item in valid_items])
        for _, bucket_indices in sorted(index_buckets.items(), key=lambda pair: pair[0]):
            bucket_items = [valid_items[idx] for idx in bucket_indices]
            if pad_to_max_layers:
                padded_configs, padded_layers = pad_tmm_configs_to_max_layers(
                    [item["config"] for item in bucket_items],
                    pad_material=pad_material,
                )
                bucket_items = [
                    dict(item, padded_layer_count=padded_layers, config=config)
                    for item, config in zip(bucket_items, padded_configs)
                ]
            grouped_eval_items.append(bucket_items)
    else:
        eval_items = list(valid_items)
        if pad_to_max_layers:
            padded_configs, padded_layers = pad_tmm_configs_to_max_layers(
                [item["config"] for item in valid_items],
                pad_material=pad_material,
            )
            eval_items = [
                dict(item, padded_layer_count=padded_layers, config=config)
                for item, config in zip(valid_items, padded_configs)
            ]
        grouped_eval_items.append(eval_items)

    total_eval_count = sum(len(group) for group in grouped_eval_items)
    effective_batch_size = int(batch_size or total_eval_count)
    effective_batch_size = max(1, effective_batch_size)

    for eval_items in grouped_eval_items:
        for start in range(0, len(eval_items), effective_batch_size):
            batch_items = eval_items[start : start + effective_batch_size]
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
                        "spectrum_loss": float(invalid_structure_penalty),
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
                        "spectrum_loss": float(nonphysical_penalty),
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
                    "spectrum_loss": float(error),
                    "status": "ok",
                }
                if return_spectra:
                    result["wavelengths_um"] = np.asarray(wavelengths, dtype=np.float32)
                    result["reflection"] = reflection
                    result["transmission"] = transmission
                    result["predicted_spectrum"] = predicted_spectrum
                results[original_idx] = result

    return results
