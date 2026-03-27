"""基于 TMM 的光谱损失计算。"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import numpy as np

from physics import calculate_optical_properties_batch
from physics.spectrum import flatten_rt, is_physical_spectrum, spectrum_error, split_rt_spectrum
from physics.structure import (
    bucket_indices_by_layer_count,
    pad_tmm_configs_to_fixed_layers,
    pad_tmm_configs_to_max_layers,
    tokens_to_tmm_config,
)


def _normalize_targets(target_spectra: Sequence[Sequence[float]] | Sequence[float], count: int) -> List[np.ndarray]:
    """把单条目标光谱或批量目标光谱统一成列表形式。"""

    arr = np.asarray(target_spectra, dtype=np.float32)
    if arr.ndim == 1:
        return [arr for _ in range(count)]
    if arr.ndim == 2 and arr.shape[0] == count:
        return [arr[i] for i in range(count)]
    raise ValueError("target_spectra 的形状必须是 (142,) 或 (batch, 142)。")


def _format_return_payload(
    results: List[dict] | None,
    aux_arrays: dict[str, np.ndarray],
    *,
    return_item_results: bool,
    return_aux_arrays: bool,
):
    """统一整理返回值，兼容训练与评测两条调用链。"""

    if return_item_results and return_aux_arrays:
        return results, aux_arrays
    if return_item_results:
        return results
    return aux_arrays


def evaluate_generated_structures(
    structure_token_groups: Sequence[Sequence[str]],
    target_spectra: Sequence[Sequence[float]] | Sequence[float],
    wavelength_range_um: Sequence[float] = (0.4, 1.1),
    num_points: int = 71,
    incident_angle: float = 0.0,
    polarization: int = 0,
    metric: str = "rt_rmse",
    invalid_structure_penalty: float = 1.0,
    nonphysical_spectrum_penalty: float | None = None,
    physical_tolerance: float = 0.01,
    database_path: str = "data/materials",
    material_aliases: Mapping[str, str] | None = None,
    return_spectra: bool = True,
    pad_to_max_layers: bool = False,
    bucket_by_layer_count: bool = False,
    fixed_max_layers: int | None = None,
    pad_material: str = "Air",
    batch_size: int | None = None,
    tmm_debug: bool = False,
    return_aux_arrays: bool = False,
    return_item_results: bool = True,
) -> List[dict] | dict[str, np.ndarray] | tuple[List[dict], dict[str, np.ndarray]]:
    """评估生成结构对应的光谱损失。

    默认仍返回逐样本结果列表，兼容现有可视化与样本落盘逻辑。
    当训练或快速评测只关心批量统计时，可以关闭 `return_item_results`，
    直接拿批量数组，避免额外构造大量 Python dict。
    """

    sample_count = len(structure_token_groups)
    results: List[dict] | None = [None] * sample_count if return_item_results else None
    normalized_targets = _normalize_targets(target_spectra, sample_count)
    nonphysical_penalty = float(
        invalid_structure_penalty if nonphysical_spectrum_penalty is None else nonphysical_spectrum_penalty
    )

    spectrum_losses = np.full((sample_count,), float(invalid_structure_penalty), dtype=np.float32)
    ok_mask = np.zeros((sample_count,), dtype=np.bool_)
    r_rmse = np.full((sample_count,), float(invalid_structure_penalty), dtype=np.float32)
    t_rmse = np.full((sample_count,), float(invalid_structure_penalty), dtype=np.float32)
    predicted_spectra = None
    has_predicted_spectrum = None
    spectrum_dim = int(normalized_targets[0].shape[0]) if sample_count > 0 else int(num_points) * 2
    if return_aux_arrays and return_spectra and sample_count > 0:
        predicted_spectra = np.full((sample_count, spectrum_dim), np.nan, dtype=np.float32)
        has_predicted_spectrum = np.zeros((sample_count,), dtype=np.bool_)

    valid_items = []
    for idx, tokens in enumerate(structure_token_groups):
        try:
            config = tokens_to_tmm_config(
                tokens=tokens,
                database_path=database_path,
                material_aliases=material_aliases,
            )
        except Exception as exc:
            if results is not None:
                results[idx] = {
                    "index": idx,
                    "structure_tokens": list(tokens),
                    "spectrum_loss": float(invalid_structure_penalty),
                    "r_rmse": float(invalid_structure_penalty),
                    "t_rmse": float(invalid_structure_penalty),
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
        aux_arrays = {
            "spectrum_losses": spectrum_losses,
            "ok_mask": ok_mask,
            "r_rmse": r_rmse,
            "t_rmse": t_rmse,
        }
        if predicted_spectra is not None and has_predicted_spectrum is not None:
            aux_arrays["predicted_spectra"] = predicted_spectra
            aux_arrays["has_predicted_spectrum"] = has_predicted_spectrum
        return _format_return_payload(
            results,
            aux_arrays,
            return_item_results=return_item_results,
            return_aux_arrays=return_aux_arrays,
        )

    grouped_eval_items: List[List[dict]] = []
    if fixed_max_layers is not None:
        fixed_max_layers = int(fixed_max_layers)
        padded_inputs = []
        for item in valid_items:
            if item["layer_count"] > fixed_max_layers:
                if results is not None:
                    results[item["index"]] = {
                        "index": item["index"],
                        "structure_tokens": item["structure_tokens"],
                        "layer_count": item["layer_count"],
                        "padded_layer_count": fixed_max_layers,
                        "spectrum_loss": float(invalid_structure_penalty),
                        "r_rmse": float(invalid_structure_penalty),
                        "t_rmse": float(invalid_structure_penalty),
                        "status": f"too_many_layers>{fixed_max_layers}",
                    }
                continue
            padded_inputs.append(item)

        if padded_inputs:
            padded_configs = pad_tmm_configs_to_fixed_layers(
                [item["config"] for item in padded_inputs],
                target_layers=fixed_max_layers,
                pad_material=pad_material,
            )
            grouped_eval_items.append(
                [
                    dict(item, padded_layer_count=fixed_max_layers, config=config)
                    for item, config in zip(padded_inputs, padded_configs)
                ]
            )
    elif bucket_by_layer_count:
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
                    if results is not None:
                        results[item["index"]] = {
                            "index": item["index"],
                            "structure_tokens": item["structure_tokens"],
                            "layer_count": item["layer_count"],
                            "padded_layer_count": int(item.get("padded_layer_count", item["layer_count"])),
                            "spectrum_loss": float(invalid_structure_penalty),
                            "r_rmse": float(invalid_structure_penalty),
                            "t_rmse": float(invalid_structure_penalty),
                            "status": "tmm_failed",
                        }
                continue

            wavelengths_array = np.asarray(wavelengths, dtype=np.float32)
            for local_idx, item in enumerate(batch_items):
                original_idx = item["index"]
                reflection = np.asarray(reflections[local_idx], dtype=np.float32)
                transmission = np.asarray(transmissions[local_idx], dtype=np.float32)
                predicted_spectrum = flatten_rt(reflection, transmission)
                base_result = {
                    "index": original_idx,
                    "structure_tokens": item["structure_tokens"],
                    "layer_count": item["layer_count"],
                    "padded_layer_count": int(item.get("padded_layer_count", item["layer_count"])),
                }

                if predicted_spectra is not None and has_predicted_spectrum is not None:
                    predicted_spectra[original_idx] = predicted_spectrum
                    has_predicted_spectrum[original_idx] = True

                if not is_physical_spectrum(reflection, transmission, tolerance=physical_tolerance):
                    spectrum_losses[original_idx] = float(nonphysical_penalty)
                    if results is not None:
                        result = {
                            **base_result,
                            "spectrum_loss": float(nonphysical_penalty),
                            "r_rmse": float(nonphysical_penalty),
                            "t_rmse": float(nonphysical_penalty),
                            "status": "nonphysical_spectrum",
                        }
                        if return_spectra:
                            result["wavelengths_um"] = wavelengths_array
                            result["reflection"] = reflection
                            result["transmission"] = transmission
                            result["predicted_spectrum"] = predicted_spectrum
                        results[original_idx] = result
                    continue

                error = spectrum_error(predicted_spectrum, normalized_targets[original_idx], metric=metric)
                target_r, target_t = split_rt_spectrum(normalized_targets[original_idx])
                # 这里的 reflection / transmission 已经是单段 71 点曲线，
                # 不能再走要求 `[R..., T...]` 拼接布局的 `spectrum_error(..., metric='r_rmse/t_rmse')`。
                # 因此直接按单段曲线计算 RMSE，避免把接口语义混用。
                r_error = float(np.sqrt(np.mean(np.square(reflection - target_r))))
                t_error = float(np.sqrt(np.mean(np.square(transmission - target_t))))
                spectrum_losses[original_idx] = float(error)
                r_rmse[original_idx] = r_error
                t_rmse[original_idx] = t_error
                ok_mask[original_idx] = True
                if results is not None:
                    result = {
                        **base_result,
                        "spectrum_loss": float(error),
                        "r_rmse": r_error,
                        "t_rmse": t_error,
                        "status": "ok",
                    }
                    if return_spectra:
                        result["wavelengths_um"] = wavelengths_array
                        result["reflection"] = reflection
                        result["transmission"] = transmission
                        result["predicted_spectrum"] = predicted_spectrum
                    results[original_idx] = result

    aux_arrays = {
        "spectrum_losses": spectrum_losses,
        "ok_mask": ok_mask,
        "r_rmse": r_rmse,
        "t_rmse": t_rmse,
    }
    if predicted_spectra is not None and has_predicted_spectrum is not None:
        aux_arrays["predicted_spectra"] = predicted_spectra
        aux_arrays["has_predicted_spectrum"] = has_predicted_spectrum
    return _format_return_payload(
        results,
        aux_arrays,
        return_item_results=return_item_results,
        return_aux_arrays=return_aux_arrays,
    )
