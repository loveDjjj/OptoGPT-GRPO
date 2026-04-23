from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from our_work.data_gen.pipeline.simulator import flatten_rt_spectrum, simulate_structure_batch


def _is_invalid_structure_token(token: str) -> bool:
    parts = str(token).rsplit("_", 1)
    if len(parts) != 2 or not parts[0]:
        return True
    try:
        int(parts[1])
    except ValueError:
        return True
    return False


def compute_rollout_rewards(
    *,
    structure_token_groups: Sequence[Sequence[str]],
    target_spectra: Sequence[Sequence[float]] | np.ndarray,
    database_path: str,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    incident_angle: float,
    polarization: int,
    tolerance: float,
    complex_dtype: str,
    batch_size: int,
    invalid_structure_penalty: float,
    spectrum_metric: str = "rt_rmse",
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    if spectrum_metric != "rt_rmse":
        raise ValueError(f"unsupported spectrum_metric: {spectrum_metric}")

    target_spectra_np = np.asarray(target_spectra, dtype=np.float32)
    if target_spectra_np.ndim != 2:
        raise ValueError(f"target_spectra must be 2D, got {target_spectra_np.shape}")

    rewards = np.full((len(structure_token_groups),), -float(invalid_structure_penalty), dtype=np.float32)
    spectrum_losses = np.full((len(structure_token_groups),), float(invalid_structure_penalty), dtype=np.float32)
    ok_mask = np.zeros((len(structure_token_groups),), dtype=np.bool_)

    valid_indices: list[int] = []
    valid_groups: list[list[str]] = []
    for index, tokens in enumerate(structure_token_groups):
        if not tokens or any(_is_invalid_structure_token(token) for token in tokens):
            continue
        valid_indices.append(index)
        valid_groups.append(list(tokens))

    grouped_by_layers: dict[int, list[tuple[int, list[str]]]] = {}
    for global_index, tokens in zip(valid_indices, valid_groups):
        grouped_by_layers.setdefault(len(tokens), []).append((global_index, tokens))

    for _, grouped_items in grouped_by_layers.items():
        for start in range(0, len(grouped_items), int(batch_size)):
            chunk_items = grouped_items[start : start + int(batch_size)]
            chunk_indices = [item[0] for item in chunk_items]
            chunk_groups = [item[1] for item in chunk_items]
            try:
                _, reflections, transmissions, local_ok = simulate_structure_batch(
                    chunk_groups,
                    database_path=database_path,
                    wavelength_range_um=wavelength_range_um,
                    num_points=num_points,
                    incident_angle=incident_angle,
                    polarization=polarization,
                    tolerance=tolerance,
                    complex_dtype=complex_dtype,
                    device=device,
                )
            except Exception:
                # Rollout reward evaluation should degrade invalid chunks to the configured
                # penalty instead of aborting the whole GRPO update on a simulator failure.
                continue
            for local_index, global_index in enumerate(chunk_indices):
                if not bool(local_ok[local_index]):
                    continue
                predicted = flatten_rt_spectrum(reflections[local_index], transmissions[local_index]).astype(np.float32)
                target = target_spectra_np[global_index].reshape(-1)
                if predicted.shape != target.shape:
                    continue
                loss = float(np.sqrt(np.mean(np.square(predicted - target))))
                spectrum_losses[global_index] = loss
                rewards[global_index] = -loss
                ok_mask[global_index] = True

    return {
        "rewards": torch.from_numpy(rewards),
        "spectrum_losses": torch.from_numpy(spectrum_losses),
        "ok_mask": torch.from_numpy(ok_mask),
    }
