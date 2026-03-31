"""光谱相关的基础计算工具。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def flatten_rt(reflection: Sequence[float], transmission: Sequence[float]) -> np.ndarray:
    """把反射/透射拼接成 OptoGPT 使用的 142 维输入布局。"""

    return np.concatenate([np.asarray(reflection, dtype=np.float32), np.asarray(transmission, dtype=np.float32)])


def split_rt_spectrum(spectrum: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """把拼接后的光谱切回 R/T 两段。"""

    spectrum_np = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    if spectrum_np.size % 2 != 0:
        raise ValueError(f"光谱长度必须为偶数，当前长度为 {spectrum_np.size}。")
    half = spectrum_np.size // 2
    return spectrum_np[:half], spectrum_np[half:]


def split_rt_spectrum_torch(spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把拼接后的 torch 光谱切回 R/T 两段。"""

    if spectrum.size(-1) % 2 != 0:
        raise ValueError(f"光谱长度必须为偶数，当前长度为 {int(spectrum.size(-1))}。")
    half = spectrum.size(-1) // 2
    return spectrum[..., :half], spectrum[..., half:]


def absorption_curve(reflection: Sequence[float], transmission: Sequence[float]) -> np.ndarray:
    """按 `A = 1 - R - T` 计算吸收曲线。"""

    reflection_np = np.asarray(reflection, dtype=np.float32)
    transmission_np = np.asarray(transmission, dtype=np.float32)
    return 1.0 - reflection_np - transmission_np


def absorption_curve_torch(reflection: torch.Tensor, transmission: torch.Tensor) -> torch.Tensor:
    """按 `A = 1 - R - T` 计算 torch 吸收曲线。"""

    return 1.0 - reflection - transmission


def spectrum_error(
    predicted_spectrum: Sequence[float],
    target_spectrum: Sequence[float],
    metric: str = "rt_rmse",
) -> float:
    """计算预测光谱与目标光谱之间的误差。"""

    predicted = np.asarray(predicted_spectrum, dtype=np.float32)
    target = np.asarray(target_spectrum, dtype=np.float32)

    # 当前项目默认直接比较 R/T 两段光谱，不再把吸收率误差作为主训练目标。
    # 为了兼容历史配置，这里仍保留 absorption_rmse 别名；
    # 同时显式支持只看 R 或只看 T 的误差指标。
    if metric in {"rt_rmse", "rmse"}:
        return float(np.sqrt(np.mean(np.square(predicted - target))))
    if metric in {"rt_mae", "mae"}:
        return float(np.mean(np.abs(predicted - target)))
    if metric == "r_rmse":
        pred_r, _ = split_rt_spectrum(predicted)
        target_r, _ = split_rt_spectrum(target)
        return float(np.sqrt(np.mean(np.square(pred_r - target_r))))
    if metric == "t_rmse":
        _, pred_t = split_rt_spectrum(predicted)
        _, target_t = split_rt_spectrum(target)
        return float(np.sqrt(np.mean(np.square(pred_t - target_t))))
    if metric == "absorption_rmse":
        pred_r, pred_t = split_rt_spectrum(predicted)
        target_r, target_t = split_rt_spectrum(target)
        pred_a = absorption_curve(pred_r, pred_t)
        target_a = absorption_curve(target_r, target_t)
        return float(np.sqrt(np.mean(np.square(pred_a - target_a))))
    raise ValueError(f"不支持的误差指标: {metric}")


def spectrum_error_torch(
    predicted_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    metric: str = "rt_rmse",
) -> torch.Tensor:
    """计算 torch 版预测光谱与目标光谱之间的误差。"""

    predicted = predicted_spectrum.to(dtype=torch.float32)
    target = target_spectrum.to(device=predicted.device, dtype=torch.float32)

    if metric in {"rt_rmse", "rmse"}:
        return torch.sqrt(torch.mean((predicted - target).square(), dim=-1))
    if metric in {"rt_mae", "mae"}:
        return torch.mean(torch.abs(predicted - target), dim=-1)
    if metric == "r_rmse":
        pred_r, _ = split_rt_spectrum_torch(predicted)
        target_r, _ = split_rt_spectrum_torch(target)
        return torch.sqrt(torch.mean((pred_r - target_r).square(), dim=-1))
    if metric == "t_rmse":
        _, pred_t = split_rt_spectrum_torch(predicted)
        _, target_t = split_rt_spectrum_torch(target)
        return torch.sqrt(torch.mean((pred_t - target_t).square(), dim=-1))
    if metric == "absorption_rmse":
        pred_r, pred_t = split_rt_spectrum_torch(predicted)
        target_r, target_t = split_rt_spectrum_torch(target)
        pred_a = absorption_curve_torch(pred_r, pred_t)
        target_a = absorption_curve_torch(target_r, target_t)
        return torch.sqrt(torch.mean((pred_a - target_a).square(), dim=-1))
    raise ValueError(f"不支持的误差指标: {metric}")


def is_physical_spectrum(
    reflection: Sequence[float],
    transmission: Sequence[float],
    tolerance: float = 0.01,
) -> bool:
    """检查光谱是否满足基本物理约束。"""

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


def physical_spectrum_mask_torch(
    reflection: torch.Tensor,
    transmission: torch.Tensor,
    tolerance: float = 0.01,
) -> torch.Tensor:
    """检查一批 torch 光谱是否满足基本物理约束。"""

    finite_mask = torch.isfinite(reflection).all(dim=-1) & torch.isfinite(transmission).all(dim=-1)
    reflection_min_ok = reflection.amin(dim=-1) >= (-float(tolerance))
    reflection_max_ok = reflection.amax(dim=-1) <= (1.0 + float(tolerance))
    transmission_min_ok = transmission.amin(dim=-1) >= (-float(tolerance))
    transmission_max_ok = transmission.amax(dim=-1) <= (1.0 + float(tolerance))
    energy_ok = (reflection + transmission).amax(dim=-1) <= (1.0 + float(tolerance))
    return finite_mask & reflection_min_ok & reflection_max_ok & transmission_min_ok & transmission_max_ok & energy_ok
