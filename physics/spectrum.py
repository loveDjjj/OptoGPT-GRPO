"""光谱相关的基础计算工具。"""

from __future__ import annotations

from typing import Sequence

import numpy as np


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


def absorption_curve(reflection: Sequence[float], transmission: Sequence[float]) -> np.ndarray:
    """按 `A = 1 - R - T` 计算吸收曲线。"""

    reflection_np = np.asarray(reflection, dtype=np.float32)
    transmission_np = np.asarray(transmission, dtype=np.float32)
    return 1.0 - reflection_np - transmission_np


def spectrum_error(
    predicted_spectrum: Sequence[float],
    target_spectrum: Sequence[float],
    metric: str = "absorption_rmse",
) -> float:
    """计算预测光谱与目标光谱之间的误差。"""

    predicted = np.asarray(predicted_spectrum, dtype=np.float32)
    target = np.asarray(target_spectrum, dtype=np.float32)

    if metric == "absorption_rmse":
        pred_r, pred_t = split_rt_spectrum(predicted)
        target_r, target_t = split_rt_spectrum(target)
        pred_a = absorption_curve(pred_r, pred_t)
        target_a = absorption_curve(target_r, target_t)
        return float(np.sqrt(np.mean(np.square(pred_a - target_a))))
    if metric == "mae":
        return float(np.mean(np.abs(predicted - target)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(predicted - target))))
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
