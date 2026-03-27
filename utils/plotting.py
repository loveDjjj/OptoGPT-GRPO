from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from physics.spectrum import absorption_curve, split_rt_spectrum


def _compute_absorption(reflection: Sequence[float], transmission: Sequence[float]) -> np.ndarray:
    reflection_np = np.asarray(reflection, dtype=np.float32)
    transmission_np = np.asarray(transmission, dtype=np.float32)
    return 1.0 - reflection_np - transmission_np


def _get_target_curves(target: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wavelengths = np.asarray(target["wavelengths_um"], dtype=np.float32)
    reflection = np.asarray(target["reflection"], dtype=np.float32)
    transmission = np.asarray(target["transmission"], dtype=np.float32)
    absorption = np.asarray(target.get("absorption", _compute_absorption(reflection, transmission)), dtype=np.float32)
    return wavelengths, reflection, transmission, absorption


def save_before_after_plot(
    path: str | Path,
    target: Mapping,
    before_record: Mapping,
    after_record: Mapping,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save GRPO comparison plots.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wavelengths, target_r, target_t, target_a = _get_target_curves(target)
    before_r = np.asarray(before_record["reflection"], dtype=np.float32)
    before_t = np.asarray(before_record["transmission"], dtype=np.float32)
    before_a = _compute_absorption(before_r, before_t)
    after_r = np.asarray(after_record["reflection"], dtype=np.float32)
    after_t = np.asarray(after_record["transmission"], dtype=np.float32)
    after_a = _compute_absorption(after_r, after_t)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    figure.suptitle(
        f"Target {int(target['target_id'])} | {target['family']} | "
        f"{float(target['left_nm']):.0f}-{float(target['right_nm']):.0f} nm"
    )

    curves = [
        ("Reflection", target_r, before_r, after_r),
        ("Transmission", target_t, before_t, after_t),
        ("Absorption", target_a, before_a, after_a),
    ]
    for axis, (label, target_curve, before_curve, after_curve) in zip(axes, curves):
        axis.plot(wavelengths, target_curve, label="target", linewidth=2.5, color="black")
        axis.plot(
            wavelengths,
            before_curve,
            label=f"before ({float(before_record['error']):.4f})",
            linestyle="--",
            color="#2a6f97",
        )
        axis.plot(
            wavelengths,
            after_curve,
            label=f"after ({float(after_record['error']):.4f})",
            linestyle="-",
            color="#bc4749",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Wavelength (um)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_metric_curve(
    path: str | Path,
    rows: Sequence[Mapping],
    x_key: str,
    y_key: str,
    title: str,
    y_label: str,
) -> None:
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save GRPO metric curves.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = [float(row[x_key]) for row in rows]
    y = [float(row[y_key]) for row in rows]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x, y, marker="o", linewidth=2.0, color="#2a6f97")
    axis.set_title(title)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_spectrum_comparison_plot(
    path: str | Path,
    target_spectrum: Sequence[float],
    predicted_spectrum: Sequence[float],
    wavelengths_um: Sequence[float],
    title: str,
    spectrum_loss: float | None = None,
    status: str | None = None,
) -> None:
    """保存单个样本的目标/预测光谱对比图。

    图中会同时绘制：
    - 反射率 R
    - 透射率 T
    - 吸收率 A = 1 - R - T

    这里默认输入为 OptoGPT 当前使用的拼接光谱格式 `[R..., T...]`。
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save spectrum comparison plots.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_r, target_t = split_rt_spectrum(target_spectrum)
    pred_r, pred_t = split_rt_spectrum(predicted_spectrum)
    target_a = absorption_curve(target_r, target_t)
    pred_a = absorption_curve(pred_r, pred_t)
    wavelengths = np.asarray(wavelengths_um, dtype=np.float32)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    subtitle_parts = [title]
    if spectrum_loss is not None:
        subtitle_parts.append(f"spectrum_loss={float(spectrum_loss):.6f}")
    if status is not None:
        subtitle_parts.append(f"status={status}")
    figure.suptitle(" | ".join(subtitle_parts))

    curve_rows = [
        ("Reflection", target_r, pred_r),
        ("Transmission", target_t, pred_t),
        ("Absorption", target_a, pred_a),
    ]
    for axis, (label, target_curve, pred_curve) in zip(axes, curve_rows):
        axis.plot(wavelengths, target_curve, label="target", linewidth=2.0, color="#1f1f1f")
        axis.plot(wavelengths, pred_curve, label="predicted", linewidth=1.8, color="#d62828")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Wavelength (um)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
