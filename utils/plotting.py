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


def save_eval_distribution_summary(
    path: str | Path,
    split_name: str,
    r_rmse_hist: Sequence[float],
    t_rmse_hist: Sequence[float],
    sequence_loss_hist: Sequence[float],
    length_heatmap: Sequence[Sequence[float]],
    rt_rmse_max: float,
    sequence_loss_max: float,
    length_max: int,
) -> None:
    """保存评测统计图总览。

    图像布局：
    1. R-RMSE 直方图
    2. T-RMSE 直方图
    3. 生成长度 vs 真值长度 2D 热力图
    4. 序列误差直方图

    这里直接基于已经累计好的计数作图，避免再次扫描样本。
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save evaluation distribution plots.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r_rmse_hist = np.asarray(r_rmse_hist, dtype=np.float64)
    t_rmse_hist = np.asarray(t_rmse_hist, dtype=np.float64)
    sequence_loss_hist = np.asarray(sequence_loss_hist, dtype=np.float64)
    length_heatmap = np.asarray(length_heatmap, dtype=np.float64)

    rt_edges = np.linspace(0.0, float(rt_rmse_max), int(r_rmse_hist.size) + 1, dtype=np.float64)
    seq_edges = np.linspace(0.0, float(sequence_loss_max), int(sequence_loss_hist.size) + 1, dtype=np.float64)
    rt_centers = 0.5 * (rt_edges[:-1] + rt_edges[1:])
    seq_centers = 0.5 * (seq_edges[:-1] + seq_edges[1:])
    rt_width = rt_edges[1] - rt_edges[0]
    seq_width = seq_edges[1] - seq_edges[0]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle(f"Evaluation Distribution Summary | split={split_name}")

    axes[0, 0].bar(rt_centers, r_rmse_hist, width=rt_width * 0.95, color="#4C78A8", alpha=0.9)
    axes[0, 0].set_title("R-RMSE vs Count")
    axes[0, 0].set_xlabel("R-RMSE")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].bar(rt_centers, t_rmse_hist, width=rt_width * 0.95, color="#F58518", alpha=0.9)
    axes[0, 1].set_title("T-RMSE vs Count")
    axes[0, 1].set_xlabel("T-RMSE")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(alpha=0.2)

    heatmap = axes[1, 0].imshow(
        length_heatmap,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap="Greys",
        extent=[0, length_max, 0, length_max],
    )
    axes[1, 0].set_title("Generated Length vs Target Length")
    axes[1, 0].set_xlabel("Generated Length")
    axes[1, 0].set_ylabel("Target Length")
    cbar = figure.colorbar(heatmap, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    axes[1, 1].bar(seq_centers, sequence_loss_hist, width=seq_width * 0.95, color="#54A24B", alpha=0.9)
    axes[1, 1].set_title("Sequence Loss vs Count")
    axes[1, 1].set_xlabel("Sequence Loss")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_grpo_epoch_summary_plot(
    path: str | Path,
    rows: Sequence[Mapping],
) -> None:
    """保存 spectral GRPO 的 epoch 级训练/验证曲线总览。"""

    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save spectral GRPO epoch plots.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.asarray([float(row["epoch"]) for row in rows], dtype=np.float32)

    def _series(key: str) -> np.ndarray:
        values = []
        for row in rows:
            value = row.get(key, float("nan"))
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(float("nan"))
        return np.asarray(values, dtype=np.float32)

    train_r = _series("mean_train_r_rmse")
    train_t = _series("mean_train_t_rmse")
    train_seq = _series("mean_train_sequence_loss")
    train_spec = _series("mean_train_spectrum_loss")
    val_r = _series("val_r_rmse")
    val_t = _series("val_t_rmse")
    val_seq = _series("val_sequence_loss")
    val_spec = _series("val_spectrum_loss")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    figure.suptitle("Spectral GRPO Epoch Summary")

    curve_specs = [
        (axes[0, 0], "R-RMSE vs Epoch", "R-RMSE", train_r, val_r),
        (axes[0, 1], "T-RMSE vs Epoch", "T-RMSE", train_t, val_t),
        (axes[1, 0], "Sequence Loss vs Epoch", "Sequence Loss", train_seq, val_seq),
        (axes[1, 1], "Spectrum Loss vs Epoch", "Spectrum Loss", train_spec, val_spec),
    ]

    for axis, title, y_label, train_values, val_values in curve_specs:
        axis.plot(epochs, train_values, marker="o", linewidth=2.0, color="#2a6f97", label="train")
        if np.isfinite(val_values).any():
            axis.plot(epochs, val_values, marker="s", linewidth=2.0, color="#bc4749", label="val")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_sft_epoch_summary_plot(
    path: str | Path,
    rows: Sequence[Mapping],
) -> None:
    """兼容旧名称：内部转发到 GRPO 版本。"""

    save_grpo_epoch_summary_plot(path=path, rows=rows)
