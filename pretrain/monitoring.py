from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch
from transformers import TrainerCallback

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _rewrite_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_scalar_plot(path: Path, rows: Iterable[dict], *, metric_key: str, title: str, ylabel: str) -> None:
    if plt is None:
        return
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if not _is_finite_number(row.get(metric_key)):
            continue
        xs.append(float(row.get("step", 0.0)))
        ys.append(float(row[metric_key]))
    if not xs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, linewidth=2.0, color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("Global Step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_overview_plot(path: Path, train_rows: list[dict], eval_rows: list[dict]) -> None:
    if plt is None:
        return
    panels = [
        ("loss", "Train Loss", "Loss", train_rows),
        ("grad_norm", "Grad Norm", "Grad Norm", train_rows),
        ("learning_rate", "Learning Rate", "LR", train_rows),
        ("eval_loss", "Eval Loss", "Loss", eval_rows),
        ("eval_token_accuracy", "Eval Token Accuracy", "Accuracy", eval_rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    flat_axes = axes.flatten()
    for axis, (key, title, ylabel, rows) in zip(flat_axes, panels):
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if not _is_finite_number(row.get(key)):
                continue
            xs.append(float(row.get("step", 0.0)))
            ys.append(float(row[key]))
        if xs:
            axis.plot(xs, ys, linewidth=2.0, color="#2a6f97")
            axis.grid(alpha=0.3)
        else:
            axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("Global Step")
        axis.set_ylabel(ylabel)
    if len(flat_axes) > len(panels):
        for axis in flat_axes[len(panels) :]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


class PretrainVisualizationCallback(TrainerCallback):
    def __init__(
        self,
        *,
        output_dir: str | Path,
        enable_tensorboard: bool = True,
        enable_jsonl: bool = True,
        enable_csv: bool = True,
        save_plots: bool = True,
        plot_every_eval: bool = True,
        flush_secs: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.enable_tensorboard = bool(enable_tensorboard)
        self.enable_jsonl = bool(enable_jsonl)
        self.enable_csv = bool(enable_csv)
        self.save_plots = bool(save_plots)
        self.plot_every_eval = bool(plot_every_eval)
        self.flush_secs = int(flush_secs)
        self.writer = None
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []
        self._last_train_log_time: float | None = None
        self._last_train_log_step: int | None = None

    def _prepare_dirs(self) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.metrics_dir / "train_metrics.jsonl",
            self.metrics_dir / "eval_metrics.jsonl",
            self.metrics_dir / "train_metrics.csv",
            self.metrics_dir / "eval_metrics.csv",
        ):
            if path.exists():
                path.unlink()

    def _write_train_row(self, row: dict) -> None:
        self.train_rows.append(row)
        if self.enable_jsonl:
            _append_jsonl(self.metrics_dir / "train_metrics.jsonl", row)
        if self.enable_csv:
            _rewrite_csv(self.metrics_dir / "train_metrics.csv", self.train_rows)

    def _write_eval_row(self, row: dict) -> None:
        self.eval_rows.append(row)
        if self.enable_jsonl:
            _append_jsonl(self.metrics_dir / "eval_metrics.jsonl", row)
        if self.enable_csv:
            _rewrite_csv(self.metrics_dir / "eval_metrics.csv", self.eval_rows)

    def _render_plots(self) -> None:
        if not self.save_plots:
            return
        _save_scalar_plot(self.plots_dir / "train_loss.png", self.train_rows, metric_key="loss", title="Train Loss", ylabel="Loss")
        _save_scalar_plot(self.plots_dir / "learning_rate.png", self.train_rows, metric_key="learning_rate", title="Learning Rate", ylabel="LR")
        _save_scalar_plot(self.plots_dir / "grad_norm.png", self.train_rows, metric_key="grad_norm", title="Grad Norm", ylabel="Grad Norm")
        _save_scalar_plot(self.plots_dir / "eval_loss.png", self.eval_rows, metric_key="eval_loss", title="Eval Loss", ylabel="Loss")
        _save_scalar_plot(
            self.plots_dir / "eval_token_accuracy.png",
            self.eval_rows,
            metric_key="eval_token_accuracy",
            title="Eval Token Accuracy",
            ylabel="Accuracy",
        )
        _save_overview_plot(self.plots_dir / "overview.png", self.train_rows, self.eval_rows)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._prepare_dirs()
        if self.enable_tensorboard:
            try:  # pragma: no cover - optional dependency
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir), flush_secs=self.flush_secs)
            except Exception:
                self.writer = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        step = int(state.global_step)
        row = {"step": step, "epoch": float(logs["epoch"]) if _is_finite_number(logs.get("epoch")) else None}
        for key in ("loss", "grad_norm", "learning_rate"):
            value = logs.get(key)
            if _is_finite_number(value):
                row[key] = float(value)

        now = time.time()
        if self._last_train_log_time is not None and self._last_train_log_step is not None and step > self._last_train_log_step:
            delta_t = max(now - self._last_train_log_time, 1e-6)
            delta_steps = step - self._last_train_log_step
            steps_per_second = float(delta_steps) / delta_t
            samples_per_second = (
                steps_per_second
                * float(args.per_device_train_batch_size)
                * float(args.gradient_accumulation_steps)
                * float(getattr(args, "world_size", 1))
            )
            row["steps_per_second"] = steps_per_second
            row["samples_per_second"] = samples_per_second
        self._last_train_log_time = now
        self._last_train_log_step = step

        if torch.cuda.is_available():
            row["cuda_memory_gb"] = float(torch.cuda.memory_allocated() / (1024**3))

        self._write_train_row(row)
        if self.writer is not None:
            for key, value in row.items():
                if key == "step" or not _is_finite_number(value):
                    continue
                self.writer.add_scalar(f"train/{key}", float(value), step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or not metrics:
            return
        step = int(state.global_step)
        row = {"step": step}
        for key, value in metrics.items():
            if _is_finite_number(value):
                row[key] = float(value)
        self._write_eval_row(row)
        if self.writer is not None:
            for key, value in row.items():
                if key == "step" or not _is_finite_number(value):
                    continue
                self.writer.add_scalar(f"eval/{key.replace('eval_', '')}", float(value), step)
        if self.plot_every_eval:
            self._render_plots()

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._render_plots()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
