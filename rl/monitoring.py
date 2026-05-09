from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


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
        ("mean_reward", "Train Mean Reward", "Reward", train_rows),
        ("valid_ratio", "Train Valid Ratio", "Ratio", train_rows),
        ("learning_rate", "Learning Rate", "LR", train_rows),
        ("grad_norm", "Grad Norm", "Norm", train_rows),
        ("mean_eval_reward", "Eval Mean Reward", "Reward", eval_rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for axis, (key, title, ylabel, rows) in zip(axes.flatten(), panels):
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
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


class RLVisualizationMonitor:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        is_main: bool,
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
        self.is_main = bool(is_main)
        self.enable_tensorboard = bool(enable_tensorboard)
        self.enable_jsonl = bool(enable_jsonl)
        self.enable_csv = bool(enable_csv)
        self.save_plots = bool(save_plots)
        self.plot_every_eval = bool(plot_every_eval)
        self.flush_secs = int(flush_secs)
        self.writer = None
        self.train_metrics_path = self.metrics_dir / "train_metrics.jsonl"
        self.eval_metrics_path = self.metrics_dir / "eval_metrics.jsonl"
        self.train_rows = _load_jsonl_rows(self.train_metrics_path)
        self.eval_rows = _load_jsonl_rows(self.eval_metrics_path)

        if not self.is_main:
            return

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_tensorboard:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            try:  # pragma: no cover - optional dependency
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir), flush_secs=self.flush_secs)
            except Exception:
                self.writer = None

    def log_train(self, row: dict) -> None:
        if not self.is_main:
            return
        self.train_rows.append(row)
        if self.enable_jsonl:
            _append_jsonl(self.train_metrics_path, row)
        if self.enable_csv:
            _rewrite_csv(self.metrics_dir / "train_metrics.csv", self.train_rows)
        if self.writer is not None:
            for key, value in row.items():
                if key in {"step", "epoch"} or not _is_finite_number(value):
                    continue
                self.writer.add_scalar(f"train/{key}", float(value), int(row["step"]))

    def log_eval(self, row: dict) -> None:
        if not self.is_main:
            return
        self.eval_rows.append(row)
        if self.enable_jsonl:
            _append_jsonl(self.eval_metrics_path, row)
        if self.enable_csv:
            _rewrite_csv(self.metrics_dir / "eval_metrics.csv", self.eval_rows)
        if self.writer is not None:
            for key, value in row.items():
                if key == "step" or not _is_finite_number(value):
                    continue
                self.writer.add_scalar(f"eval/{key.replace('eval_', '')}", float(value), int(row["step"]))
        if self.plot_every_eval:
            self.render_plots()

    def render_plots(self) -> None:
        if not self.is_main or not self.save_plots:
            return
        _save_scalar_plot(self.plots_dir / "train_loss.png", self.train_rows, metric_key="loss", title="Train Loss", ylabel="Loss")
        _save_scalar_plot(
            self.plots_dir / "train_mean_reward.png",
            self.train_rows,
            metric_key="mean_reward",
            title="Train Mean Reward",
            ylabel="Reward",
        )
        _save_scalar_plot(
            self.plots_dir / "train_valid_ratio.png",
            self.train_rows,
            metric_key="valid_ratio",
            title="Train Valid Ratio",
            ylabel="Ratio",
        )
        _save_scalar_plot(
            self.plots_dir / "learning_rate.png",
            self.train_rows,
            metric_key="learning_rate",
            title="Learning Rate",
            ylabel="LR",
        )
        _save_scalar_plot(
            self.plots_dir / "grad_norm.png",
            self.train_rows,
            metric_key="grad_norm",
            title="Grad Norm",
            ylabel="Norm",
        )
        _save_scalar_plot(
            self.plots_dir / "eval_mean_reward.png",
            self.eval_rows,
            metric_key="mean_eval_reward",
            title="Eval Mean Reward",
            ylabel="Reward",
        )
        _save_overview_plot(self.plots_dir / "overview.png", self.train_rows, self.eval_rows)

    def close(self) -> None:
        if not self.is_main:
            return
        self.render_plots()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
