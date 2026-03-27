"""评测指标聚合工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.dist import reduce_tensor


@dataclass
class MetricAccumulator:
    """流式统计评测指标。"""

    sample_count: int = 0
    valid_structure_count: int = 0
    sequence_loss_sum: float = 0.0
    spectrum_loss_sum: float = 0.0
    min_sequence_loss: float = float("inf")
    max_sequence_loss: float = float("-inf")
    min_spectrum_loss: float = float("inf")
    max_spectrum_loss: float = float("-inf")

    def update(self, sequence_loss: float, spectrum_loss: float, status: str) -> None:
        self.sample_count += 1
        self.sequence_loss_sum += float(sequence_loss)
        self.spectrum_loss_sum += float(spectrum_loss)
        self.min_sequence_loss = min(self.min_sequence_loss, float(sequence_loss))
        self.max_sequence_loss = max(self.max_sequence_loss, float(sequence_loss))
        self.min_spectrum_loss = min(self.min_spectrum_loss, float(spectrum_loss))
        self.max_spectrum_loss = max(self.max_spectrum_loss, float(spectrum_loss))
        if status == "ok":
            self.valid_structure_count += 1

    def update_batch(
        self,
        sequence_losses: np.ndarray,
        spectrum_losses: np.ndarray,
        ok_mask: np.ndarray,
    ) -> None:
        """批量更新指标，减少 Python 逐样本循环开销。"""

        if sequence_losses.size == 0:
            return
        sequence_losses = np.asarray(sequence_losses, dtype=np.float64).reshape(-1)
        spectrum_losses = np.asarray(spectrum_losses, dtype=np.float64).reshape(-1)
        ok_mask = np.asarray(ok_mask, dtype=np.bool_).reshape(-1)

        self.sample_count += int(sequence_losses.size)
        self.valid_structure_count += int(ok_mask.sum())
        self.sequence_loss_sum += float(sequence_losses.sum())
        self.spectrum_loss_sum += float(spectrum_losses.sum())
        self.min_sequence_loss = min(self.min_sequence_loss, float(sequence_losses.min()))
        self.max_sequence_loss = max(self.max_sequence_loss, float(sequence_losses.max()))
        self.min_spectrum_loss = min(self.min_spectrum_loss, float(spectrum_losses.min()))
        self.max_spectrum_loss = max(self.max_spectrum_loss, float(spectrum_losses.max()))

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                float(self.sample_count),
                float(self.valid_structure_count),
                float(self.sequence_loss_sum),
                float(self.spectrum_loss_sum),
                float(self.min_sequence_loss if self.sample_count > 0 else 0.0),
                float(self.max_sequence_loss if self.sample_count > 0 else 0.0),
                float(self.min_spectrum_loss if self.sample_count > 0 else 0.0),
                float(self.max_spectrum_loss if self.sample_count > 0 else 0.0),
            ],
            dtype=torch.float64,
            device=device,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "MetricAccumulator":
        values = tensor.detach().cpu().tolist()
        return cls(
            sample_count=int(values[0]),
            valid_structure_count=int(values[1]),
            sequence_loss_sum=float(values[2]),
            spectrum_loss_sum=float(values[3]),
            min_sequence_loss=float(values[4]),
            max_sequence_loss=float(values[5]),
            min_spectrum_loss=float(values[6]),
            max_spectrum_loss=float(values[7]),
        )

    def to_summary_row(self, split: str, checkpoint_path: str) -> dict:
        mean_sequence_loss = self.sequence_loss_sum / self.sample_count if self.sample_count > 0 else float("nan")
        mean_spectrum_loss = self.spectrum_loss_sum / self.sample_count if self.sample_count > 0 else float("nan")
        valid_ratio = self.valid_structure_count / self.sample_count if self.sample_count > 0 else 0.0
        return {
            "split": split,
            "checkpoint_path": checkpoint_path,
            "sample_count": int(self.sample_count),
            "valid_structure_count": int(self.valid_structure_count),
            "valid_structure_ratio": float(valid_ratio),
            "mean_sequence_loss": float(mean_sequence_loss),
            "mean_spectrum_loss": float(mean_spectrum_loss),
            "min_sequence_loss": float(self.min_sequence_loss if self.sample_count > 0 else float("nan")),
            "max_sequence_loss": float(self.max_sequence_loss if self.sample_count > 0 else float("nan")),
            "min_spectrum_loss": float(self.min_spectrum_loss if self.sample_count > 0 else float("nan")),
            "max_spectrum_loss": float(self.max_spectrum_loss if self.sample_count > 0 else float("nan")),
        }


def reduce_metric_accumulator(accumulator: MetricAccumulator, device: torch.device) -> MetricAccumulator:
    """把各 rank 的局部统计归并成全局统计。"""

    if accumulator.sample_count == 0:
        local_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
    else:
        local_tensor = accumulator.to_tensor(device)
    # sum 项
    sum_tensor = local_tensor[:4]
    # 极值项单独 reduce，避免把空 rank 干扰到最值结果。
    min_tensor = local_tensor[4:5]
    max_tensor = local_tensor[5:6]
    min_spec_tensor = local_tensor[6:7]
    max_spec_tensor = local_tensor[7:8]

    reduced_sum = reduce_tensor(sum_tensor, op="sum")

    if accumulator.sample_count == 0:
        min_tensor = torch.tensor([float("inf")], dtype=torch.float64, device=device)
        max_tensor = torch.tensor([float("-inf")], dtype=torch.float64, device=device)
        min_spec_tensor = torch.tensor([float("inf")], dtype=torch.float64, device=device)
        max_spec_tensor = torch.tensor([float("-inf")], dtype=torch.float64, device=device)

    reduced_min = reduce_tensor(min_tensor, op="min")
    reduced_max = reduce_tensor(max_tensor, op="max")
    reduced_min_spec = reduce_tensor(min_spec_tensor, op="min")
    reduced_max_spec = reduce_tensor(max_spec_tensor, op="max")

    merged = torch.cat([reduced_sum, reduced_min, reduced_max, reduced_min_spec, reduced_max_spec], dim=0)
    return MetricAccumulator.from_tensor(merged)


@dataclass
class DistributionPlotAccumulator:
    """用于快速累计统计图所需的直方图/热力图计数。

    这里不保存全量样本，只保存已经离散化后的计数矩阵：
    - R-RMSE 直方图
    - T-RMSE 直方图
    - 序列误差直方图
    - 生成长度 vs 真值长度 2D 热力图

    这样在全量评测场景下开销很小，也更适合多卡并行后做 all-reduce 汇总。
    """

    rt_rmse_bins: int
    rt_rmse_max: float
    sequence_loss_bins: int
    sequence_loss_max: float
    length_max: int

    def __post_init__(self) -> None:
        self.r_rmse_hist = np.zeros((self.rt_rmse_bins,), dtype=np.int64)
        self.t_rmse_hist = np.zeros((self.rt_rmse_bins,), dtype=np.int64)
        self.sequence_loss_hist = np.zeros((self.sequence_loss_bins,), dtype=np.int64)
        # 最后一格同时承接所有 >= length_max 的样本，避免出现越界。
        self.length_heatmap = np.zeros((self.length_max + 1, self.length_max + 1), dtype=np.int64)

    def _bin_index(self, value: float, bin_count: int, max_value: float) -> int:
        if not np.isfinite(value) or max_value <= 0:
            return bin_count - 1
        clipped = min(max(float(value), 0.0), float(max_value))
        ratio = clipped / float(max_value)
        index = int(ratio * bin_count)
        return min(max(index, 0), bin_count - 1)

    def _clip_length(self, value: int) -> int:
        return min(max(int(value), 0), self.length_max)

    def update(
        self,
        r_rmse: float,
        t_rmse: float,
        sequence_loss: float,
        generated_length: int,
        target_length: int,
    ) -> None:
        self.r_rmse_hist[self._bin_index(r_rmse, self.rt_rmse_bins, self.rt_rmse_max)] += 1
        self.t_rmse_hist[self._bin_index(t_rmse, self.rt_rmse_bins, self.rt_rmse_max)] += 1
        self.sequence_loss_hist[
            self._bin_index(sequence_loss, self.sequence_loss_bins, self.sequence_loss_max)
        ] += 1
        target_idx = self._clip_length(target_length)
        generated_idx = self._clip_length(generated_length)
        self.length_heatmap[target_idx, generated_idx] += 1

    def update_batch(
        self,
        r_rmse: np.ndarray,
        t_rmse: np.ndarray,
        sequence_loss: np.ndarray,
        generated_length: np.ndarray,
        target_length: np.ndarray,
    ) -> None:
        """批量累计直方图/热力图计数。"""

        r_rmse = np.asarray(r_rmse, dtype=np.float64).reshape(-1)
        t_rmse = np.asarray(t_rmse, dtype=np.float64).reshape(-1)
        sequence_loss = np.asarray(sequence_loss, dtype=np.float64).reshape(-1)
        generated_length = np.asarray(generated_length, dtype=np.int64).reshape(-1)
        target_length = np.asarray(target_length, dtype=np.int64).reshape(-1)

        r_indices = np.asarray(
            [self._bin_index(value, self.rt_rmse_bins, self.rt_rmse_max) for value in r_rmse],
            dtype=np.int64,
        )
        t_indices = np.asarray(
            [self._bin_index(value, self.rt_rmse_bins, self.rt_rmse_max) for value in t_rmse],
            dtype=np.int64,
        )
        seq_indices = np.asarray(
            [self._bin_index(value, self.sequence_loss_bins, self.sequence_loss_max) for value in sequence_loss],
            dtype=np.int64,
        )
        target_indices = np.clip(target_length, 0, self.length_max)
        generated_indices = np.clip(generated_length, 0, self.length_max)

        np.add.at(self.r_rmse_hist, r_indices, 1)
        np.add.at(self.t_rmse_hist, t_indices, 1)
        np.add.at(self.sequence_loss_hist, seq_indices, 1)
        np.add.at(self.length_heatmap, (target_indices, generated_indices), 1)

    def to_device_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "r_rmse_hist": torch.as_tensor(self.r_rmse_hist, dtype=torch.float64, device=device),
            "t_rmse_hist": torch.as_tensor(self.t_rmse_hist, dtype=torch.float64, device=device),
            "sequence_loss_hist": torch.as_tensor(self.sequence_loss_hist, dtype=torch.float64, device=device),
            "length_heatmap": torch.as_tensor(self.length_heatmap, dtype=torch.float64, device=device),
        }

    @classmethod
    def from_device_tensors(
        cls,
        rt_rmse_bins: int,
        rt_rmse_max: float,
        sequence_loss_bins: int,
        sequence_loss_max: float,
        length_max: int,
        tensors: dict[str, torch.Tensor],
    ) -> "DistributionPlotAccumulator":
        accumulator = cls(
            rt_rmse_bins=rt_rmse_bins,
            rt_rmse_max=rt_rmse_max,
            sequence_loss_bins=sequence_loss_bins,
            sequence_loss_max=sequence_loss_max,
            length_max=length_max,
        )
        accumulator.r_rmse_hist = tensors["r_rmse_hist"].detach().cpu().numpy().astype(np.int64, copy=False)
        accumulator.t_rmse_hist = tensors["t_rmse_hist"].detach().cpu().numpy().astype(np.int64, copy=False)
        accumulator.sequence_loss_hist = tensors["sequence_loss_hist"].detach().cpu().numpy().astype(np.int64, copy=False)
        accumulator.length_heatmap = tensors["length_heatmap"].detach().cpu().numpy().astype(np.int64, copy=False)
        return accumulator


def reduce_distribution_plot_accumulator(
    accumulator: DistributionPlotAccumulator,
    device: torch.device,
) -> DistributionPlotAccumulator:
    """将各 rank 的统计图计数做 all-reduce 汇总。"""

    local_tensors = accumulator.to_device_tensors(device)
    reduced_tensors = {
        key: reduce_tensor(value, op="sum")
        for key, value in local_tensors.items()
    }
    return DistributionPlotAccumulator.from_device_tensors(
        rt_rmse_bins=accumulator.rt_rmse_bins,
        rt_rmse_max=accumulator.rt_rmse_max,
        sequence_loss_bins=accumulator.sequence_loss_bins,
        sequence_loss_max=accumulator.sequence_loss_max,
        length_max=accumulator.length_max,
        tensors=reduced_tensors,
    )
