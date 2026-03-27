"""评测指标聚合工具。"""

from __future__ import annotations

from dataclasses import dataclass

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
