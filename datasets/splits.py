"""训练/验证数据集切分入口。"""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import Subset

from .optogpt_dataset import OptoGPTPairDataset


def build_split_datasets(config: Dict[str, Any]) -> Dict[str, OptoGPTPairDataset | Subset]:
    """按配置构造 train / val 数据集。

    当前仓库已有 `train/test` 两份 pkl。这里优先支持显式给出验证集路径；
    如果没有给出，再退回到“从训练集内部切一份验证集”。
    """

    data_cfg = config["data"]
    train_dataset = OptoGPTPairDataset(
        spectrum_path=data_cfg["train_spectrum_path"],
        structure_path=data_cfg["train_structure_path"],
        max_samples=data_cfg.get("max_train_samples"),
    )

    val_spectrum_path = data_cfg.get("val_spectrum_path")
    val_structure_path = data_cfg.get("val_structure_path")
    val_ratio = float(data_cfg.get("val_ratio", 0.0))

    if val_spectrum_path and val_structure_path:
        val_dataset: OptoGPTPairDataset | Subset = OptoGPTPairDataset(
            spectrum_path=val_spectrum_path,
            structure_path=val_structure_path,
            max_samples=data_cfg.get("max_val_samples"),
        )
    elif val_ratio > 0:
        total_count = len(train_dataset)
        split_count = max(1, int(total_count * val_ratio))
        val_indices = list(range(total_count - split_count, total_count))
        train_indices = list(range(0, total_count - split_count))
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)
    else:
        val_dataset = None

    return {
        "train": train_dataset,
        "val": val_dataset,
    }
