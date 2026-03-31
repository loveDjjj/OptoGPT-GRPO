"""训练/验证数据集切分入口。"""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import Subset

from .optogpt_dataset import OptoGPTPairDataset


def build_split_datasets(config: Dict[str, Any]) -> Dict[str, OptoGPTPairDataset | Subset]:
    """按配置构造 train / val 数据集。

    优先读取显式给出的 train/test(or val) 文件；
    如果没有给出验证集路径，再退回到“从训练集内部切一份验证集”。
    """

    data_cfg = config["data"]
    # spectral GRPO 训练阶段只依赖目标光谱，不依赖真值结构。
    # 当该开关打开时，可以避免把超大的 Structure_train.npy
    # 在每个 DDP rank 中各加载一份。
    skip_train_structure_loading = bool(data_cfg.get("skip_train_structure_loading", False))

    train_dataset = OptoGPTPairDataset(
        spectrum_path=data_cfg["train_spectrum_path"],
        structure_path=None if skip_train_structure_loading else data_cfg["train_structure_path"],
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
