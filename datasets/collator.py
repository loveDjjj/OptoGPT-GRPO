"""DataLoader 批处理组装函数。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


def optogpt_batch_collator(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把样本列表打包成批。

    这里故意保留 `structure_tokens` 为 Python 列表，因为不同样本的结构长度不同，
    后续需要由模型词表来做编码和 padding。
    """

    if not samples:
        return {
            "sample_indices": torch.empty((0,), dtype=torch.long),
            "spectra": torch.empty((0, 0), dtype=torch.float32),
            "structure_tokens": [],
        }

    # 这里直接在 collator 中把数值数据转成 torch.Tensor，
    # 这样 DataLoader 的 pin_memory 才能真正发挥作用，
    # 后续拷贝到 GPU 时也可以配合 non_blocking=True 降低等待时间。
    return {
        "sample_indices": torch.as_tensor(
            np.asarray([sample["sample_index"] for sample in samples], dtype=np.int64),
            dtype=torch.long,
        ),
        "spectra": torch.as_tensor(
            np.stack([np.asarray(sample["spectrum"], dtype=np.float32) for sample in samples], axis=0),
            dtype=torch.float32,
        ),
        "structure_tokens": [list(sample["structure_tokens"]) for sample in samples],
    }
