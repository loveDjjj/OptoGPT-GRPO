"""DataLoader 批处理组装函数。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def optogpt_batch_collator(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把样本列表打包成批。

    这里故意保留 `structure_tokens` 为 Python 列表，因为不同样本的结构长度不同，
    后续需要由模型词表来做编码和 padding。
    """

    if not samples:
        return {
            "sample_indices": np.empty((0,), dtype=np.int64),
            "spectra": np.empty((0, 0), dtype=np.float32),
            "structure_tokens": [],
        }

    return {
        "sample_indices": np.asarray([sample["sample_index"] for sample in samples], dtype=np.int64),
        "spectra": np.stack([np.asarray(sample["spectrum"], dtype=np.float32) for sample in samples], axis=0),
        "structure_tokens": [list(sample["structure_tokens"]) for sample in samples],
    }
