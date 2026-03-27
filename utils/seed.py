"""随机种子设置工具。"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, rank_offset: int = 0) -> int:
    """设置 Python / NumPy / PyTorch 随机种子。"""

    effective_seed = int(seed) + int(rank_offset)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed
