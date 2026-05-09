from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, rank_offset: int = 0) -> None:
    resolved_seed = int(seed) + int(rank_offset)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
