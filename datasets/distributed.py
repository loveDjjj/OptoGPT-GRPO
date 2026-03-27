"""数据集分布式采样辅助。"""

from __future__ import annotations

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.dist import is_distributed_enabled


def build_distributed_sampler(
    dataset: Dataset,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> DistributedSampler | None:
    """按当前分布式状态构造 sampler。"""

    if not is_distributed_enabled():
        return None
    return DistributedSampler(
        dataset,
        shuffle=shuffle,
        seed=int(seed),
        drop_last=drop_last,
    )
