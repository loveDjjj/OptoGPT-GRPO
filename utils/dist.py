"""分布式训练与评测辅助。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    """当前进程的分布式上下文。"""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def is_distributed_enabled() -> bool:
    """判断当前是否处于 torchrun 等分布式环境。"""

    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _choose_backend(device: torch.device) -> str:
    """按设备和平台选择通信后端。

    - Linux + CUDA: 优先 NCCL；
    - 其他情况：回退到 Gloo。
    """

    if device.type == "cuda" and os.name != "nt" and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def init_distributed(device: torch.device, timeout_minutes: int = 30) -> DistributedContext:
    """初始化当前进程的分布式状态。"""

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1 or not dist.is_available():
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return DistributedContext(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
        )

    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend=_choose_backend(device),
        init_method="env://",
        timeout=timedelta(minutes=int(timeout_minutes)),
    )
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def barrier() -> None:
    """在分布式场景下执行同步屏障。"""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    """清理分布式进程组。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, op: Literal["sum", "mean", "min", "max"] = "sum") -> torch.Tensor:
    """对张量做 all-reduce。"""

    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "mean":
        reduce_op = dist.ReduceOp.SUM
    elif op == "min":
        reduce_op = dist.ReduceOp.MIN
    elif op == "max":
        reduce_op = dist.ReduceOp.MAX
    else:
        raise ValueError(f"不支持的 reduce 操作: {op}")

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=reduce_op)
    if op == "mean":
        reduced = reduced / dist.get_world_size()
    return reduced
