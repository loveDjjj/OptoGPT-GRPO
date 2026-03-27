"""序列损失计算工具。"""

from __future__ import annotations

import torch


def masked_mean_negative_logprob(
    token_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    normalize_by_length: bool = True,
) -> torch.Tensor:
    """根据 teacher forcing 的 token 对数概率计算样本级序列损失。

    返回值形状为 `[batch]`，便于后续：
    - 评测时直接统计均值；
    - 训练时与光谱损失组合。
    """

    if token_logprobs.shape != token_mask.shape:
        raise ValueError("token_logprobs 与 token_mask 的形状必须一致。")

    token_mask_float = token_mask.to(dtype=token_logprobs.dtype)
    negative_logprob = -token_logprobs * token_mask_float
    token_count = token_mask.sum(dim=-1).clamp_min(1)
    sequence_loss = negative_logprob.sum(dim=-1)
    if normalize_by_length:
        sequence_loss = sequence_loss / token_count.to(dtype=sequence_loss.dtype)
    return sequence_loss
