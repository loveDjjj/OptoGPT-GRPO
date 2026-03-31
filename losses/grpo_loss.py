"""GRPO / PPO-style 目标的张量计算工具。"""

from __future__ import annotations

import torch

from .sequence_loss import masked_mean_negative_logprob


def masked_sequence_logprob(
    token_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    normalize_by_length: bool = True,
) -> torch.Tensor:
    """把 token 级 logprob 聚合成序列级 logprob。"""

    return -masked_mean_negative_logprob(
        token_logprobs=token_logprobs,
        token_mask=token_mask,
        normalize_by_length=normalize_by_length,
    )


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    target_count: int,
    group_size: int,
    mode: str = "zscore",
    eps: float = 1e-6,
) -> torch.Tensor:
    """按同一 target spectrum 的候选组计算 group-relative advantage。"""

    if rewards.dim() != 1:
        raise ValueError(f"rewards 必须是一维张量，当前形状为 {tuple(rewards.shape)}")
    if target_count <= 0:
        raise ValueError("target_count 必须大于 0。")
    if group_size <= 0:
        raise ValueError("group_size 必须大于 0。")
    if rewards.numel() != target_count * group_size:
        raise ValueError(
            "rewards 数量必须等于 target_count * group_size，"
            f"当前为 {rewards.numel()} vs {target_count} * {group_size}"
        )
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"center", "zscore"}:
        raise ValueError(f"advantage_mode 只支持 'center' 或 'zscore'，当前为 {mode!r}")

    grouped = rewards.view(target_count, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if resolved_mode == "center":
        return centered.reshape(-1)

    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    normalized = centered / std.clamp_min(float(eps))
    return normalized.reshape(-1)


def grpo_clipped_surrogate(
    current_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_epsilon: float,
) -> dict[str, torch.Tensor]:
    """计算 GRPO / PPO-style clipped surrogate 及常用统计量。"""

    if current_logprob.shape != old_logprob.shape or current_logprob.shape != advantage.shape:
        raise ValueError("current_logprob、old_logprob、advantage 的形状必须一致。")
    if clip_epsilon < 0:
        raise ValueError("clip_epsilon 不能小于 0。")

    log_ratio = current_logprob - old_logprob
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    surrogate = torch.minimum(unclipped, clipped)
    clip_mask = clipped_ratio.ne(ratio)
    approx_kl = old_logprob - current_logprob
    return {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "surrogate": surrogate,
        "clip_mask": clip_mask,
        "approx_kl": approx_kl,
    }
