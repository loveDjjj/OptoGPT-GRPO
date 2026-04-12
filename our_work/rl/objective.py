from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    target_count: int,
    group_size: int,
    mode: str = "zscore",
    eps: float = 1e-6,
) -> torch.Tensor:
    if rewards.dim() != 1:
        raise ValueError(f"rewards must be 1D, got {tuple(rewards.shape)}")
    if rewards.numel() != int(target_count) * int(group_size):
        raise ValueError("rewards length must equal target_count * group_size")
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"center", "zscore"}:
        raise ValueError(f"unsupported advantage mode: {mode}")

    grouped = rewards.view(int(target_count), int(group_size))
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if resolved_mode == "center":
        return centered.reshape(-1)

    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(eps))
    return (centered / std).reshape(-1)


def grpo_clipped_surrogate(
    current_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_epsilon: float,
) -> dict[str, torch.Tensor]:
    if current_logprob.shape != old_logprob.shape or current_logprob.shape != advantage.shape:
        raise ValueError("current_logprob, old_logprob, and advantage must have the same shape")
    if float(clip_epsilon) < 0:
        raise ValueError("clip_epsilon must be non-negative")

    log_ratio = current_logprob - old_logprob
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
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
