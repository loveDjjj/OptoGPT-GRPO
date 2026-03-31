"""OptoGPT rollout/update 共用的策略分布定义。"""

from __future__ import annotations

from typing import Protocol

import torch


class PolicyConfigLike(Protocol):
    """最小策略配置协议，避免 generation/scoring 之间互相依赖具体类。"""

    decode: str
    top_k: int
    top_p: float
    temperature: float


def validate_policy_config(policy_config: PolicyConfigLike) -> None:
    """校验 rollout / scoring 共享的策略配置。"""

    if policy_config.temperature <= 0:
        raise ValueError("temperature 必须大于 0。")
    if policy_config.top_k < 0:
        raise ValueError("top_k 不能小于 0。")
    if not 0 < policy_config.top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 之间。")
    if policy_config.decode not in {"greedy", "top-k", "sample"}:
        raise ValueError(f"不支持的 decode 模式: {policy_config.decode}")


def _impossible_log_prob(reference: torch.Tensor) -> torch.Tensor:
    return torch.full_like(reference, torch.finfo(reference.dtype).min)


def _apply_top_k_top_p_filter(prob: torch.Tensor, policy_config: PolicyConfigLike) -> torch.Tensor:
    """把温度缩放后的概率分布裁剪到 top-k / top-p 支持集。"""

    filtered = prob.clone()
    vocab_size = filtered.size(-1)
    if policy_config.top_k > 0 and policy_config.top_k < vocab_size:
        topk_values = torch.topk(filtered, policy_config.top_k, dim=-1).values
        kth = topk_values[..., -1:].expand_as(filtered)
        filtered = torch.where(filtered < kth, torch.zeros_like(filtered), filtered)

    if policy_config.top_p < 1.0:
        sorted_prob, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_prob, dim=-1)
        remove_mask = cumulative > policy_config.top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_prob = torch.where(remove_mask, torch.zeros_like(sorted_prob), sorted_prob)
        filtered.zero_()
        filtered.scatter_(-1, sorted_idx, sorted_prob)

    filtered_sum = filtered.sum(dim=-1, keepdim=True)
    empty_mask = filtered_sum <= 0
    if bool(empty_mask.any().item()):
        filtered = torch.where(empty_mask, prob, filtered)
        filtered_sum = filtered.sum(dim=-1, keepdim=True)
    return filtered / filtered_sum.clamp_min(1e-12)


def policy_log_probs_from_raw_log_probs(
    raw_log_probs: torch.Tensor,
    policy_config: PolicyConfigLike,
) -> torch.Tensor:
    """把模型原始 logprob 转成 rollout/update 共用的策略 logprob。"""

    if raw_log_probs.dim() < 2:
        raise ValueError(
            "raw_log_probs 至少需要二维，最后一维为词表维度；"
            f"当前形状为 {tuple(raw_log_probs.shape)}"
        )
    validate_policy_config(policy_config)

    original_shape = raw_log_probs.shape
    flat = raw_log_probs.reshape(-1, raw_log_probs.size(-1))

    # 这里的 `raw_log_probs` 已经是 log_softmax(logits)；由于 softmax 对常数平移不敏感，
    # 对它再除 temperature 并做一次 log_softmax，与直接对原始 logits/temperature 等价。
    tempered = torch.log_softmax(flat / policy_config.temperature, dim=-1)

    if policy_config.decode == "sample":
        return tempered.reshape(original_shape)

    if policy_config.decode == "greedy":
        greedy_idx = torch.argmax(tempered, dim=-1, keepdim=True)
        greedy_log_probs = _impossible_log_prob(tempered)
        greedy_log_probs.scatter_(-1, greedy_idx, 0.0)
        return greedy_log_probs.reshape(original_shape)

    filtered_prob = _apply_top_k_top_p_filter(tempered.exp(), policy_config)
    filtered_log_probs = torch.where(
        filtered_prob > 0,
        filtered_prob.log(),
        _impossible_log_prob(filtered_prob),
    )
    return filtered_log_probs.reshape(original_shape)
