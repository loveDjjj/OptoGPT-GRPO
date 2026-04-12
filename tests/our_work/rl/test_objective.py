from __future__ import annotations

import torch

from our_work.rl.objective import grpo_clipped_surrogate, group_relative_advantages


def test_group_relative_advantages_returns_groupwise_zscore() -> None:
    rewards = torch.tensor([1.0, 3.0, 10.0, 14.0], dtype=torch.float32)

    advantages = group_relative_advantages(rewards, target_count=2, group_size=2, mode="zscore")

    assert advantages.shape == rewards.shape
    assert torch.allclose(advantages.view(2, 2).mean(dim=1), torch.zeros(2), atol=1e-6)


def test_grpo_clipped_surrogate_returns_ratio_and_mask() -> None:
    current = torch.tensor([0.0, -0.2], dtype=torch.float32)
    old = torch.tensor([0.0, -0.5], dtype=torch.float32)
    advantage = torch.tensor([1.0, -1.0], dtype=torch.float32)

    outputs = grpo_clipped_surrogate(current, old, advantage, clip_epsilon=0.2)

    assert set(outputs.keys()) == {"ratio", "clipped_ratio", "surrogate", "clip_mask", "approx_kl"}
    assert outputs["ratio"].shape == current.shape
