"""兼容入口：spectral_sft_trainer 已迁移到 grpo_trainer。"""

from __future__ import annotations

from .grpo_trainer import GRPOTrainer

SpectralSFTTrainer = GRPOTrainer

__all__ = ["SpectralSFTTrainer", "GRPOTrainer"]
