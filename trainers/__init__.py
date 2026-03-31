"""训练器模块。"""

from .grpo_trainer import GRPOTrainer
from .spectral_sft_trainer import SpectralSFTTrainer

__all__ = [
    "GRPOTrainer",
    "SpectralSFTTrainer",
]
