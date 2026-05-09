from __future__ import annotations

import torch
import torch.nn as nn


class SpectrumProjector(nn.Module):
    def __init__(self, spectrum_dim: int, prefix_length: int, hidden_size: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, prefix_length * hidden_size),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        prefix = self.proj(spectra)
        return prefix.view(spectra.size(0), self.prefix_length, self.hidden_size)
