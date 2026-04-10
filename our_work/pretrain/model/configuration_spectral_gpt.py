from __future__ import annotations

from transformers import PretrainedConfig


class SpectralGPTConfig(PretrainedConfig):
    model_type = "spectral_gpt"

    def __init__(
        self,
        vocab_size: int = 0,
        spectrum_dim: int = 2048,
        prefix_length: int = 8,
        n_positions: int = 32,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.spectrum_dim = spectrum_dim
        self.prefix_length = prefix_length
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
