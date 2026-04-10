import torch

from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def test_model_forward_returns_loss_and_logits():
    config = SpectralGPTConfig(
        vocab_size=8,
        spectrum_dim=2048,
        prefix_length=4,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SpectralGPTForCausalLM(config)
    outputs = model(
        spectra=torch.randn(2, 2048),
        input_ids=torch.tensor([[1, 4, 5, 2], [1, 4, 2, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1] * 8, [1] * 7 + [0]], dtype=torch.long),
        labels=torch.tensor(
            [
                [-100, -100, -100, -100, -100, 4, 5, 2],
                [-100, -100, -100, -100, -100, 4, 2, -100],
            ],
            dtype=torch.long,
        ),
    )
    assert outputs.loss is not None
    assert outputs.logits.shape == (2, 8, 8)
