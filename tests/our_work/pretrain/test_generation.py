import torch

from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.generation import generate_structure_tokens
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def test_generate_structure_tokens_returns_token_lists():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    config = SpectralGPTConfig(
        vocab_size=len(tokenizer.tokens),
        spectrum_dim=2048,
        prefix_length=2,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=16,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = SpectralGPTForCausalLM(config)
    results = generate_structure_tokens(
        model=model,
        tokenizer=tokenizer,
        spectra=torch.randn(2, 2048),
        max_new_tokens=3,
    )
    assert len(results) == 2
    assert all(isinstance(item, list) for item in results)
