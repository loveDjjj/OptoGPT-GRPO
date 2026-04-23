import torch

from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.generation import generate_structure_tokens, score_structure_tokens
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
    model.eval()
    results = generate_structure_tokens(
        model=model,
        tokenizer=tokenizer,
        spectra=torch.randn(2, 2048),
        max_new_tokens=3,
    )
    assert len(results) == 2
    assert all(isinstance(item, list) for item in results)


def test_generate_structure_tokens_blocks_non_structural_special_tokens(monkeypatch):
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    config = SpectralGPTConfig(
        vocab_size=len(tokenizer.tokens),
        spectrum_dim=16,
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
    model.eval()
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        outputs.logits[..., tokenizer.pad_token_id] = 10.0
        outputs.logits[..., tokenizer.bos_token_id] = 9.0
        outputs.logits[..., tokenizer.unk_token_id] = 8.0
        outputs.logits[..., tokenizer.eos_token_id] = 7.0
        return outputs

    monkeypatch.setattr(model, "forward", wrapped_forward)

    results = generate_structure_tokens(
        model=model,
        tokenizer=tokenizer,
        spectra=torch.randn(1, 16),
        max_new_tokens=1,
    )

    assert results == [[]]


def test_score_structure_tokens_uses_token_positions_after_prefix():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    config = SpectralGPTConfig(
        vocab_size=len(tokenizer.tokens),
        spectrum_dim=16,
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
    model.eval()
    spectra = torch.randn(2, 16)
    input_ids = torch.tensor(
        [
            [tokenizer.bos_token_id, tokenizer.token_to_id["Ge_10"], tokenizer.eos_token_id],
            [tokenizer.bos_token_id, tokenizer.token_to_id["SiO2_20"], tokenizer.eos_token_id],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.cat(
        [
            torch.ones((2, config.prefix_length), dtype=torch.long),
            input_ids.ne(tokenizer.pad_token_id).long(),
        ],
        dim=1,
    )

    scores = score_structure_tokens(
        model=model,
        spectra=spectra,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    outputs = model(
        spectra=spectra,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    log_probs = outputs.logits.log_softmax(dim=-1)
    expected = log_probs[
        :,
        config.prefix_length : config.prefix_length + input_ids.size(1) - 1,
        :,
    ].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(scores, expected, atol=1e-6)
