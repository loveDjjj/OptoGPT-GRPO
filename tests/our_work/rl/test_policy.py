from __future__ import annotations

import torch

from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.rl.policy import RolloutConfig, batch_sequence_logprobs, sample_structure_rollouts


def _tiny_components():
    tokenizer = SpectralStructureTokenizer(tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"])
    config = SpectralGPTConfig(
        vocab_size=len(tokenizer.tokens),
        spectrum_dim=16,
        prefix_length=2,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = SpectralGPTForCausalLM(config)
    spectra = torch.zeros((2, 16), dtype=torch.float32)
    return model, tokenizer, spectra


def test_sample_structure_rollouts_returns_grouped_samples() -> None:
    model, tokenizer, spectra = _tiny_components()

    samples = sample_structure_rollouts(
        model,
        tokenizer,
        spectra,
        ["sample-0", "sample-1"],
        group_size=2,
        config=RolloutConfig(decode="greedy", max_new_tokens=3, batch_size=4),
    )

    assert len(samples) == 4
    assert {sample.sample_id for sample in samples} == {"sample-0", "sample-1"}


def test_batch_sequence_logprobs_returns_one_score_per_sequence() -> None:
    model, tokenizer, spectra = _tiny_components()
    token_id_groups = [
        [tokenizer.token_to_id["Ge_10"], tokenizer.eos_token_id],
        [tokenizer.token_to_id["SiO2_20"], tokenizer.eos_token_id],
    ]

    sequence_logprobs, token_mask = batch_sequence_logprobs(
        model,
        tokenizer,
        spectra,
        token_id_groups,
        batch_size=2,
    )

    assert sequence_logprobs.shape == (2,)
    assert token_mask.shape[0] == 2
