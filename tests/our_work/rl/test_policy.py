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
    model.eval()

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


def test_sample_structure_rollouts_respects_rollout_batch_size(monkeypatch) -> None:
    model, tokenizer, spectra = _tiny_components()
    model.eval()
    call_sizes: list[int] = []
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        call_sizes.append(int(kwargs["spectra"].shape[0]))
        outputs = original_forward(*args, **kwargs)
        outputs.logits[..., tokenizer.eos_token_id] = float("-inf")
        return outputs

    monkeypatch.setattr(model, "forward", wrapped_forward)

    samples = sample_structure_rollouts(
        model,
        tokenizer,
        spectra,
        ["sample-0", "sample-1"],
        group_size=3,
        config=RolloutConfig(decode="greedy", max_new_tokens=2, batch_size=2),
    )

    assert len(samples) == 6
    assert all(size <= 2 for size in call_sizes)
    assert len(call_sizes) == 6


def test_sample_structure_rollouts_blocks_non_structural_special_tokens(monkeypatch) -> None:
    model, tokenizer, spectra = _tiny_components()
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

    samples = sample_structure_rollouts(
        model,
        tokenizer,
        spectra[:1],
        ["sample-0"],
        group_size=2,
        config=RolloutConfig(decode="greedy", max_new_tokens=1, batch_size=2),
    )

    assert [sample.token_ids for sample in samples] == [[tokenizer.eos_token_id], [tokenizer.eos_token_id]]


def test_batch_sequence_logprobs_returns_one_score_per_sequence() -> None:
    model, tokenizer, spectra = _tiny_components()
    model.eval()
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


def test_batch_sequence_logprobs_matches_rollout_logprobs() -> None:
    torch.manual_seed(2)
    model, tokenizer, spectra = _tiny_components()
    model.eval()

    rollout_samples = sample_structure_rollouts(
        model,
        tokenizer,
        spectra,
        ["sample-0", "sample-1"],
        group_size=2,
        config=RolloutConfig(decode="greedy", max_new_tokens=4, batch_size=4),
    )

    sequence_logprobs, _ = batch_sequence_logprobs(
        model,
        tokenizer,
        spectra.repeat_interleave(2, dim=0),
        [sample.token_ids for sample in rollout_samples],
        batch_size=4,
    )

    expected = torch.tensor(
        [sample.sequence_logprob for sample in rollout_samples],
        dtype=torch.float32,
    )
    assert torch.allclose(sequence_logprobs, expected, atol=1e-6)


def test_batch_sequence_logprobs_handles_token_ids_equal_to_pad_token() -> None:
    model, tokenizer, spectra = _tiny_components()
    model.eval()

    token_id_groups = [[tokenizer.pad_token_id, tokenizer.token_to_id["SiO2_20"], tokenizer.eos_token_id]]
    sequence_logprobs, token_mask = batch_sequence_logprobs(
        model,
        tokenizer,
        spectra[:1],
        token_id_groups,
        batch_size=1,
        suppress_non_structural_tokens=False,
    )

    input_ids = torch.tensor([[tokenizer.bos_token_id, *token_id_groups[0]]], dtype=torch.long)
    attention_mask = torch.cat(
        [
            torch.ones((1, model.config.prefix_length), dtype=torch.long),
            torch.ones_like(input_ids),
        ],
        dim=1,
    )
    outputs = model(
        spectra=spectra[:1],
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    manual = (
        outputs.logits.log_softmax(dim=-1)[:, model.config.prefix_length : model.config.prefix_length + input_ids.size(1) - 1, :]
        .gather(-1, input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
        .sum(dim=1)
    )

    assert token_mask.tolist() == [[True, True, True]]
    assert torch.allclose(sequence_logprobs, manual, atol=1e-6)
