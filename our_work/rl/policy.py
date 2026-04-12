from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class RolloutConfig:
    decode: str = "sample"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 12
    batch_size: int = 256


@dataclass
class RolloutSample:
    sample_id: str
    target_index: int
    candidate_index: int
    token_ids: list[int]
    structure_tokens: list[str]
    sequence_logprob: float
    terminated_by_eos: bool


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _apply_sampling_filter(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits
    if int(top_k) > 0:
        top_k = min(int(top_k), filtered.size(-1))
        values, _ = torch.topk(filtered, top_k, dim=-1)
        threshold = values[..., -1, None]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, float("-inf")), filtered)

    if float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > float(top_p)
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits = torch.where(remove_mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


@torch.inference_mode()
def sample_structure_rollouts(
    model,
    tokenizer,
    spectra: torch.Tensor,
    sample_ids: Sequence[str],
    *,
    group_size: int,
    config: RolloutConfig,
) -> list[RolloutSample]:
    core_model = _unwrap_model(model)
    expanded_spectra = spectra.repeat_interleave(int(group_size), dim=0)
    expanded_sample_ids: list[str] = []
    expanded_target_indices: list[int] = []
    expanded_candidate_indices: list[int] = []
    for target_index, sample_id in enumerate(sample_ids):
        expanded_sample_ids.extend([str(sample_id)] * int(group_size))
        expanded_target_indices.extend([int(target_index)] * int(group_size))
        expanded_candidate_indices.extend(range(int(group_size)))

    batch_size = int(expanded_spectra.size(0))
    input_ids = torch.full(
        (batch_size, 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=expanded_spectra.device,
    )
    sequence_logprobs = torch.zeros((batch_size,), dtype=torch.float32, device=expanded_spectra.device)
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=expanded_spectra.device)
    token_rows: list[list[int]] = [[] for _ in range(batch_size)]

    for _ in range(int(config.max_new_tokens)):
        attention_mask = torch.ones(
            (batch_size, core_model.config.prefix_length + input_ids.size(1)),
            dtype=torch.long,
            device=expanded_spectra.device,
        )
        outputs = model(
            spectra=expanded_spectra,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        next_token_logits = outputs.logits[:, -1, :]
        if float(config.temperature) != 1.0:
            next_token_logits = next_token_logits / float(config.temperature)
        filtered_logits = _apply_sampling_filter(
            next_token_logits,
            top_k=int(config.top_k),
            top_p=float(config.top_p),
        )
        log_probs = torch.log_softmax(filtered_logits, dim=-1)
        if str(config.decode).strip().lower() == "greedy":
            next_ids = torch.argmax(log_probs, dim=-1)
        else:
            probs = torch.softmax(filtered_logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        chosen_logprobs = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, tokenizer.eos_token_id), next_ids)
        chosen_logprobs = torch.where(finished, torch.zeros_like(chosen_logprobs), chosen_logprobs)
        sequence_logprobs = sequence_logprobs + chosen_logprobs
        input_ids = torch.cat([input_ids, next_ids.unsqueeze(-1)], dim=1)

        for row_idx, token_id in enumerate(next_ids.tolist()):
            if not finished[row_idx].item():
                token_rows[row_idx].append(int(token_id))
        finished = finished | next_ids.eq(int(tokenizer.eos_token_id))
        if bool(finished.all().item()):
            break

    samples: list[RolloutSample] = []
    for row_idx, token_ids in enumerate(token_rows):
        samples.append(
            RolloutSample(
                sample_id=expanded_sample_ids[row_idx],
                target_index=expanded_target_indices[row_idx],
                candidate_index=expanded_candidate_indices[row_idx],
                token_ids=list(token_ids),
                structure_tokens=tokenizer.decode(token_ids),
                sequence_logprob=float(sequence_logprobs[row_idx].item()),
                terminated_by_eos=bool(len(token_ids) > 0 and token_ids[-1] == tokenizer.eos_token_id),
            )
        )
    return samples


def batch_sequence_logprobs(
    model,
    tokenizer,
    spectra: torch.Tensor,
    token_id_groups: Sequence[Sequence[int]],
    *,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_model = _unwrap_model(model)
    if not token_id_groups:
        empty = torch.empty((0,), dtype=torch.float32, device=spectra.device)
        return empty, torch.empty((0, 0), dtype=torch.bool, device=spectra.device)

    encoded = [[int(tokenizer.bos_token_id), *[int(token_id) for token_id in token_ids]] for token_ids in token_id_groups]
    max_len = max(len(row) for row in encoded)
    all_sequence_logprobs = []
    all_token_masks = []

    for start in range(0, len(encoded), int(batch_size)):
        chunk_encoded = encoded[start : start + int(batch_size)]
        chunk_spectra = spectra[start : start + len(chunk_encoded)]
        chunk_max_len = max(len(row) for row in chunk_encoded)
        input_ids = torch.full(
            (len(chunk_encoded), chunk_max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=chunk_spectra.device,
        )
        token_mask = torch.zeros((len(chunk_encoded), chunk_max_len - 1), dtype=torch.bool, device=chunk_spectra.device)
        for row_idx, row in enumerate(chunk_encoded):
            input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=chunk_spectra.device)
            if len(row) > 1:
                token_mask[row_idx, : len(row) - 1] = True
        attention_mask = torch.cat(
            [
                torch.ones((len(chunk_encoded), core_model.config.prefix_length), dtype=torch.long, device=chunk_spectra.device),
                input_ids.ne(tokenizer.pad_token_id).long(),
            ],
            dim=1,
        )
        outputs = model(
            spectra=chunk_spectra,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        log_probs = outputs.logits.log_softmax(dim=-1)
        gathered = log_probs[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        gathered = torch.where(token_mask, gathered, torch.zeros_like(gathered))
        all_sequence_logprobs.append(gathered.sum(dim=1))
        all_token_masks.append(token_mask)

    return torch.cat(all_sequence_logprobs, dim=0), torch.cat(all_token_masks, dim=0)
