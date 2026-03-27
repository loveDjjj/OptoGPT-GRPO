"""OptoGPT teacher forcing 打分工具。"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from core.transformer import subsequent_mask
from .checkpoint import OptoGPTModel


_SUBSEQUENT_MASK_CACHE: dict[tuple[str, str, int], torch.Tensor] = {}


def _cached_subsequent_mask(length: int, reference: torch.Tensor) -> torch.Tensor:
    """缓存 teacher forcing 使用的下三角 mask。"""

    cache_key = (str(reference.device), str(reference.dtype), int(length))
    cached = _SUBSEQUENT_MASK_CACHE.get(cache_key)
    if cached is None:
        cached = subsequent_mask(length).type_as(reference).to(reference.device)
        _SUBSEQUENT_MASK_CACHE[cache_key] = cached
    return cached


def _build_teacher_forcing_tensors(
    *,
    batch_sequences: Sequence[Sequence[int]],
    prompt_ids: Sequence[int],
    pad_id: int,
    max_target_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把不等长 token 序列批量整理成 teacher forcing 所需张量。

    这里刻意避免“每行单独构造一个 torch.tensor”这种高频 Python/CPU 开销，
    改成先扁平化，再用掩码一次性回填到大张量里。
    """

    current_batch = len(batch_sequences)
    prompt_len = len(prompt_ids)
    max_input_len = prompt_len + max_target_len - 1

    tgt_input = torch.full(
        (current_batch, max_input_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full(
        (current_batch, max_target_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    if current_batch == 0:
        token_mask = torch.zeros((0, max_target_len), dtype=torch.bool, device=device)
        return tgt_input, target_ids, token_mask

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    tgt_input[:, :prompt_len] = prompt_tensor

    lengths = torch.tensor([len(seq) for seq in batch_sequences], dtype=torch.long, device=device)
    token_positions = torch.arange(max_target_len, device=device).unsqueeze(0)
    token_mask = token_positions < lengths.unsqueeze(1)

    if bool(token_mask.any().item()):
        flat_targets = torch.tensor(
            [token_id for token_ids in batch_sequences for token_id in token_ids],
            dtype=torch.long,
            device=device,
        )
        target_ids[token_mask] = flat_targets

    if max_target_len > 1:
        prefix_lengths = (lengths - 1).clamp_min(0)
        prefix_positions = torch.arange(max_target_len - 1, device=device).unsqueeze(0)
        prefix_mask = prefix_positions < prefix_lengths.unsqueeze(1)
        if bool(prefix_mask.any().item()):
            flat_prefix = torch.tensor(
                [token_id for token_ids in batch_sequences for token_id in token_ids[:-1]],
                dtype=torch.long,
                device=device,
            )
            tgt_input[:, prompt_len : prompt_len + max_target_len - 1][prefix_mask] = flat_prefix

    return tgt_input, target_ids, token_mask


def sequence_logprobs_multi_target_batch_tensor(
    model: OptoGPTModel,
    target_spectra: Sequence[Sequence[float]],
    token_id_groups: Sequence[Sequence[int]],
    start_symbol: str = "BOS",
    start_mat: Optional[str] = None,
    require_grad: bool = False,
    batch_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """批量重算给定结构序列在当前模型下的 token 对数概率。

    用途：
    - 评测时计算真实结构的序列损失；
    - 训练时计算模型自己生成结构的对数概率，用于构造可反传目标。
    """

    sequences = [[int(token_id) for token_id in token_ids] for token_ids in token_id_groups]
    batch_count = len(sequences)
    if batch_count == 0:
        empty = torch.empty((0, 0), dtype=torch.float32, device=model.device)
        return empty, empty.to(dtype=torch.bool)
    if len(target_spectra) != batch_count:
        raise ValueError("target_spectra 与 token_id_groups 的长度必须一致。")

    prompt_ids = model.prompt_ids(start_symbol=start_symbol, start_mat=start_mat)
    prompt_len = len(prompt_ids)
    max_target_len = max((len(sequence) for sequence in sequences), default=0)
    if max_target_len == 0:
        empty = torch.empty((batch_count, 0), dtype=torch.float32, device=model.device)
        return empty, torch.zeros((batch_count, 0), dtype=torch.bool, device=model.device)

    pad_id = int(model.struc_word_dict.get(model.pad_token, model.struc_word_dict.get("EOS", 0)))
    effective_batch_size = int(batch_size or batch_count)
    effective_batch_size = max(1, effective_batch_size)

    all_logprobs = torch.zeros((batch_count, max_target_len), dtype=torch.float32, device=model.device)
    all_masks = torch.zeros((batch_count, max_target_len), dtype=torch.bool, device=model.device)
    grad_context = torch.enable_grad() if require_grad else torch.inference_mode()

    with grad_context:
        for start in range(0, batch_count, effective_batch_size):
            batch_sequences = sequences[start : start + effective_batch_size]
            batch_target_spectra = target_spectra[start : start + effective_batch_size]
            chunk_max_target_len = max((len(sequence) for sequence in batch_sequences), default=0)
            if chunk_max_target_len <= 0:
                continue
            src = model.targets_to_tensor_batch(batch_target_spectra)
            tgt_input, target_ids, token_mask = _build_teacher_forcing_tensors(
                batch_sequences=batch_sequences,
                prompt_ids=prompt_ids,
                pad_id=pad_id,
                max_target_len=chunk_max_target_len,
                device=model.device,
            )
            trg_mask = _cached_subsequent_mask(tgt_input.size(1), src.data)
            out = model.model(src, tgt_input, None, trg_mask)
            raw_log_probs = model.generator(out)
            gather_positions = slice(prompt_len - 1, prompt_len - 1 + chunk_max_target_len)
            relevant_log_probs = raw_log_probs[:, gather_positions, :]
            gathered = relevant_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            stop = start + len(batch_sequences)
            all_logprobs[start:stop, :chunk_max_target_len] = gathered
            all_masks[start:stop, :chunk_max_target_len] = token_mask

    return all_logprobs, all_masks
