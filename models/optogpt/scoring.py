"""OptoGPT teacher forcing 打分工具。"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.autograd import Variable

from core.transformer import subsequent_mask
from .checkpoint import OptoGPTModel


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

    max_input_len = prompt_len + max_target_len - 1
    pad_id = int(model.struc_word_dict.get(model.pad_token, model.struc_word_dict.get("EOS", 0)))
    effective_batch_size = int(batch_size or batch_count)
    effective_batch_size = max(1, effective_batch_size)

    all_logprobs: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()

    with grad_context:
        for start in range(0, batch_count, effective_batch_size):
            batch_sequences = sequences[start : start + effective_batch_size]
            batch_target_spectra = target_spectra[start : start + effective_batch_size]
            current_batch = len(batch_sequences)
            src = model.targets_to_tensor_batch(batch_target_spectra)
            tgt_input = torch.full(
                (current_batch, max_input_len),
                pad_id,
                dtype=torch.long,
                device=model.device,
            )
            target_ids = torch.full(
                (current_batch, max_target_len),
                pad_id,
                dtype=torch.long,
                device=model.device,
            )
            token_mask = torch.zeros(
                (current_batch, max_target_len),
                dtype=torch.bool,
                device=model.device,
            )

            for row_idx, token_id_list in enumerate(batch_sequences):
                prefix_ids = prompt_ids + token_id_list[:-1]
                prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=model.device)
                tgt_input[row_idx, : prefix_tensor.numel()] = prefix_tensor
                if token_id_list:
                    target_tensor = torch.tensor(token_id_list, dtype=torch.long, device=model.device)
                    target_ids[row_idx, : target_tensor.numel()] = target_tensor
                    token_mask[row_idx, : target_tensor.numel()] = True

            trg_mask = Variable(subsequent_mask(max_input_len).type_as(src.data)).to(model.device)
            out = model.model(src, tgt_input, None, trg_mask)
            raw_log_probs = model.generator(out)
            gather_positions = slice(prompt_len - 1, prompt_len - 1 + max_target_len)
            relevant_log_probs = raw_log_probs[:, gather_positions, :]
            gathered = relevant_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(gathered)
            all_masks.append(token_mask)

    return torch.cat(all_logprobs, dim=0), torch.cat(all_masks, dim=0)
