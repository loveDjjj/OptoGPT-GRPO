"""OptoGPT 结构生成相关工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import torch
from torch.autograd import Variable

from core.transformer import subsequent_mask
from .checkpoint import OptoGPTModel


@dataclass
class DecodeConfig:
    """结构生成超参数。"""

    decode: str = "greedy"
    top_k: int = 10
    top_p: float = 0.9
    temperature: float = 1.0
    start_symbol: str = "BOS"
    start_mat: Optional[str] = None
    max_len: Optional[int] = None
    batch_size: Optional[int] = None


@dataclass
class GeneratedStructure:
    """一次生成得到的结构样本。"""

    target_index: Optional[int]
    candidate_index: int
    prompt_ids: List[int]
    token_ids: List[int]
    tokens: List[str]
    structure_tokens: List[str]
    raw_logprobs: List[float]
    sequence_raw_logprob: float
    terminated_by_eos: bool
    max_len_reached: bool
    decode: str


def build_decode_config(config: Mapping[str, object], default_max_len: int | None = None) -> DecodeConfig:
    """从配置字典构造生成参数。"""

    configured_max_len = config.get("max_len", default_max_len)
    return DecodeConfig(
        decode=str(config.get("decode", "greedy")),
        top_k=int(config.get("top_k", 10)),
        top_p=float(config.get("top_p", 0.9)),
        temperature=float(config.get("temperature", 1.0)),
        start_symbol=str(config.get("start_symbol", "BOS")),
        start_mat=config.get("start_mat"),
        max_len=None if configured_max_len is None else int(configured_max_len),
        batch_size=int(config.get("batch_size")) if config.get("batch_size") is not None else None,
    )


def _combine_seed_sequence(seeds: Optional[Sequence[Optional[int]]]) -> Optional[int]:
    """把多条样本的种子折叠成一个 batched 生成种子。"""

    if seeds is None:
        return None
    combined = 1469598103934665603
    for idx, seed in enumerate(seeds):
        value = 0 if seed is None else int(seed)
        combined ^= (value + 0x9E3779B97F4A7C15 + idx) & 0xFFFFFFFFFFFFFFFF
        combined = (combined * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(combined % (2**63 - 1))


def _filtered_distribution_batch(
    raw_log_prob: torch.Tensor,
    decode_config: DecodeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对下一 token 分布做 greedy / top-k / top-p 过滤。"""

    if raw_log_prob.dim() != 2:
        raise ValueError(f"raw_log_prob 必须是二维张量 [batch, vocab]，当前形状为 {tuple(raw_log_prob.shape)}")
    if decode_config.temperature <= 0:
        raise ValueError("temperature 必须大于 0。")
    if decode_config.top_k < 0:
        raise ValueError("top_k 不能小于 0。")
    if not 0 < decode_config.top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 之间。")
    if decode_config.decode not in {"greedy", "top-k", "sample"}:
        raise ValueError(f"不支持的 decode 模式: {decode_config.decode}")

    tempered_log_prob = raw_log_prob / decode_config.temperature
    prob = torch.softmax(tempered_log_prob, dim=-1)

    if decode_config.decode == "greedy":
        greedy_idx = torch.argmax(prob, dim=-1, keepdim=True)
        filtered = torch.zeros_like(prob)
        filtered.scatter_(1, greedy_idx, 1.0)
        return prob, filtered

    filtered = prob.clone()
    vocab_size = filtered.size(-1)
    if decode_config.decode == "top-k":
        if decode_config.top_k > 0 and decode_config.top_k < vocab_size:
            topk_values = torch.topk(filtered, decode_config.top_k, dim=-1).values
            kth = topk_values[:, -1:].expand_as(filtered)
            filtered = torch.where(filtered < kth, torch.zeros_like(filtered), filtered)

        if decode_config.top_p < 1.0:
            sorted_prob, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_prob, dim=-1)
            remove_mask = cumulative > decode_config.top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_prob = torch.where(remove_mask, torch.zeros_like(sorted_prob), sorted_prob)
            filtered.zero_()
            filtered.scatter_(1, sorted_idx, sorted_prob)

    filtered_sum = filtered.sum(dim=-1, keepdim=True)
    empty_mask = filtered_sum <= 0
    if empty_mask.any():
        filtered = torch.where(empty_mask, prob, filtered)
        filtered_sum = filtered.sum(dim=-1, keepdim=True)
    filtered = filtered / filtered_sum.clamp_min(1e-12)
    return prob, filtered


def generate_structures_for_targets(
    model: OptoGPTModel,
    target_spectra: Sequence[Sequence[float]],
    decode_config: DecodeConfig,
    num_samples_per_target: int = 1,
    target_indices: Optional[Sequence[Optional[int]]] = None,
    seeds: Optional[Sequence[Optional[int]]] = None,
) -> List[GeneratedStructure]:
    """对多条目标光谱批量生成结构。

    当前项目的多卡加速策略是“数据并行”，所以每个 rank 只负责自己那部分样本，
    每张卡本地完成生成即可，不需要跨卡同步生成过程。
    """

    if num_samples_per_target <= 0 or len(target_spectra) == 0:
        return []

    target_count = len(target_spectra)
    resolved_target_indices = (
        list(target_indices)
        if target_indices is not None
        else [int(idx) for idx in range(target_count)]
    )
    if len(resolved_target_indices) != target_count:
        raise ValueError("target_indices 长度必须与 target_spectra 一致。")
    if seeds is not None and len(seeds) != target_count:
        raise ValueError("seeds 长度必须与 target_spectra 一致。")

    generator = None
    combined_seed = _combine_seed_sequence(seeds)
    if combined_seed is not None:
        generator = torch.Generator(device=model.device.type)
        generator.manual_seed(combined_seed)

    expanded_spectra: List[Sequence[float]] = []
    expanded_target_indices: List[Optional[int]] = []
    expanded_candidate_indices: List[int] = []
    if torch.is_tensor(target_spectra):
        if target_spectra.dim() == 1:
            target_spectra = target_spectra.view(1, -1)
        expanded_spectra_tensor = target_spectra.repeat((num_samples_per_target, 1))
        for candidate_index in range(num_samples_per_target):
            expanded_target_indices.extend(resolved_target_indices)
            expanded_candidate_indices.extend([candidate_index] * target_count)
    else:
        expanded_spectra_tensor = None
        for candidate_index in range(num_samples_per_target):
            for target_spectrum, target_index in zip(target_spectra, resolved_target_indices):
                expanded_spectra.append(target_spectrum)
                expanded_target_indices.append(target_index)
                expanded_candidate_indices.append(candidate_index)

    total_expanded_count = (
        int(expanded_spectra_tensor.size(0))
        if expanded_spectra_tensor is not None
        else len(expanded_spectra)
    )
    effective_batch_size = int(decode_config.batch_size or total_expanded_count)
    effective_batch_size = max(1, effective_batch_size)

    all_samples: List[GeneratedStructure] = []
    for start in range(0, total_expanded_count, effective_batch_size):
        stop = start + effective_batch_size
        src_input = (
            expanded_spectra_tensor[start:stop]
            if expanded_spectra_tensor is not None
            else expanded_spectra[start:stop]
        )
        all_samples.extend(
            _decode_from_src_batch(
                model=model,
                src=model.targets_to_tensor_batch(src_input),
                decode_config=decode_config,
                target_indices=expanded_target_indices[start:stop],
                candidate_indices=expanded_candidate_indices[start:stop],
                rng=generator,
            )
        )
    return all_samples


def _decode_from_src_batch(
    model: OptoGPTModel,
    src: torch.Tensor,
    decode_config: DecodeConfig,
    target_indices: Sequence[Optional[int]],
    candidate_indices: Sequence[int],
    rng: Optional[torch.Generator] = None,
) -> List[GeneratedStructure]:
    """共享的自回归解码循环。"""

    num_samples = int(src.size(0))
    if len(target_indices) != num_samples or len(candidate_indices) != num_samples:
        raise ValueError("target_indices 和 candidate_indices 必须与 batch 大小一致。")

    prompt_ids = model.prompt_ids(
        start_symbol=decode_config.start_symbol,
        start_mat=decode_config.start_mat,
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    ys = prompt_tensor.expand(num_samples, -1).clone()
    generation_limit = int(decode_config.max_len or model.max_len)
    pad_id = int(model.struc_word_dict.get(model.pad_token, model.struc_word_dict.get("EOS", 0)))

    generated_ids: List[List[int]] = [[] for _ in range(num_samples)]
    generated_tokens: List[List[str]] = [[] for _ in range(num_samples)]
    structure_tokens: List[List[str]] = [[] for _ in range(num_samples)]
    raw_logprobs: List[List[float]] = [[] for _ in range(num_samples)]
    terminated_by_eos = [False for _ in range(num_samples)]
    active_mask = torch.ones(num_samples, dtype=torch.bool, device=model.device)

    with torch.no_grad():
        while ys.size(1) < generation_limit and bool(active_mask.any().item()):
            trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(model.device)
            out = model.model(src, ys, None, trg_mask)
            raw_log_prob = model.generator(out[:, -1])
            _, sample_prob = _filtered_distribution_batch(raw_log_prob, decode_config)

            if decode_config.decode == "greedy":
                next_ids = torch.argmax(sample_prob, dim=-1)
            else:
                next_ids = torch.multinomial(sample_prob, num_samples=1, generator=rng).squeeze(-1)

            step_raw_logprob = raw_log_prob.gather(1, next_ids.unsqueeze(-1)).squeeze(-1)
            next_ids_to_append = torch.where(active_mask, next_ids, torch.full_like(next_ids, pad_id))
            ys = torch.cat([ys, next_ids_to_append.unsqueeze(1)], dim=1)

            active_indices = torch.nonzero(active_mask, as_tuple=False).view(-1).tolist()
            finished_indices: List[int] = []
            for idx in active_indices:
                next_id = int(next_ids[idx].item())
                token = model.token_id_to_str(next_id)
                generated_ids[idx].append(next_id)
                generated_tokens[idx].append(token)
                raw_logprobs[idx].append(float(step_raw_logprob[idx].item()))
                if token == model.eos_token:
                    terminated_by_eos[idx] = True
                    finished_indices.append(idx)
                elif token not in {model.bos_token, model.pad_token, "UNK"}:
                    structure_tokens[idx].append(token)

            if finished_indices:
                active_mask[torch.tensor(finished_indices, dtype=torch.long, device=model.device)] = False

    samples: List[GeneratedStructure] = []
    for sample_idx in range(num_samples):
        samples.append(
            GeneratedStructure(
                target_index=target_indices[sample_idx],
                candidate_index=int(candidate_indices[sample_idx]),
                prompt_ids=list(prompt_ids),
                token_ids=generated_ids[sample_idx],
                tokens=generated_tokens[sample_idx],
                structure_tokens=structure_tokens[sample_idx],
                raw_logprobs=raw_logprobs[sample_idx],
                sequence_raw_logprob=float(sum(raw_logprobs[sample_idx])),
                terminated_by_eos=terminated_by_eos[sample_idx],
                max_len_reached=not terminated_by_eos[sample_idx],
                decode=decode_config.decode,
            )
        )
    return samples
