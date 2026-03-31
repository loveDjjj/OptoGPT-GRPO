"""OptoGPT 结构生成相关工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import torch

from core.transformer import subsequent_mask
from .checkpoint import OptoGPTModel
from .policy import policy_log_probs_from_raw_log_probs, validate_policy_config


_SUBSEQUENT_MASK_CACHE: dict[tuple[str, str, int], torch.Tensor] = {}


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
    policy_logprobs: List[float]
    sequence_policy_logprob: float
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


def _cached_subsequent_mask(length: int, reference: torch.Tensor) -> torch.Tensor:
    """缓存下三角 mask，避免自回归每一步都重新在 CPU 构造。"""

    cache_key = (str(reference.device), str(reference.dtype), int(length))
    cached = _SUBSEQUENT_MASK_CACHE.get(cache_key)
    if cached is None:
        cached = subsequent_mask(length).type_as(reference).to(reference.device)
        _SUBSEQUENT_MASK_CACHE[cache_key] = cached
    return cached


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
    validate_policy_config(decode_config)

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
        expanded_spectra_tensor = target_spectra.repeat_interleave(num_samples_per_target, dim=0)
        for target_index in resolved_target_indices:
            expanded_target_indices.extend([target_index] * num_samples_per_target)
            expanded_candidate_indices.extend(range(num_samples_per_target))
    else:
        expanded_spectra_tensor = None
        for target_spectrum, target_index in zip(target_spectra, resolved_target_indices):
            for candidate_index in range(num_samples_per_target):
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
    prompt_len = int(prompt_tensor.size(1))
    generation_limit = int(decode_config.max_len or model.max_len)
    pad_id = int(model.struc_word_dict.get(model.pad_token, model.struc_word_dict.get("EOS", 0)))
    eos_id = int(model.struc_word_dict[model.eos_token])
    max_decode_steps = max(0, generation_limit - prompt_len)

    # 预先分配整个解码张量，避免每一步 `torch.cat` 造成的重复显存分配。
    ys = torch.full(
        (num_samples, generation_limit),
        pad_id,
        dtype=torch.long,
        device=model.device,
    )
    ys[:, :prompt_len] = prompt_tensor
    step_logprobs = torch.zeros(
        (num_samples, max_decode_steps),
        dtype=torch.float32,
        device=model.device,
    )
    step_policy_logprobs = torch.zeros(
        (num_samples, max_decode_steps),
        dtype=torch.float32,
        device=model.device,
    )
    generated_lengths = torch.zeros((num_samples,), dtype=torch.long, device=model.device)
    terminated_by_eos = torch.zeros((num_samples,), dtype=torch.bool, device=model.device)
    active_indices = torch.arange(num_samples, device=model.device, dtype=torch.long)
    current_len = prompt_len
    step_idx = 0

    with torch.inference_mode():
        while current_len < generation_limit and int(active_indices.numel()) > 0:
            active_src = src.index_select(0, active_indices)
            active_prefix = ys.index_select(0, active_indices)[:, :current_len]
            trg_mask = _cached_subsequent_mask(current_len, active_src.data)
            out = model.model(active_src, active_prefix, None, trg_mask)
            raw_log_prob = model.generator(out[:, -1])
            policy_log_prob = policy_log_probs_from_raw_log_probs(raw_log_prob, decode_config)
            sample_prob = policy_log_prob.exp()

            if decode_config.decode == "greedy":
                next_ids = torch.argmax(policy_log_prob, dim=-1)
            else:
                next_ids = torch.multinomial(sample_prob, num_samples=1, generator=rng).squeeze(-1)

            step_raw_logprob = raw_log_prob.gather(1, next_ids.unsqueeze(-1)).squeeze(-1)
            step_policy_logprob = policy_log_prob.gather(1, next_ids.unsqueeze(-1)).squeeze(-1)
            ys[active_indices, current_len] = next_ids
            step_logprobs[active_indices, step_idx] = step_raw_logprob
            step_policy_logprobs[active_indices, step_idx] = step_policy_logprob
            generated_lengths[active_indices] = generated_lengths[active_indices] + 1

            eos_finished = next_ids.eq(eos_id)
            if bool(eos_finished.any().item()):
                terminated_by_eos[active_indices[eos_finished]] = True
            active_indices = active_indices[~eos_finished]
            current_len += 1
            step_idx += 1

    samples: List[GeneratedStructure] = []
    generated_token_ids = ys[:, prompt_len:current_len]
    for sample_idx in range(num_samples):
        token_count = int(generated_lengths[sample_idx].item())
        token_ids = generated_token_ids[sample_idx, :token_count].tolist()
        raw_logprob_list = step_logprobs[sample_idx, :token_count].tolist()
        policy_logprob_list = step_policy_logprobs[sample_idx, :token_count].tolist()
        tokens = [model.token_id_to_str(token_id) for token_id in token_ids]
        samples.append(
            GeneratedStructure(
                target_index=target_indices[sample_idx],
                candidate_index=int(candidate_indices[sample_idx]),
                prompt_ids=list(prompt_ids),
                token_ids=token_ids,
                tokens=tokens,
                structure_tokens=model.token_ids_to_structure_tokens(token_ids, stop_at_eos=True),
                raw_logprobs=raw_logprob_list,
                sequence_raw_logprob=float(sum(raw_logprob_list)),
                policy_logprobs=policy_logprob_list,
                sequence_policy_logprob=float(sum(policy_logprob_list)),
                terminated_by_eos=bool(terminated_by_eos[sample_idx].item()),
                max_len_reached=not bool(terminated_by_eos[sample_idx].item()),
                decode=decode_config.decode,
            )
        )
    return samples
