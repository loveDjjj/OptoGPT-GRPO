from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence

from policy.optogpt_policy import DecodeConfig, OptoGPTPolicy, RolloutSample
from utils.structure import structure_key


@dataclass
class SampleGroupResult:
    raw_samples: List[RolloutSample]
    selected_samples: List[RolloutSample]
    sampled_count: int
    unique_count: int
    selected_count: int
    duplicate_count: int


def decode_config_from_dict(config: Mapping[str, Any], max_len: int | None = None) -> DecodeConfig:
    configured_max_len = config.get("max_len", max_len)
    return DecodeConfig(
        decode=str(config.get("decode", "top-k")),
        top_k=int(config.get("top_k", 10)),
        top_p=float(config.get("top_p", 0.8)),
        temperature=float(config.get("temperature", 1.0)),
        start_symbol=str(config.get("start_symbol", "BOS")),
        start_mat=config.get("start_mat"),
        max_len=None if configured_max_len is None else int(configured_max_len),
        batch_size=int(config.get("batch_size")) if config.get("batch_size") is not None else None,
    )


def rollout_to_record(sample: RolloutSample) -> Dict[str, Any]:
    return asdict(sample)


def _select_unique_samples(
    raw_samples: Sequence[RolloutSample],
    unique_candidates: int,
) -> tuple[List[RolloutSample], int]:
    seen = set()
    unique_samples: List[RolloutSample] = []
    for sample in raw_samples:
        key = structure_key(sample.structure_tokens)
        if key in seen:
            continue
        seen.add(key)
        unique_samples.append(sample)
    return unique_samples[:unique_candidates], len(seen)


def sample_rollout_group(
    policy: OptoGPTPolicy,
    target_spectrum: Sequence[float],
    num_candidates: int,
    sampling_config: Mapping[str, Any],
    target_index: int | None = None,
    seed: int | None = None,
) -> List[RolloutSample]:
    decode_config = decode_config_from_dict(sampling_config, max_len=policy.max_len)
    return policy.sample_group(
        target_spectrum=target_spectrum,
        num_samples=num_candidates,
        decode_config=decode_config,
        target_index=target_index,
        seed=seed,
    )


def sample_unique_rollout_group(
    policy: OptoGPTPolicy,
    target_spectrum: Sequence[float],
    sampling_config: Mapping[str, Any],
    target_index: int | None = None,
    seed: int | None = None,
) -> SampleGroupResult:
    results = sample_unique_rollout_groups(
        policy=policy,
        target_spectra=[target_spectrum],
        sampling_config=sampling_config,
        target_indices=[target_index],
        seeds=[seed],
    )
    return results[0]


def sample_unique_rollout_groups(
    policy: OptoGPTPolicy,
    target_spectra: Sequence[Sequence[float]],
    sampling_config: Mapping[str, Any],
    target_indices: Sequence[int | None] | None = None,
    seeds: Sequence[int | None] | None = None,
) -> List[SampleGroupResult]:
    oversample_count = int(sampling_config.get("oversample_count", sampling_config.get("candidates_per_target", 8)))
    unique_candidates = int(sampling_config.get("unique_candidates", sampling_config.get("candidates_per_target", 8)))
    decode_config = decode_config_from_dict(sampling_config, max_len=policy.max_len)
    resolved_target_indices = (
        list(target_indices)
        if target_indices is not None
        else [int(idx) for idx in range(len(target_spectra))]
    )
    raw_samples = policy.sample_group_multi_target(
        target_spectra=target_spectra,
        num_samples_per_target=oversample_count,
        decode_config=decode_config,
        target_indices=resolved_target_indices,
        seeds=seeds,
    )

    raw_sample_map = {target_index: [] for target_index in resolved_target_indices}
    for sample in raw_samples:
        raw_sample_map.setdefault(sample.target_index, []).append(sample)

    results: List[SampleGroupResult] = []
    for target_index in resolved_target_indices:
        target_raw_samples = sorted(
            raw_sample_map.get(target_index, []),
            key=lambda item: int(item.candidate_index),
        )
        selected_samples, unique_count = _select_unique_samples(
            raw_samples=target_raw_samples,
            unique_candidates=unique_candidates,
        )
        results.append(
            SampleGroupResult(
                raw_samples=target_raw_samples,
                selected_samples=selected_samples,
                sampled_count=len(target_raw_samples),
                unique_count=unique_count,
                selected_count=len(selected_samples),
                duplicate_count=max(0, len(target_raw_samples) - unique_count),
            )
        )
    return results
