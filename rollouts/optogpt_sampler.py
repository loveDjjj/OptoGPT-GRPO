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
    oversample_count = int(sampling_config.get("oversample_count", sampling_config.get("candidates_per_target", 8)))
    unique_candidates = int(sampling_config.get("unique_candidates", sampling_config.get("candidates_per_target", 8)))
    raw_samples = sample_rollout_group(
        policy=policy,
        target_spectrum=target_spectrum,
        num_candidates=oversample_count,
        sampling_config=sampling_config,
        target_index=target_index,
        seed=seed,
    )

    seen = set()
    unique_samples: List[RolloutSample] = []
    for sample in raw_samples:
        key = structure_key(sample.structure_tokens)
        if key in seen:
            continue
        seen.add(key)
        unique_samples.append(sample)

    selected_samples = unique_samples[:unique_candidates]

    return SampleGroupResult(
        raw_samples=raw_samples,
        selected_samples=selected_samples,
        sampled_count=len(raw_samples),
        unique_count=len(seen),
        selected_count=len(selected_samples),
        duplicate_count=max(0, len(raw_samples) - len(seen)),
    )
