from __future__ import annotations

import random
import torch


def resolve_sampling_device(device: str) -> torch.device:
    if str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("sampling.device requested CUDA but CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


def sample_structure_tokens(
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
    rng_seed: int | None = None,
) -> list[str]:
    rng = random.Random(rng_seed)
    return [
        f"{rng.choice(material_names)}_{rng.choice(thickness_values_nm)}"
        for _ in range(layer_count)
    ]


def sample_structure_token_batch(
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
    batch_size: int,
    device: str = "auto",
    rng_seed: int | None = None,
) -> list[list[str]]:
    torch_device = resolve_sampling_device(device)

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if layer_count <= 0:
        raise ValueError("layer_count must be a positive integer")
    if not material_names:
        raise ValueError("material_names must not be empty")
    if not thickness_values_nm:
        raise ValueError("thickness_values_nm must not be empty")

    generator = torch.Generator(device=torch_device)
    if rng_seed is not None:
        generator.manual_seed(int(rng_seed))

    material_idx = torch.randint(
        low=0,
        high=len(material_names),
        size=(batch_size, layer_count),
        generator=generator,
        device=torch_device,
    )
    thickness_idx = torch.randint(
        low=0,
        high=len(thickness_values_nm),
        size=(batch_size, layer_count),
        generator=generator,
        device=torch_device,
    )

    material_idx_cpu = material_idx.cpu().tolist()
    thickness_idx_cpu = thickness_idx.cpu().tolist()
    return [
        [f"{material_names[m]}_{thickness_values_nm[t]}" for m, t in zip(material_row, thickness_row)]
        for material_row, thickness_row in zip(material_idx_cpu, thickness_idx_cpu)
    ]


def sample_unique_bucket(
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
    target_count: int,
    rng_seed: int,
) -> list[list[str]]:
    rng = random.Random(rng_seed)
    seen: set[tuple[str, ...]] = set()
    results: list[list[str]] = []
    while len(results) < target_count:
        candidate = [
            f"{rng.choice(material_names)}_{rng.choice(thickness_values_nm)}"
            for _ in range(layer_count)
        ]
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        results.append(candidate)
    return results
