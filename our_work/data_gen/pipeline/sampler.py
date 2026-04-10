from __future__ import annotations

import random


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
