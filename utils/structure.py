"""Helpers for converting decoded OptoGPT tokens into TMM-ready structures.

These helpers deliberately stay small and dependency-light because they sit on
the hot path between policy sampling and physics evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_DATABASE_PATH = "nk"


def split_structure_token(token: str) -> Tuple[str, float]:
    """Split a token like ``SiO2_120`` into material name and thickness in nm."""

    parts = token.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid structure token: {token}")

    material = parts[0]
    thickness_nm = float(parts[1])
    return material, thickness_nm


def tokens_to_tmm_config(
    tokens: Sequence[str],
    database_path: str = DEFAULT_DATABASE_PATH,
    material_aliases: Mapping[str, str] | None = None,
) -> Dict[str, List[float]]:
    """Convert decoded structure tokens into the config schema expected by TMM."""

    if not tokens:
        raise ValueError("Empty structure token sequence.")

    aliases = dict(material_aliases or {})
    materials: List[str] = []
    thicknesses_um: List[float] = []

    for token in tokens:
        material, thickness_nm = split_structure_token(token)
        materials.append(aliases.get(material, material))
        thicknesses_um.append(thickness_nm / 1000.0)

    return {
        "materials": materials,
        "thicknesses": thicknesses_um,
        "database_path": database_path,
    }


def structure_key(tokens: Sequence[str]) -> str:
    """Build the canonical structure string used for exact deduplication."""

    cleaned = [str(token).strip() for token in tokens if str(token).strip()]
    return "|".join(cleaned) if cleaned else "<EMPTY>"


def pad_tmm_config(
    config: Mapping[str, Sequence[float] | str],
    target_layers: int,
    pad_material: str = "Air",
) -> Dict[str, List[float] | str]:
    """Pad one structure to a target layer count using zero-thickness layers."""

    materials = list(config["materials"])
    thicknesses = list(config["thicknesses"])
    if len(materials) != len(thicknesses):
        raise ValueError("materials and thicknesses must have the same length.")
    if len(materials) > target_layers:
        raise ValueError(f"Structure has {len(materials)} layers, exceeds target_layers={target_layers}.")

    pad_count = target_layers - len(materials)
    if pad_count > 0:
        materials.extend([pad_material] * pad_count)
        thicknesses.extend([0.0] * pad_count)

    return {
        "materials": materials,
        "thicknesses": thicknesses,
        "database_path": str(config["database_path"]),
    }


def pad_tmm_configs_to_max_layers(
    configs: Sequence[Mapping[str, Sequence[float] | str]],
    pad_material: str = "Air",
) -> Tuple[List[Dict[str, List[float] | str]], int]:
    """Pad a batch of structures to the maximum layer count within that batch."""

    if not configs:
        return [], 0

    max_layers = max(len(config["materials"]) for config in configs)
    padded = [pad_tmm_config(config, target_layers=max_layers, pad_material=pad_material) for config in configs]
    return padded, max_layers


def bucket_indices_by_layer_count(structure_token_groups: Iterable[Sequence[str]]) -> Dict[int, List[int]]:
    """Group structure indices by raw decoded layer count."""

    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, tokens in enumerate(structure_token_groups):
        buckets[len(tokens)].append(idx)
    return dict(buckets)
