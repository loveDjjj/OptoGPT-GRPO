"""结构 token 与 TMM 配置之间的转换工具。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_DATABASE_PATH = "data/materials"


def split_structure_token(token: str) -> Tuple[str, float]:
    """把 `SiO2_120` 这样的 token 拆成材料名和厚度（nm）。"""

    parts = token.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"非法结构 token: {token}")

    material = parts[0]
    thickness_nm = float(parts[1])
    return material, thickness_nm


def tokens_to_tmm_config(
    tokens: Sequence[str],
    database_path: str = DEFAULT_DATABASE_PATH,
    material_aliases: Mapping[str, str] | None = None,
) -> Dict[str, List[float] | str]:
    """把结构 token 序列转换成 TMM 输入配置。"""

    if not tokens:
        raise ValueError("结构 token 序列为空。")

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
    """把结构 token 序列拼成稳定的去重 key。"""

    cleaned = [str(token).strip() for token in tokens if str(token).strip()]
    return "|".join(cleaned) if cleaned else "<EMPTY>"


def pad_tmm_config(
    config: Mapping[str, Sequence[float] | str],
    target_layers: int,
    pad_material: str = "Air",
) -> Dict[str, List[float] | str]:
    """用零厚度 Air 层把结构补到固定层数。"""

    materials = list(config["materials"])
    thicknesses = list(config["thicknesses"])
    if len(materials) != len(thicknesses):
        raise ValueError("materials 与 thicknesses 的长度必须一致。")
    if len(materials) > target_layers:
        raise ValueError(f"结构层数 {len(materials)} 超过目标层数 {target_layers}。")

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
    """把一个 batch 中的结构补到当前 batch 的最大层数。"""

    if not configs:
        return [], 0

    max_layers = max(len(config["materials"]) for config in configs)
    padded = [pad_tmm_config(config, target_layers=max_layers, pad_material=pad_material) for config in configs]
    return padded, max_layers


def pad_tmm_configs_to_fixed_layers(
    configs: Sequence[Mapping[str, Sequence[float] | str]],
    target_layers: int,
    pad_material: str = "Air",
) -> List[Dict[str, List[float] | str]]:
    """把一组结构统一补到固定层数。

    这比“按当前 batch 最大层数补齐”更适合长期批量计算：
    - 所有 batch 的张量形状固定
    - 便于稳定调优 TMM batch size
    - 避免不同 batch 之间反复切换不同 layer shape
    """

    return [pad_tmm_config(config, target_layers=target_layers, pad_material=pad_material) for config in configs]


def bucket_indices_by_layer_count(structure_token_groups: Iterable[Sequence[str]]) -> Dict[int, List[int]]:
    """按层数给结构分桶。"""

    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, tokens in enumerate(structure_token_groups):
        buckets[len(tokens)].append(idx)
    return dict(buckets)
