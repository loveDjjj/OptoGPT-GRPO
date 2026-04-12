from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .plots import save_bar_chart, save_material_heatmap, save_thickness_heatmap


def analyze_structure_distribution(
    *,
    scope_name: str,
    batches: Iterable[list[dict]],
    material_names: Sequence[str],
    thickness_values_nm: Sequence[int],
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    layer_count_max = 0
    material_counts_by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    thickness_counts_by_layer: dict[int, Counter[int]] = defaultdict(Counter)
    global_material_counts: Counter[str] = Counter()
    global_thickness_counts: Counter[int] = Counter()
    sample_count = 0

    for batch in batches:
        for record in batch:
            sample_count += 1
            materials = list(record.get("materials", []))
            thicknesses = [int(value) for value in record.get("thickness_nm", [])]
            layer_count_max = max(layer_count_max, len(materials))
            for layer_index, material in enumerate(materials):
                material_counts_by_layer[layer_index][material] += 1
                global_material_counts[material] += 1
            for layer_index, thickness in enumerate(thicknesses):
                thickness_counts_by_layer[layer_index][thickness] += 1
                global_thickness_counts[thickness] += 1

    resolved_materials = sorted(set(material_names) | set(global_material_counts.keys()))
    resolved_thicknesses = sorted(set(int(value) for value in thickness_values_nm) | set(global_thickness_counts.keys()))
    layer_labels = [f"L{index + 1}" for index in range(layer_count_max)]

    if sample_count == 0 or layer_count_max == 0 or not resolved_materials or not resolved_thicknesses:
        summary = {
            "scope": scope_name,
            "sample_count": int(sample_count),
            "max_layer_count": int(layer_count_max),
            "artifacts": {},
            "skipped_reason": "no records",
        }
        (output_dir / "structure_analysis.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    material_index = {name: idx for idx, name in enumerate(resolved_materials)}
    thickness_index = {value: idx for idx, value in enumerate(resolved_thicknesses)}
    material_heatmap = np.zeros((len(resolved_materials), layer_count_max), dtype=np.int64)
    thickness_heatmap = np.zeros((len(resolved_thicknesses), layer_count_max), dtype=np.int64)

    for layer_index, counter in material_counts_by_layer.items():
        for name, count in counter.items():
            material_heatmap[material_index[name], layer_index] = int(count)
    for layer_index, counter in thickness_counts_by_layer.items():
        for thickness, count in counter.items():
            thickness_heatmap[thickness_index[int(thickness)], layer_index] = int(count)

    save_material_heatmap(
        material_heatmap,
        material_names=resolved_materials,
        layer_labels=layer_labels,
        output_path=output_dir / "structure_material_by_layer.png",
    )
    save_thickness_heatmap(
        thickness_heatmap,
        thickness_values_nm=resolved_thicknesses,
        layer_labels=layer_labels,
        output_path=output_dir / "structure_thickness_by_layer.png",
    )
    top_materials = [name for name, _ in global_material_counts.most_common(20)]
    save_bar_chart(
        top_materials,
        [global_material_counts[name] for name in top_materials],
        title=f"Top Materials ({scope_name})",
        ylabel="Count",
        output_path=output_dir / "structure_material_global.png",
    )
    save_bar_chart(
        [str(value) for value in resolved_thicknesses],
        [global_thickness_counts.get(int(value), 0) for value in resolved_thicknesses],
        title=f"Thickness Distribution ({scope_name})",
        ylabel="Count",
        output_path=output_dir / "structure_thickness_global.png",
    )

    summary = {
        "scope": scope_name,
        "sample_count": int(sample_count),
        "max_layer_count": int(layer_count_max),
        "materials": {
            "global_counts": {name: int(count) for name, count in global_material_counts.items()},
            "by_layer": {
                str(layer_index + 1): {name: int(count) for name, count in counter.items()}
                for layer_index, counter in material_counts_by_layer.items()
            },
        },
        "thickness_nm": {
            "global_counts": {str(value): int(count) for value, count in global_thickness_counts.items()},
            "by_layer": {
                str(layer_index + 1): {str(value): int(count) for value, count in counter.items()}
                for layer_index, counter in thickness_counts_by_layer.items()
            },
        },
        "artifacts": {
            "material_by_layer": "structure_material_by_layer.png",
            "thickness_by_layer": "structure_thickness_by_layer.png",
            "material_global": "structure_material_global.png",
            "thickness_global": "structure_thickness_global.png",
        },
    }
    (output_dir / "structure_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
