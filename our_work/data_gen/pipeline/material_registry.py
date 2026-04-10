from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterialRecord:
    name: str
    filename: str


@dataclass(frozen=True)
class MaterialRegistry:
    root_dir: str
    material_names: list[str]
    records: dict[str, MaterialRecord]


def build_material_registry(database_dir: str | Path) -> MaterialRegistry:
    root = Path(database_dir)
    files = sorted(root.glob("*.csv"))
    records = {
        file.stem: MaterialRecord(name=file.stem, filename=file.name)
        for file in files
    }
    return MaterialRegistry(
        root_dir=str(root),
        material_names=sorted(records.keys()),
        records=records,
    )
