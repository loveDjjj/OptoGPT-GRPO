from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_records_to_parquet(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)


def write_split_manifest(path: str | Path, payload: dict[str, list[str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
