from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml


def create_eval_run_dir(output_root: str | Path, *, run_name: str, timestamp: str | None = None) -> Path:
    base_stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_root = Path(output_root) / run_name
    if timestamp is not None:
        run_dir = run_root / base_stamp
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        suffix = 0
        while True:
            stamp = base_stamp if suffix == 0 else f"{base_stamp}-{suffix}"
            run_dir = run_root / stamp
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
    (run_dir / "plots").mkdir()
    (run_dir / "samples").mkdir()
    return run_dir


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def write_config_snapshot(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
