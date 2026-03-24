from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for GRPO yaml configs.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml_config(path: str | Path, config: Dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for GRPO yaml configs.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
