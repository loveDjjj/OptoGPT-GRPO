from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PATH_KEYS = {
    "checkpoint",
    "dataset_dir",
    "database_dir",
    "materials_dir",
    "output_dir",
    "output_json",
    "vocab_path",
}


def resolve_repo_path(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    root = Path(project_root)
    for search_root in (root, *root.parents):
        resolved = search_root / candidate
        if resolved.exists():
            return resolved
    return root / candidate


def resolve_config_paths(payload: dict[str, Any], *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    def _resolve(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {item_key: _resolve(item_value, key=item_key) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [_resolve(item) for item in value]
        if isinstance(value, str) and key is not None and (key in _PATH_KEYS or key.endswith(("_dir", "_path"))):
            return str(resolve_repo_path(value, project_root=project_root))
        return value

    return _resolve(deepcopy(payload))


def load_yaml_config(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    resolve_relative_paths: bool = False,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if resolve_relative_paths:
        return resolve_config_paths(payload, project_root=project_root)
    return payload
