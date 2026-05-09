from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _shared.io.config import load_yaml_config, resolve_repo_path
from eval.pipeline import run_eval_suite


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run config-driven eval suite for train/val splits.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
    config = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)
    payload = run_eval_suite(config)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return payload


if __name__ == "__main__":
    main()
