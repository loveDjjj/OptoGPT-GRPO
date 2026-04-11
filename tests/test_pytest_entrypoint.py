from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pytest_entrypoint_can_import_our_work_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["pytest", "tests/our_work/pretrain/test_collator.py", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    message = (
        "direct pytest invocation should import our_work from the repo root\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.returncode == 0, message
    assert "2 passed" in result.stdout, message
