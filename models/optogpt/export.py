"""OptoGPT checkpoint 导出辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .checkpoint import OptoGPTModel


def export_optogpt_checkpoint(
    model: OptoGPTModel,
    path: str | Path,
    extra_state: Optional[dict] = None,
) -> None:
    """统一导出接口，便于 runner / trainer 复用。"""

    model.export_checkpoint(path=path, extra_state=extra_state)
