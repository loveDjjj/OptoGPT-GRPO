"""兼容入口：spectral SFT 训练入口已迁移到 run_grpo.py。"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接通过 `python runners/run_spectral_sft.py ...` 运行。
# Python 直接执行脚本时，默认只把 `runners/` 放进 sys.path，
# 这里显式补上仓库根目录，避免找不到 `datasets/`、`models/` 等本地包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.run_grpo import main


if __name__ == "__main__":
    main()
