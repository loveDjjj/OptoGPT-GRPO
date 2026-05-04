# 本次修改摘要

## 需求
- 修复逐 shard 评测脚本在服务器执行时报错：
  - `#!/usr/bin/env` 无法识别
  - `ModuleNotFoundError: No module named 'our_work'`

## 实际修改
- `our_work/pretrain/scripts/eval_each_shard.py`
  - 新增 `PROJECT_ROOT` 注入 `sys.path`，与仓库其他入口脚本保持一致，修复 `our_work` 包导入失败。
- `run_eval_ga_custom_checkpoint980_each_shard.sh`
  - 以 ASCII 重新写入，避免 BOM/CRLF 导致 shebang 在 Linux 上失效。

## 结果
- 脚本满足 Linux shebang 解析要求。
- 逐 shard 评测入口可在仓库根目录直接导入 `our_work` 模块执行。

## Git
- branch: main
- commit: pending
