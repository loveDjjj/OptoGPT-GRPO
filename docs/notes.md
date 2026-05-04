# 本次修改摘要

## 需求
- 将 4 卡评测脚本从“运行时生成临时 YAML”改为“固定配置文件 + 简单启动命令”。

## 实际修改
- 新增固定配置文件：
  - `our_work/eval/configs/ga_custom_checkpoint980_4gpu.yaml`
  - 包含 checkpoint、ga_custom_tasks 数据集、输出目录与评测参数。
- 更新启动脚本：
  - `run_eval_ga_custom_checkpoint980_each_shard.sh`
  - 仅保留 `torchrun --nproc_per_node=4 ... --config ...`。

## 结果
- 评测入口更清晰，配置可追踪、可复用、可版本化。

## Git
- branch: main
- commit: pending
