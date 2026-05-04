# 本次修改摘要

## 需求
- 对 `ga_custom_tasks/shards/shard-00000..00007.parquet` 做逐 shard 评测。
- 使用模型 `outputs/our_work/pretrain/a100_4gpu/checkpoint-980`。
- 每个 shard 独立输出评测结果与可视化图。

## 实际修改
- 新增脚本 `our_work/pretrain/scripts/eval_each_shard.py`：
  - 逐个读取 `shard-*.parquet`。
  - 复用现有评测路径（生成结构 -> TMM 回算 -> 误差统计）。
  - 每个 shard 独立创建 run 目录并输出：
    - `summary.json`
    - `results.jsonl`
    - `plots/*.png`
    - `samples/*.png`
  - 汇总写出 `combined_summary.json`。
- 新增一键运行脚本 `run_eval_ga_custom_checkpoint980_each_shard.sh`：
  - 固定 checkpoint= `outputs/our_work/pretrain/a100_4gpu/checkpoint-980`
  - 固定 shards_dir= `outputs/our_work/data_gen/ga_custom_tasks/shards`
  - 固定输出根目录= `outputs/our_work/eval/ga_custom_tasks_checkpoint980_each_shard`

## 结果
- 可以一次性完成 8 个 shard 的独立评测和独立可视化。

## Git
- branch: main
- commit: pending
