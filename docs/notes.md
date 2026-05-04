# 本次修改摘要

## 需求
- 新增单脚本：读取 `our_work/ga/configs/ga_custom_tasks.yaml` 的目标光谱任务，使用 `checkpoint-980` 推理结构，做 TMM 批量回算与误差可视化分析。

## 实际修改
- 新增 `our_work/eval/scripts/run_ga_target_inference_analysis.py`：
  - 读取 GA custom tasks 中所有目标 band 定义（支持 include_ids 过滤）。
  - 每个目标采样 1024 个结构（可配置），随机采样解码（temperature/top-k/top-p）。
  - 按层数分桶后做 TMM 批量计算（chunk 批处理）。
  - 误差定义：目标 absorption 的带掩码 MSE（只在任务 bands 内计算）。
  - 可视化输出：
    - 最优样本 `R/T/A` 三张曲线图（纵坐标固定 0-1）
    - 全体有效样本误差直方图
    - 全体有效样本排序误差曲线
  - 同时导出 best/target 光谱数组与 per-target summary.json、overall_summary.json。

## 结果
- 可以单卡端到端完成“目标->结构推理->真实光谱回算->误差统计与可视化”的分析。

## Git
- branch: main
- commit: pending
