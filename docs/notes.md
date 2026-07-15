# 本次修改摘要

## 需求
- 扩大 2-25 um 六波段 PSO 的搜索范围，同时减少总运行时间，并提交 GitHub。

## 实际修改
- 更新 `our_work/pso/configs/pso_2_25_binary_bands.yaml`：
  - 单层厚度由 `10-500 nm / step 10` 扩展为 `20-1000 nm / step 20`，厚度离散数量保持 `50` 个。
  - `population_size`：`8192 -> 16384`，扩大每轮并行探索广度。
  - `iterations`：`20 -> 8`。
  - `max_restarts`：`5 -> 2`。
  - search/TMM `batch_size`：`8192 -> 16384`。
  - `acceptance_mse_threshold`：`0.01 -> 0.015`。
  - `max_stagnant_iterations`：`5 -> 3`。
- 保持 `31` 个目标、`5-10` 层、`1024` 波长点和 `complex128` 不变。

## 预算变化
- 最大 TMM 候选评测量由约 `1.52 亿` 降至约 `4876 万`，减少约 `68%`。
- 每个 restart 的初始种群覆盖扩大为原来的 `2` 倍。

## 验证
- YAML 加载、厚度离散值、目标数量和最大搜索预算检查通过。
- `pytest tests/our_work/pso/test_targets.py tests/our_work/pso/test_visualization.py -q`：7 passed。
- Python 语法编译与 `git diff --check` 通过。
- 完整 PSO/TMM GPU 性能与显存占用待服务器验证；若 OOM，优先只把两个 `batch_size` 降为 `8192`。

## Git
- branch: main
- commit: pending
