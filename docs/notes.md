# 本次修改摘要

## 需求
- 解决 `our_work/ga` 在达到 `max_samples_per_target` 之后明显变慢的问题。
- 重点优化池满后的去重、替换和候选筛选，尽量把可并行部分继续留在 GPU。

## 实际修改
- `our_work/ga/search.py`
  - 新增 `build_numeric_structure_keys()`，候选池 key 由 token tuple 改为数值 key：
    - `((material_idx...), (thickness_idx...))`
  - 新增 `unique_population_rows()`，在每个 chunk 进入 TMM 前先做 batch 内 GPU 去重。
  - 新增 `GAEvaluatedCandidate`，evaluator 先返回轻量候选，再只对真正入池的样本 materialize 为 `GAStructure`。
  - 候选池满后：
    - 每个 chunk 使用 `min(acceptance_floor_mse, current_worst_mse)` 作为动态 cutoff；
    - 只保留本 batch 中最有可能替换池中最差样本的候选。
  - 新增 `worst_heap`，用堆维护当前最差样本，替代反复 `max(accepted_map.items())` 的全池扫描。
  - `total_evaluated` 现在统计“真正送入 evaluator/TMM 的去重后结构数”，不再把 batch 内重复结构重复计数。
- `tests/our_work/ga/test_search.py`
  - 新增数值 key 测试。
  - 新增 batch 内 unique 测试。
  - 新增池满后动态 cutoff 测试。
  - 同步更新已有 full-budget 测试的评估计数预期。

## 性能方向
- 达到数量上限前：
  - 主要瓶颈仍是 TMM，本轮不改变其数值逻辑。
- 达到数量上限后：
  - 先在 GPU 上去重、筛掉不可能入池的候选；
  - 再用最差样本堆缩小 CPU 替换判定开销；
  - 只为真正可能入池的候选生成 token 和样本对象。

## 验证
- `D:\anaconda\envs\oneday\python.exe -B -m pytest tests\our_work\ga -q -p no:cacheprovider --basetemp tests\.tmp-ga-opt`
  - 结果：`17 passed`
- 最小主入口烟测：
  - 结果：`ga opt smoke ok`
- `D:\anaconda\envs\oneday\python.exe -m compileall our_work\ga our_work\_shared\physics tests\our_work\ga`
  - 待本次收尾后重新执行

## Git
- branch: `main`
- commit: pending
