# 本次修改摘要

## 需求
- 修复 `our_work/rl` 的两个正式训练问题：
  - 训练期 rollout / scoring 应常驻 `eval()`，避免 checkpoint 默认 dropout 干扰 GRPO old/current logprob
  - RL 入口缺少显式 seed，需在多卡场景按 `rank_offset` 做可复现初始化

## 实际修改
- `our_work/rl/trainer.py`
  - 新增 `_set_policy_eval()` / `_set_policy_train()` 统一管理 policy mode
  - `train(...)` 改为整个 RL 主循环常驻 `eval()`
  - `_evaluate(...)` 改为同时切 `raw_model` 和 `model`，并按进入前 mode 恢复
- `our_work/rl/scripts/run_grpo.py`
  - 新增 `set_global_seed(...)`
  - 在 `init_distributed(...)` 之后立刻执行 `set_global_seed(seed, rank_offset=dist_ctx.rank)`
- `our_work/rl/configs/grpo/base_grpo.yaml`
  - 新增 `seed: 42` 及注释
- `our_work/rl/configs/grpo/a100_4gpu.yaml`
  - 新增 `seed: 42` 及注释
- `our_work/rl/configs/grpo/a100_8gpu.yaml`
  - 新增 `seed: 42` 及注释
- `tests/our_work/rl/test_run_grpo.py`
  - 新增 seed + rank_offset 回归测试
- `tests/our_work/rl/test_trainer.py`
  - 新增训练期 rollout / scoring 常驻 eval 模式回归测试

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -m compileall our_work/rl tests/our_work/rl`
  - 结果：通过
- 手工回归脚本：`run_grpo.py` seed 注入
  - 结果：`seed-regression-ok`
- 手工回归脚本：`trainer.train()` rollout / scoring mode
  - 结果：`eval-mode-regression-ok`
- `pytest`
  - 定向测试主体已触达并完成行为验证，但当前环境的 `pytest` session 清理阶段仍会对临时目录报 `PermissionError`
  - 结果：未获得干净退出码

## Git
- branch: `fix/our-work-rl-policy-eval-seed`
- commit: `fix: stabilize rl policy mode and seeding`
