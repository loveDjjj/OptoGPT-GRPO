# 本次修改摘要

## 需求
- 修复 `our_work/pretrain/scripts/run_pretrain.py` 在单卡直接执行时误触发分布式初始化的问题。
- 当前现象是在 `python run_pretrain.py ...` 下，`TrainingArguments` 因为读取到 `ddp_backend` 或脏的 `LOCAL_RANK/WORLD_SIZE` 环境变量而走进 `accelerate` 分布式分支，最终报 `Default process group has not been initialized`。

## 实际修改
- `our_work/pretrain/scripts/run_pretrain.py`
  - 新增 `_distributed_training_requested()`，只在真实多卡环境下开启 DDP 相关参数。
  - 新增 `_sanitize_single_process_distributed_env()`，单进程路径会清理 `LOCAL_RANK/RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT`。
  - `build_trainer(...)` 里现在只有在真实分布式环境下才把 `ddp_backend` 和 `ddp_find_unused_parameters` 传给 `TrainingArguments`。
- `tests/our_work/pretrain/test_training_smoke.py`
  - 新增回归测试：即使环境里残留 `LOCAL_RANK=0`、`WORLD_SIZE=1` 等变量，单进程构造 Trainer 也不应再误走 DDP。
- `README.md`
  - 补充说明：`base_train.yaml` 里的 `distributed.*` 只有在 `torchrun` 多卡场景下才会生效，单进程会忽略。

## 说明
- 这次修复不改训练算法，也不改多卡配置本身；只修正“单进程误进入 DDP 路径”的问题。
- 多卡仍应使用 `torchrun --nproc_per_node=... our_work/pretrain/scripts/run_pretrain.py ...`。

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -m compileall our_work/pretrain tests/our_work/pretrain`
  - 通过
- 手工 smoke
  - 人为设置 `LOCAL_RANK=0`、`RANK=0`、`WORLD_SIZE=1`
  - 调用 `build_trainer(...)` 且传入 `ddp_backend='nccl'`
  - 结果：成功构造 Trainer，`ddp_backend` 被忽略，脏环境变量被清理
- `pytest`
  - 当前 Windows 环境仍会在 session 收尾阶段被临时目录权限问题打断，没拿到干净退出码；本次修复相关逻辑已用手工 smoke 验证。

## Git
- branch: `fix/pretrain-single-process-ddp-guard`
- commit: pending
