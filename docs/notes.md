# 本次修改摘要

## 需求
- 检查 spectral SFT 训练中 rollout 采样与重算 `logprob` 是否使用了同一个策略前向。
- 若训练态 `dropout` 会导致两次前向对应不同策略，则修复该问题。

## 代码分析结论
- `trainers/spectral_sft_trainer.py` 在每个 epoch 开头把模型切到 `train()`。
- 训练 batch 内先调用 `generate_structures_for_targets(...)` 做 rollout，再调用 `sequence_logprobs_multi_target_batch_tensor(...)` 重算生成序列的 `logprob` 并反传。
- 原实现中，这两次前向都继承外层 `train()` 模式：
  - rollout 虽然用了 `torch.inference_mode()`，但不会关闭 `dropout`
  - scoring 虽然保留 autograd，但也仍然会使用另一份训练态 `dropout` mask
- 因此会出现“采样策略”和“反传策略”不完全一致的问题，额外增加方差。

## 实际修改
- `trainers/spectral_sft_trainer.py`
  - 新增 `training.policy_forward_mode` 配置读取与校验，仅支持 `train/eval`
  - 在 `_train_batch(...)` 内让 rollout 和 score 两次前向共用同一个策略前向模式
  - 默认使用 `eval` 前向模式关闭 `dropout`，但不影响 scoring 的 autograd 与 `backward()`
  - batch 结束后恢复进入 `_train_batch(...)` 前的模块模式，避免影响外层流程
  - 训练 epoch 开头改为显式对 `self.model.model` 调用 `train()`，兼容 DDP 包装
- `configs/sft/spectral_sft.yaml`
  - 新增 `training.policy_forward_mode: eval` 及注释

## 影响
- 默认情况下，rollout 采样和重算 `logprob` 将对应同一个无 `dropout` 的策略前向。
- 训练仍然会在 scoring 阶段正常计算梯度并更新参数。
- 若后续确实希望把 `dropout` 也视作策略随机性的一部分，可手动把配置切回 `train`。

## 验证
- `python -m compileall trainers models runners`

结果：待验证

## Git
- branch: `fix/sft-policy-forward-mode`
- commit: `git commit -m "fix: align spectral sft rollout and scoring policy mode"`
