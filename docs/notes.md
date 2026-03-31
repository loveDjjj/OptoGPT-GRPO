# 本次修改摘要

## 需求
- 修正 GRPO ratio 定义，改为基于整条序列 logprob 求和，不再把长度归一化混入 clipped objective。
- 把训练期 reward 链路改成 torch-only，在 GPU 上直接计算 TMM 光谱误差，避免 numpy 往返。
- 给 TMM 暴露 `complex_dtype` 配置，并在 GRPO 配置中默认改为 `complex64`。
- 优化 rollout，至少做 active-row compaction，减少 EOS 后无效样本继续参与解码前向。

## 实际修改
- `trainers/grpo_trainer.py`
  - 训练期 reward 评估改用 `evaluate_generated_structures_torch(...)`
  - `old/current sequence logprob` 统一按未归一化求和参与 ratio
  - `normalize_logprob_by_length` 仅保留给日志里的 sequence loss
  - TMM 训练参数新增 `device` 和 `complex_dtype`
- `losses/spectrum_loss.py`
  - 新增/接通 torch-only 结构评估路径，直接在 GPU 上计算 `rt_rmse`
  - 修正 nonphysical penalty 的张量填充值与 `ok_mask` 写回逻辑
- `physics/spectrum.py`
  - 新增 torch 版 R/T 拆分、吸收率和光谱误差函数
- `physics/optical_calculator.py`
  - 新增 `resolve_complex_dtype(...)`
  - TMM batch 接口支持从配置读取 `complex64/complex128`
- `models/optogpt/generation.py`
  - rollout 解码改为 active-row compaction
  - EOS 后样本不再继续占用后续解码前向
- `evaluators/spectrum_evaluator.py`
  - 评测 TMM 同步接入 `device` 和 `complex_dtype`
- `configs/grpo/spectral_grpo.yaml`
  - 新增 `tmm.complex_dtype: complex64`
  - 明确 `normalize_logprob_by_length` 只影响日志，不影响 GRPO ratio

## 验证
- `python -m compileall trainers models losses evaluators physics utils`
  - 结果：通过
- `@' ... yaml.safe_load("configs/grpo/spectral_grpo.yaml") ... '@ | python -`
  - 结果：通过，可读到 `training.normalize_logprob_by_length=True` 和 `tmm.complex_dtype=complex64`
- 运行态 smoke
  - 结果：未验证；当前终端默认 `python` 缺少 `torch`

## Git
- branch: `fix/grpo-ratio-reward-rollout`
- commit: `git commit -m "fix: correct grpo ratio and gpu reward path"`
