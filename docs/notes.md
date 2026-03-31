# 本次修改摘要

## 需求
- 把现有 spectral SFT 训练路径升级成真正可切换的 GRPO 路径。
- 修复 rollout 与 update 的 policy 定义不一致问题。
- 保留现有 `model / generation / scoring / evaluator / TMM / checkpoint` 体系，并尽量提高批处理并行度。

## 代码修改结论
- 当前训练主入口已切到 `runners/run_grpo.py`，对应主配置为 `configs/grpo/spectral_grpo.yaml`。
- 旧的 `run_spectral_sft.py`、`configs/sft/spectral_sft.yaml`、`trainers/spectral_sft_trainer.py` 保留为兼容薄封装或兼容配置，实际逻辑已转发到 GRPO 路径。

## 关键实现
- `models/optogpt/policy.py`
  - 新增 rollout / scoring 共用的 policy 定义。
  - 统一处理 `temperature / top-k / top-p / greedy / sample`。
- `models/optogpt/generation.py`
  - rollout 采样改为直接从共享 policy 分布采样。
  - 生成结果同时记录 `policy_logprobs`，作为 GRPO 的 `old_logprobs`。
  - 多候选展开顺序改成按 target 分组，方便组内 advantage 计算。
- `models/optogpt/scoring.py`
  - `sequence_logprobs_multi_target_batch_tensor(...)` 新增 `decode_config` 参数。
  - 当传入 `decode_config` 时，会按与 rollout 完全一致的 policy 定义重算 current logprobs。
- `losses/grpo_loss.py`
  - 新增组内 advantage、序列级 logprob 聚合、clipped surrogate、clip fraction、approx KL 计算。
- `trainers/grpo_trainer.py`
  - 用 `reward = -spectrum_loss` 构造 reward。
  - 按同一 target spectrum 的候选组做 `center/zscore` advantage。
  - update 阶段使用 PPO/GRPO-style clipped objective。
  - 支持变长序列、EOS、padding mask。
  - 支持现有 gradient accumulation 流程，并修正累积梯度时的 loss 缩放。
  - rollout / score 默认共用 `policy_forward_mode: eval`，关闭 dropout 噪声但保留 autograd。
- `configs/grpo/spectral_grpo.yaml`
  - 新增 `sampling.rollout` 与 `sampling.eval` 区分训练与评测策略。
  - GRPO 默认训练 policy 为 `decode: sample, top_k: 0, top_p: 1.0, temperature: 1.0`。
  - 新增 `group_size / clip_epsilon / advantage_mode / advantage_eps / scoring_batch_size`。
  - rollout / scoring / TMM 的默认 batch 都调大到 `4096`，优先提高吞吐。
- `evaluators/spectrum_evaluator.py`
  - 评测时改为对 `self.model.model` 统一切 `eval()`，兼容 DDP 包装。
- `utils/plotting.py`
  - 新增 `save_grpo_epoch_summary_plot(...)`，保留旧名字兼容转发。

## 影响
- 训练不再是 “centered spectrum loss * raw logprob”。
- 现在的训练目标是按组采样、按组归一化 reward、再做 clipped policy update。
- rollout 和 update 共用同一 policy 定义，不再出现 “filtered rollout / raw scoring”。
- 评测入口与 TMM 链路保持可复用。

## 验证
- `python -m compileall trainers models runners losses evaluators utils datasets`
  - 结果：通过
- `@' ... load_yaml_config(\"configs/grpo/spectral_grpo.yaml\") ... '@ | python -`
  - 结果：通过
- `@' ... load_yaml_config(\"configs/sft/spectral_sft.yaml\") ... '@ | python -`
  - 结果：通过
- `python runners/run_grpo.py --help`
  - 结果：当前终端默认 `python` 缺少 `torch`，未完成运行态验证

## Git
- branch: `feat/spectral-grpo-path`
- commit: `git commit -m "feat: replace spectral sft with grpo training path"`
