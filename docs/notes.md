# 本次修改摘要

## 需求
- 彻底删除旧的 SFT 兼容文件，只保留 GRPO 训练路径。
- 删除后仍保证训练入口、导出链路和代码引用都能走通。

## 实际修改
- 已删除：
  - `trainers/spectral_sft_trainer.py`
  - `runners/run_spectral_sft.py`
  - `configs/sft/spectral_sft.yaml`
  - 空目录 `configs/sft/`
  - 对应旧 `__pycache__` 产物
- `trainers/__init__.py`
  - 不再导出 `SpectralSFTTrainer`
  - 只保留 `GRPOTrainer`
- `utils/plotting.py`
  - 删除 `save_sft_epoch_summary_plot(...)` 兼容入口
  - 只保留 `save_grpo_epoch_summary_plot(...)`
- `README.md`
  - 删除旧 SFT 目录和旧训练入口说明
  - 只保留 `run_grpo.py + configs/grpo/spectral_grpo.yaml`
- `AGENTS.md`
  - 训练入口补读要求切到 `run_grpo.py` 和 `configs/grpo/spectral_grpo.yaml`

## 影响
- 当前仓库训练路径只剩 GRPO。
- 任何继续引用 `spectral_sft_trainer.py`、`run_spectral_sft.py`、`configs/sft/spectral_sft.yaml` 的外部脚本都会失效，需要改为 GRPO 路径。
- 仓库内部代码路径已切干净，不再依赖这些旧文件。

## 验证
- `rg -n "spectral_sft|SpectralSFT|run_spectral_sft|configs/sft|save_sft_epoch_summary_plot|outputs/sft" -S README.md AGENTS.md trainers runners configs models losses datasets evaluators utils docs/notes.md`
  - 结果：通过；非历史文档范围内无残留引用
- `python -m compileall trainers models runners losses evaluators utils datasets`
  - 结果：通过
- 运行态训练 smoke
  - 结果：未验证；当前终端默认 `python` 缺少 `torch`

## Git
- branch: `refactor/remove-sft-path`
- commit: `git commit -m "refactor: remove legacy sft training path"`
