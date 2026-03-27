# 本次修改摘要

## 需求
- 在 spectral SFT 训练结束后，自动绘制按 epoch 汇总的训练/验证曲线。
- 曲线至少包含：
  - 训练集光谱损失随 epoch 变化的 `R-RMSE`、`T-RMSE`
  - 验证集光谱损失随 epoch 变化的 `R-RMSE`、`T-RMSE`
  - 训练集、验证集 `sequence_loss` 随 epoch 变化
- 保持多卡兼容，只允许主进程写图和写汇总文件。

## 实际修改
- `losses/spectrum_loss.py`
  - 在批量评估生成结构时，新增 `r_rmse`、`t_rmse` 辅助数组。
  - 对无效结构、非物理解和 TMM 失败样本，`R/T` 误差统一回填为惩罚值，便于按 epoch 稳定汇总。
- `evaluators/metrics.py`
  - `MetricAccumulator` 新增 `R-RMSE / T-RMSE` 的和、均值、最小值、最大值。
  - 多卡归并时一并对 `R/T` 指标做 reduce。
- `evaluators/spectrum_evaluator.py`
  - 评测摘要 CSV 现在会写出 `mean_r_rmse`、`mean_t_rmse`。
  - 评测进度条会同时显示当前累计的 `seq/spec/r/t`。
- `trainers/spectral_sft_trainer.py`
  - 训练阶段按样本数汇总 epoch 指标，而不是简单平均 batch 均值。
  - 每个 epoch 额外记录：
    - `mean_train_sequence_loss`
    - `mean_train_r_rmse`
    - `mean_train_t_rmse`
    - `val_r_rmse`
    - `val_t_rmse`
  - 训练结束后由主进程自动绘制 `plots/epoch_summary.png`。
- `utils/plotting.py`
  - 新增 `save_sft_epoch_summary_plot(...)`，统一输出 epoch 级总览图。
- `configs/sft/spectral_sft.yaml`
  - 新增 `training.save_epoch_plots: true`，控制训练结束后是否保存 epoch 曲线。

## 输出
- 训练指标 CSV：
  - `outputs/sft/<run>/metrics/train_metrics.csv`
- 训练结束后的 epoch 曲线图：
  - `outputs/sft/<run>/plots/epoch_summary.png`

## 说明
- 当前训练集 `sequence_loss` 来自训练过程中生成序列的负对数概率均值。
- 当前验证集 `sequence_loss` 来自验证集真值结构的 teacher-forcing 评测。
- 两者都能反映模型变化趋势，但语义不完全相同，读图时要区分。

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -m compileall trainers evaluators losses utils runners models datasets`
- `@' ... MetricAccumulator + save_sft_epoch_summary_plot smoke test ... '@ | D:\\anaconda\\envs\\oneday\\python.exe -`
- `@' ... evaluate_generated_structures(..., return_aux_arrays=True) smoke test ... '@ | D:\\anaconda\\envs\\oneday\\python.exe -`

结果：通过

## Git
- branch: `feat/sft-epoch-curves`
- commit: `git commit -m "feat: add spectral sft epoch summary curves"`
