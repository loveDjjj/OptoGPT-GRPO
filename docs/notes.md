# 本次修改摘要

## 需求
- 修复多卡 spectral SFT 在训练早期 TMM 阶段崩溃的问题。
- 当前报错由 `R/T` 统计新增逻辑触发：单段 `71` 点反射/透射曲线被误传给只接受拼接 `[R..., T...]` 光谱的误差接口。

## 实际修改
- `losses/spectrum_loss.py`
  - 修复 `r_rmse / t_rmse` 的计算方式。
  - 当前在 `evaluate_generated_structures(...)` 内，针对单段 `reflection`、`transmission` 曲线直接计算 RMSE：
    - `r_error = sqrt(mean((reflection - target_r)^2))`
    - `t_error = sqrt(mean((transmission - target_t)^2))`
  - 不再把单段 `71` 点曲线传给 `spectrum_error(..., metric='r_rmse/t_rmse')`，避免触发拼接光谱长度检查。

## 原因
- `physics/spectrum.py::spectrum_error(...)` 的 `r_rmse / t_rmse` 分支仍按拼接光谱 `[R..., T...]` 解析。
- 训练中新增的 epoch 汇总在 `losses/spectrum_loss.py` 里传入的是单段 `R` 或 `T` 曲线，因此在 Linux 多卡训练时会报：
  - `ValueError: 光谱长度必须为偶数，当前长度为 71`

## 影响
- 不影响原有总光谱损失 `rt_rmse` 计算。
- 只修正新增的 `R-RMSE / T-RMSE` 统计逻辑。
- 同步后可直接继续重跑训练，无需改 YAML。

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -m compileall losses evaluators trainers`
- `@' ... split_rt_spectrum + 单段 RMSE smoke test ... '@ | D:\\anaconda\\envs\\oneday\\python.exe -`

结果：通过

## Git
- branch: `fix/sft-rt-rmse-crash`
- commit: `git commit -m "fix: handle single-channel rt rmse in spectral sft"`