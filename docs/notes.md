# 本次修改摘要

## 需求
- 调整 eval 逐样本光谱图的绘制方式。
- 不再直接用原始离散点连折线，而是按给定光谱点做平滑处理后再画图。
- 光谱强度坐标范围固定在 `0-1`。
- 同时把 TMM 精度控制项显式写到 `configs/eval/spectrum_eval.yaml`。

## 实际修改
- `utils/plotting.py`
  - 为 `save_spectrum_comparison_plot(...)` 新增基于光谱点的平滑插值能力。
  - 默认使用 `PCHIP` 做形状保持插值，避免普通三次样条过冲。
  - 新增可配置项：
    - `smooth_curves`
    - `smoothing_method`
    - `smoothing_upsample_factor`
    - `show_original_points`
    - `clip_to_unit_interval`
  - 每个子图的强度坐标范围固定为 `0.0 ~ 1.0`。
  - 平滑后曲线会裁剪到 `[0, 1]`，避免插值过冲影响观感。
- `evaluators/spectrum_evaluator.py`
  - 读取 `evaluation.plot_smoothing` 配置，并在保存逐样本图时传给绘图函数。
- `configs/eval/spectrum_eval.yaml`
  - 新增 `evaluation.plot_smoothing` 配置段。
  - 在 `tmm` 段显式补充 `complex_dtype` 说明与默认值。

## 新增配置项
```yaml
evaluation:
  plot_smoothing:
    enabled: true
    method: pchip
    upsample_factor: 8
    show_original_points: true

tmm:
  complex_dtype: complex64
```

## 说明
- 当前逐样本图仍然使用原始 `R / T / A` 三个子图布局。
- 平滑只作用于展示，不影响评测数值本身。
- 若后续要排查“原始点是否异常”，可保留 `show_original_points: true`；若只想要干净曲线，可改成 `false`。

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -m compileall utils evaluators`
- `@' ... save_spectrum_comparison_plot(...) smoke test ... '@ | D:\\anaconda\\envs\\oneday\\python.exe -`

结果：通过

## Git
- branch: `feat/eval-plot-smoothing`
- commit: `git commit -m "feat: smooth eval spectrum plots and expose tmm precision"` 
