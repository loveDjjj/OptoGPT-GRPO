# Notes

## 需求
在评测阶段增加统计图，总结展示：
- R-RMSE 与数量的关系
- T-RMSE 与数量的关系
- 生成结构长度与原始结构长度的关系
- 序列误差与数量的关系

要求优先考虑速度，尽量利用多卡并行评测结果做汇总。

## 修改文件
- evaluators/metrics.py
- evaluators/spectrum_evaluator.py
- utils/plotting.py
- configs/eval/spectrum_eval.yaml
- README.md
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- 新增 `DistributionPlotAccumulator`，在每个 rank 本地直接累计直方图/热力图计数，不保存全量样本。
- 评测结束后使用 all-reduce 汇总各 rank 的计数，只在主进程绘制一次汇总图。
- 新增评测统计总览图，包含 2x2 子图：
  - `R-RMSE vs Count`
  - `T-RMSE vs Count`
  - `Generated Length vs Target Length`
  - `Sequence Loss vs Count`
- 在评测配置中新增：
  - `evaluation.save_distribution_plots`
  - `evaluation.distribution_plots.rt_rmse_bins`
  - `evaluation.distribution_plots.rt_rmse_max`
  - `evaluation.distribution_plots.sequence_loss_bins`
  - `evaluation.distribution_plots.sequence_loss_max`
  - `evaluation.distribution_plots.length_max`
- README 已同步补充统计图输出路径：
  - `outputs/eval/<experiment>_<timestamp>/plots/summary/<split>_distribution.png`

## 验证
```bash
D:\anaconda\envs\oneday\python.exe -c "import compileall; paths=['runners','models','datasets','evaluators','losses','physics','trainers','utils','scripts','core']; ok=True
for path in paths:
    result=compileall.compile_dir(path, quiet=1)
    print(f'{path}: {result}')
    ok = ok and result
print(f'overall: {ok}')"
```

结果：通过

```bash
D:\anaconda\envs\oneday\python.exe -c "import numpy as np; from pathlib import Path; from utils.plotting import save_eval_distribution_summary; out=Path('outputs/_debug/test_distribution.png'); save_eval_distribution_summary(path=out, split_name='debug', r_rmse_hist=np.arange(1,101), t_rmse_hist=np.arange(101,201), sequence_loss_hist=np.arange(1,101)[::-1], length_heatmap=np.arange(21*21).reshape(21,21), rt_rmse_max=1.0, sequence_loss_max=10.0, length_max=20); print(out.exists(), out)"
```

结果：通过；调试文件已删除

## Git
- branch: `feat/eval-distribution-plots`
- commit: `git commit -m "feat: add evaluation distribution summary plots"`
