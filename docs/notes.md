# Notes

## 需求
继续优化评测与训练阶段的速度，重点减少 GPU 空转：
- 减少评测/训练中的强制同步点
- 将样本级统计改成批量累计
- 在保证统计图输出的前提下，降低逐样本绘图开销

## 修改文件
- evaluators/metrics.py
- evaluators/spectrum_evaluator.py
- trainers/spectral_sft_trainer.py
- configs/eval/spectrum_eval.yaml
- README.md
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- 训练阶段去掉了每个 batch 都做 `.cpu().item()` 的同步，改为：
  - batch 级结果先留在 GPU 上累计
  - 只在 `log_interval` 和 epoch 结束时做必要同步
- 评测阶段把样本级统计改成批量累计：
  - `MetricAccumulator` 新增 `update_batch`
  - `DistributionPlotAccumulator` 新增 `update_batch`
- 评测阶段的 `R-RMSE`、`T-RMSE`、序列误差、长度关系图，现在由各 rank 本地累计计数，再 all-reduce 汇总后只画一次。
- 默认关闭逐样本折线图：
  - `evaluation.save_plots: false`
  - 保留汇总统计图 `evaluation.save_distribution_plots: true`
- 汇总统计图输出路径：
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
D:\anaconda\envs\oneday\python.exe -c "import numpy as np; from evaluators.metrics import MetricAccumulator, DistributionPlotAccumulator; m=MetricAccumulator(); m.update_batch(np.array([1.0,2.0],dtype=np.float32), np.array([0.1,0.2],dtype=np.float32), np.array([True,False])); print(m.sample_count, m.valid_structure_count, round(m.sequence_loss_sum,3), round(m.spectrum_loss_sum,3)); d=DistributionPlotAccumulator(rt_rmse_bins=10, rt_rmse_max=1.0, sequence_loss_bins=10, sequence_loss_max=5.0, length_max=20); d.update_batch(np.array([0.1,0.9]), np.array([0.2,0.8]), np.array([1.0,4.0]), np.array([5,7]), np.array([6,8])); print(int(d.r_rmse_hist.sum()), int(d.t_rmse_hist.sum()), int(d.sequence_loss_hist.sum()), int(d.length_heatmap.sum()))"
```

结果：通过

## Git
- branch: `perf/reduce-sync-and-vectorize-eval`
- commit: `git commit -m "perf: reduce sync points and vectorize evaluation stats"`
