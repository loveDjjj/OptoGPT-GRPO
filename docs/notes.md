# Notes

## 需求
优化单卡显卡利用率低、数据加载慢的问题；确认当前是否为多进程加载；将 TMM 结构统一填充到 20 层并写入 YAML；同时继续提高 batch 与数据吞吐。

## 修改文件
- datasets/collator.py
- models/optogpt/checkpoint.py
- models/optogpt/generation.py
- physics/structure.py
- losses/spectrum_loss.py
- trainers/spectral_sft_trainer.py
- evaluators/spectrum_evaluator.py
- configs/sft/spectral_sft.yaml
- configs/eval/spectrum_eval.yaml
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- 当前 DataLoader 本来就是多进程加载，只要 `num_workers > 0` 就会启用 worker 进程。
- 原实现虽然打开了 `pin_memory`，但 collator 返回的是 `numpy`，导致 pin memory 实际没有真正生效。
- 现在 collator 直接返回 `torch.Tensor`，并配合模型侧 `non_blocking=True` 搬运到 GPU，减少 CPU 到 GPU 的等待。
- 生成阶段补了 `torch.Tensor` batch 快路，避免把一整个 batch 先拆成 Python 列表再逐条转张量。
- TMM 结构 padding 新增 `fixed_max_layers`，并在训练/评测配置里固定为 `20` 层。
- 固定 20 层后，关闭 `bucket_by_layer_count`，减少分桶开销和 batch 形状抖动。
- 默认提高了单卡吞吐相关配置：
  - `data.num_workers: 8`
  - `data.prefetch_factor: 4`
  - `sampling.*.batch_size: 128`
  - `training.batch_size: 32`
  - `evaluation.batch_size/scoring_batch_size: 64/128`
  - `tmm.batch_size: 256`

## 验证
```bash
D:\anaconda\envs\oneday\python.exe -c "import numpy as np; from datasets.collator import optogpt_batch_collator; batch=optogpt_batch_collator([{'sample_index': 1, 'spectrum': np.zeros(142, dtype=np.float32), 'structure_tokens': ['A_10']}, {'sample_index': 2, 'spectrum': np.ones(142, dtype=np.float32), 'structure_tokens': ['B_20']}]); print(type(batch['spectra']).__name__, batch['spectra'].dtype, tuple(batch['spectra'].shape)); print(type(batch['sample_indices']).__name__, batch['sample_indices'].dtype, tuple(batch['sample_indices'].shape))"
```

结果：通过，`spectra` 与 `sample_indices` 都已变为 `Tensor`

```bash
D:\anaconda\envs\oneday\python.exe -c "import numpy as np; from losses.spectrum_loss import evaluate_generated_structures; tokens=[['Air_0']*21]; target=np.zeros((1,142), dtype=np.float32); result=evaluate_generated_structures(tokens, target, fixed_max_layers=20, return_spectra=False); print(result[0]['status'], result[0]['padded_layer_count'], result[0]['spectrum_loss'])"
```

结果：通过，返回 `too_many_layers>20`

```bash
D:\anaconda\envs\oneday\python.exe -c "import compileall; paths=['runners','models','datasets','evaluators','losses','physics','trainers','utils','scripts','core']; ok=True
for path in paths:
    result=compileall.compile_dir(path, quiet=1)
    print(f'{path}: {result}')
    ok = ok and result
print(f'overall: {ok}')"
```

结果：通过

## Git
- branch: `perf/fixed-20layer-and-loader-throughput`
- commit: `git commit -m "perf: increase throughput with fixed 20-layer padding"`
