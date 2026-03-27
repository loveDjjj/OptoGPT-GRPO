# Notes

## 需求
继续优化 SFT 主链的吞吐，重点降低训练中验证阶段和光谱损失计算带来的 GPU 空转。

## 修改文件
- configs/sft/spectral_sft.yaml
- evaluators/spectrum_evaluator.py
- losses/spectrum_loss.py
- trainers/spectral_sft_trainer.py
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- SFT 训练中的验证默认关闭 `save_plots` 和 `save_distribution_plots`，避免每个 epoch 额外生成图像和统计图。
- 将 SFT 验证的 `evaluation.batch_size` 与 `scoring_batch_size` 默认提升到 `128`，利用无梯度验证阶段的显存余量。
- `evaluate_generated_structures(...)` 新增批量数组快路：
  - 支持只返回 `spectrum_losses` 和 `ok_mask`
  - 需要时再额外返回逐样本结果
  - 训练阶段不再为每个样本构造 Python dict
- `SpectrumEvaluator` 增加真值结构 token id 缓存，减少训练过程中重复验证时的 CPU 编码开销。
- `SpectrumEvaluator.evaluate(...)` 使用 `torch.inference_mode()`，并在不保存样本/图片时直接消费批量数组结果，减少 Python 遍历。
- `SpectralSFTTrainer._train_batch(...)` 直接消费光谱损失批量数组，避免训练阶段的二次 list/dict 提取。

## 验证
```bash
D:\anaconda\envs\oneday\python.exe -m compileall trainers evaluators losses runners models datasets utils
```

结果：通过

```bash
@'
import numpy as np
from losses import evaluate_generated_structures

out = evaluate_generated_structures(
    structure_token_groups=[['BAD_TOKEN']],
    target_spectra=np.zeros((1, 142), dtype=np.float32),
    database_path='data/materials',
    return_item_results=False,
    return_aux_arrays=True,
    return_spectra=False,
)
print(sorted(out.keys()))
print(out['spectrum_losses'].dtype, out['spectrum_losses'].shape, float(out['spectrum_losses'][0]))
print(out['ok_mask'].dtype, out['ok_mask'].shape, bool(out['ok_mask'][0]))

results, aux = evaluate_generated_structures(
    structure_token_groups=[['BAD_TOKEN']],
    target_spectra=np.zeros((1, 142), dtype=np.float32),
    database_path='data/materials',
    return_item_results=True,
    return_aux_arrays=True,
    return_spectra=False,
)
print(type(results).__name__, type(aux).__name__, results[0]['status'])
'@ | D:\anaconda\envs\oneday\python.exe -
```

结果：通过

## Git
- branch: `perf/sft-throughput-fastpath`
- commit: `git commit -m "perf: reduce spectral sft validation overhead"`
