# Notes

## 需求
将项目从原先的 `GRPO/RL` 主线重构为：

- 光谱测评主线
- 基于光谱损失的 spectral SFT 主线

并删除 RL 相关代码，迁移 TMM 模块与材料库目录，补齐多卡运行入口。

## 修改文件
- README.md
- AGENTS.md
- configs/eval/spectrum_eval.yaml
- configs/sft/spectral_sft.yaml
- runners/run_spectrum_eval.py
- runners/run_spectral_sft.py
- models/optogpt/__init__.py
- models/optogpt/checkpoint.py
- models/optogpt/generation.py
- models/optogpt/scoring.py
- models/optogpt/export.py
- datasets/__init__.py
- datasets/optogpt_dataset.py
- datasets/collator.py
- datasets/distributed.py
- datasets/splits.py
- evaluators/__init__.py
- evaluators/metrics.py
- evaluators/spectrum_evaluator.py
- losses/__init__.py
- losses/sequence_loss.py
- losses/spectrum_loss.py
- physics/__init__.py
- physics/demo.py
- physics/optical_calculator.py
- physics/structure.py
- physics/spectrum.py
- trainers/__init__.py
- trainers/spectral_sft_trainer.py
- utils/logging.py
- utils/dist.py
- utils/seed.py
- scripts/merge_eval_results.py
- scripts/pkl_to_npy.py
- data/materials/*.csv
- 删除：run_grpo.py、configs/grpo_base.yaml、policy/、rewards/、rollouts/、trainers/grpo_trainer.py、data/spectrum_generator.py、utils/structure.py、TMM/、nk/

## 修改内容
- 删除 RL/GRPO 训练链路，保留 `core/` 作为旧 checkpoint 兼容层。
- 新增 `run_spectrum_eval.py`，支持输入光谱后计算：
  - 真实结构的序列损失
  - 生成结构的光谱损失
- 新增 `run_spectral_sft.py` 与 `SpectralSFTTrainer`，实现基于生成结构光谱误差的微调流程。
- 将原 `TMM/` 整体迁移到 `physics/`，将原 `nk/` 材料库迁移到 `data/materials/`。
- 新增 `models/optogpt/`、`datasets/`、`evaluators/`、`losses/` 等新主线目录。
- 新增 `utils/dist.py` 与分布式 runner，按 DDP 数据并行支持 1-8 卡运行。
- 重写 README 和 AGENTS 相关入口说明，切换到新的项目主线。
- 新增 `scripts/pkl_to_npy.py`，把超大 pkl 数据转换为 `.npy`，便于多卡场景下重复读取。

## 验证
```bash
@'
import compileall
paths = [
    'runners',
    'models',
    'datasets',
    'evaluators',
    'losses',
    'physics',
    'trainers',
    'utils',
    'core',
]
ok = True
for path in paths:
    result = compileall.compile_dir(path, quiet=1)
    print(f'{path}: {result}')
    ok = ok and result
print('overall:', ok)
'@ | python -
```

结果：通过

## Git
- branch: `refactor/spectral-sft-pipeline`
- commit: `git commit -m "refactor: replace grpo pipeline with spectral sft workflow"`
