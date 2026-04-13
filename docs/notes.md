# 本次修改摘要

## 需求
- 按当前 `our_work` 数据规模调整正式训练节奏，把 `base_train.yaml` 中过于频繁的 `logging_steps / eval_steps / save_steps` 改成能真正长时间训练的版本。

## 背景
- 当前默认数据规模约为：
  - `train ≈ 2700000`
  - `val ≈ 300000`
- 单卡默认训练配置：
  - `per_device_train_batch_size: 16`
  - `gradient_accumulation_steps: 2`
  - `per_device_eval_batch_size: 16`
- 在这组参数下：
  - 训练一个 epoch 约 `84375` 个 optimizer steps
  - 全量 val 评估一次约 `18750` 个 eval batches
- 原配置 `eval_steps: 500` / `save_steps: 500` 会导致训练每推进很短一段就被一次完整全量验证打断，评估时间远超训练时间。

## 实际修改
- `our_work/pretrain/configs/train/base_train.yaml`
  - `training.logging_steps: 50 -> 200`
  - `training.eval_steps: 500 -> 100000`
  - `training.save_steps: 500 -> 50000`
  - 新增注释，说明这组值是针对当前全量 `train/val` 规模的正式训练节奏。
- `README.md`
  - 同步更新 `base_train.yaml` 默认值说明。
- `docs/notes.md`
  - 覆盖为本次配置调整摘要。
- `docs/logs/2026-04.md`
  - 追加本次记录。

## 说明
- 这次只调整 `base_train.yaml` 的训练/评估/保存频率，不改 batch size、epoch 数、学习率或数据路径。
- 新配置的意图是：
  - 日志仍能持续观察
  - checkpoint 有足够恢复点
  - 全量 val 评估频率大幅下降，不再吞掉大部分训练时间

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -c "import yaml, pathlib; cfg=yaml.safe_load(pathlib.Path('our_work/pretrain/configs/train/base_train.yaml').read_text(encoding='utf-8')); print(cfg['training']['logging_steps'], cfg['training']['eval_steps'], cfg['training']['save_steps'])"`
  - 结果：`200 100000 50000`

## Git
- branch: `config/pretrain-formal-cadence`
- commit: pending
