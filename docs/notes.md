# 本次修改摘要

## 需求
- 在默认模型已调整为 `n_embd: 1024`、`n_layer: 6`、`n_head: 16` 的前提下，直接把默认训练学习率改成更适合这档模型的值。
- 同时给出当前这组默认模型的大致精确参数量。

## 实际修改
- `our_work/pretrain/configs/train/base_train.yaml`
  - `training.learning_rate: 5e-4 -> 1e-4`
  - 新增注释，说明这是针对默认 `1024/6/16` 模型的更稳妥默认值。
- `README.md`
  - 同步更新 `base_train.yaml` 默认学习率说明。
  - 顺手把默认 `logging_steps` 说明同步到当前 YAML 的 `1000`。
- `docs/notes.md`
  - 覆盖为本次学习率与参数量摘要。
- `docs/logs/2026-04.md`
  - 追加本次记录。

## 参数量说明
- 在当前默认设计下：
  - `spectrum_dim: 2048`
  - `prefix_length: 8`
  - `n_embd: 1024`
  - `n_layer: 6`
  - `n_head: 16`
- 若按当前默认 token 设计 `vocab_size = 1204` 估算，总参数量约为：
  - `99,059,712`
- 其中大致拆分：
  - backbone: `76,845,056`
  - projector: `20,981,760`
  - lm_head: `1,232,896`
- 说明：
  - 这意味着当前默认模型更接近 `100M` 档，不是 `60M` 档。
  - 由于 projector 仍是把光谱编码成 `8 x 1024` 的 prefix token 表示，所以参数会明显大于“单个 1024 维光谱 embedding”方案。

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -c "import yaml, pathlib; cfg=yaml.safe_load(pathlib.Path('our_work/pretrain/configs/train/base_train.yaml').read_text(encoding='utf-8')); print(cfg['training']['learning_rate'], cfg['training']['logging_steps'])"`
  - 结果：`0.0001 1000`

## Git
- branch: `config/pretrain-lr-1e4-for-1024x6x16`
- commit: pending
