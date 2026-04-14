# 本次修改摘要

## 需求
- 按指定配置把 `our_work/pretrain` 的默认模型参数改成：
  - `n_embd: 1024`
  - `n_layer: 6`
  - `n_head: 16`

## 实际修改
- `our_work/pretrain/configs/model/base_gpt.yaml`
  - `n_embd: 256 -> 1024`
  - `n_layer` 保持 `6`
  - `n_head: 8 -> 16`
  - 新增注释，说明这是一组更强的 decoder 容量配置，且 `1024 / 16 = 64`
- `README.md`
  - 同步更新 `base_gpt.yaml` 默认值说明：
    - `prefix_length: 8`
    - `n_embd: 1024`
    - `n_layer: 6`
    - `n_head: 16`
- `docs/notes.md`
  - 覆盖为本次模型配置调整摘要
- `docs/logs/2026-04.md`
  - 追加本次记录

## 说明
- 这次只改默认模型配置，不改训练流程、优化器、学习率或数据配置。
- 在当前设计下，光谱仍然先被 projector 编码成 `8` 个 prefix vectors；这次修改后，每个 prefix vector 的维度变成 `1024`。
- 相比参考模型常见的“单个 1024 维光谱 embedding”思路，我们当前仍是“8 个 1024 维 prefix tokens”设计。

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -c "import yaml, pathlib; cfg=yaml.safe_load(pathlib.Path('our_work/pretrain/configs/model/base_gpt.yaml').read_text(encoding='utf-8')); print(cfg['model']['n_embd'], cfg['model']['n_layer'], cfg['model']['n_head'])"`
  - 结果：`1024 6 16`

## Git
- branch: `config/pretrain-model-1024x6x16`
- commit: pending
