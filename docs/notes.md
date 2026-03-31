# 本次修改摘要

## 需求
- 修复当前默认 GRPO 训练在第一批 reward 计算时会因为多余参数报错的问题。

## 实际修改
- `trainers/grpo_trainer.py`
  - 删除传给 `evaluate_generated_structures_torch(...)` 的无效参数 `return_spectra`
  - 保持 torch-only reward 链路其余参数不变

## 影响
- 当前默认 GRPO 训练不再因为 `unexpected keyword argument 'return_spectra'` 在第一批 reward 阶段直接失败。
- 该修复只影响训练期 torch-only reward 路径，不影响 evaluator 使用的 numpy/item-results 路径。

## 验证
- `python -m py_compile trainers/grpo_trainer.py`
  - 结果：通过
- `rg -n "\"return_spectra\"" trainers/grpo_trainer.py`
  - 结果：无命中
- 运行态训练 smoke
  - 结果：未验证；当前终端默认 `python` 缺少 `torch`

## Git
- branch: `fix/grpo-reward-kwargs`
- commit: `git commit -m "fix: remove stale reward kwargs from grpo trainer"`
