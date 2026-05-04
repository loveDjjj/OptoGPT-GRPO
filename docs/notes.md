# 本次修改摘要

## 需求
- 修复 4 卡评测中 `tmm.device=auto` 传入 `torch.device` 报错。
- 为多卡评测增加可见进度条。

## 实际修改
- `our_work/eval/pipeline.py`
  - 新增 `tmm.device` 解析逻辑：
    - `auto` 在多卡时解析为 `cuda:<LOCAL_RANK>`；
    - 单进程时解析为 `cuda` 或 `cpu`。
  - `_evaluate_records()` 中 TMM 计算改为 chunk 级进度条 `eval:tmm:<split>`。
  - `_predict_token_groups()` 增加 batch 级进度条 `eval:predict`。
  - split 主循环增加 rank 级进度条 `eval:rank<rank>`。
  - 所有进度条均在缺失 `tqdm` 时自动降级。

## 结果
- 解决 `RuntimeError: device type at start of device string: auto`。
- 评测可实时看到 split / 预测 / TMM 三层进度。

## Git
- branch: main
- commit: pending
