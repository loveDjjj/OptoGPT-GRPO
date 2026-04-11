# 本次修改摘要

## 需求
- 为 `our_work/pretrain/scripts/run_eval.py` 设计批量结果落盘与可视化规范。
- 默认输出需要支持 `summary.json + results.jsonl + PNG`，并包含分层统计与样本光谱对比图。

## 实际修改
- `docs/superpowers/specs/2026-04-11-run-eval-batch-visualization-design.md`
  - 新增 `run_eval.py` 批量可视化与结果落盘设计。
  - 明确输出目录结构、`summary.json` / `results.jsonl` schema、图表集合、样本抽样策略、容错规则与 CLI 扩展项。
- `docs/notes.md`
  - 覆盖为本次设计摘要。
- `docs/logs/2026-04.md`
  - 追加本次设计记录。

## 说明
- 本次仅落设计文档，不修改业务代码。
- 设计已固定为单入口 `run_eval.py`，避免后续服务器批处理拆成两步。

## 验证
- 未验证

## Git
- branch: `feat/our-work-bootstrap`
- commit: 待提交
