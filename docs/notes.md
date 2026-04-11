# 本次修改摘要

## 需求
- 为 `our_work/data_gen` 的 “GPU 分块采样 + 分块 TMM + bucket 内全局去重补采” 方案落正式设计文档。
- 固定后续实现会新增 `sampling.batch_size`、`tmm.batch_size`、GPU 结构采样、分块 TMM 和 chunk 级进度反馈。

## 实际修改
- `docs/superpowers/specs/2026-04-12-our-work-data-gen-gpu-batching-design.md`
  - 新增正式设计文档。
  - 覆盖当前问题、选定方案、配置设计、模块拆分、数据流、错误处理、测试策略与默认值。
- `docs/notes.md`
  - 覆盖为本次设计摘要。
- `docs/logs/2026-04.md`
  - 追加本次设计记录。

## 说明
- 设计明确采用“GPU 分块采样 + GPU 分块 TMM + bucket 内全局严格唯一补采”。
- 本次仅落设计文档，尚未开始代码实现。

## 验证
- 未验证

## Git
- branch: `spec/our-work-data-gen-gpu-batching`
- commit: pending
