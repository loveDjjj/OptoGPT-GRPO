# 本次修改摘要

## 需求
- 将已确认的 `our_work` 设计拆成可执行的实现计划。
- 由于 `data_gen` 与 `pretrain` 是相对独立的子系统，因此分别落两份计划文档。

## 实际修改
- `docs/superpowers/plans/2026-04-11-our-work-data-generation.md`
  - 新增 `data_gen` 实现计划。
  - 覆盖 `_shared` 迁移、材料 registry、采样、TMM 模拟、分片与 CLI。
- `docs/superpowers/plans/2026-04-11-our-work-pretraining.md`
  - 新增 `pretrain` 实现计划。
  - 覆盖 tokenizer、collator、自定义 HF 模型、generation/scoring、Trainer 入口。
- `docs/notes.md`
  - 覆盖为最近一次修改摘要。
- `docs/logs/2026-04.md`
  - 追加本次计划记录。

## 说明
- 本次仅新增计划文档，不改业务代码。
- 计划明确把实现拆成 `data_gen` 与 `pretrain` 两条执行线，便于后续分阶段落地。

## 验证
- 未验证

## Git
- branch: `docs/our-work-plans`
- commit: `git commit -m "docs: add our_work implementation plans"` 
