# 本次修改摘要

## 需求
- 将 PSO 补充数据集相关命令写入 `README.md`。
- 命令需要覆盖 PSO 数据生成、可选多进程拆分运行、PSO 数据集分析和可视化。

## 实际修改
- `README.md`
  - 在默认服务器配置列表中新增 `our_work/pso/configs/pso_supplement.yaml`。
  - 在默认配置说明中补充 PSO 的路径、层数、厚度、目标光谱、搜索参数和 TMM 参数。
  - 新增 `Step 4.2: 生成 PSO 补充数据集`，包含单进程运行命令、输出目录、数据格式和多进程注意事项。
  - 新增 `Step 4.3: 分析 PSO 补充数据集`，包含分析 CLI、全量绘图参数和主要输出文件。
  - 修正原 README 中的 `说明：F` 笔误。
- `docs/notes.md`
  - 覆盖为本次 README 命令补充摘要。
- `docs/logs/2026-04.md`
  - 追加本次文档修改记录。

## 验证
- `git diff --check -- README.md docs/notes.md docs/logs/2026-04.md`
  - 结果：通过
- `rg -n "PSO|pso_supplement|run_analyze_pso" README.md`
  - 结果：通过，能检索到新增命令和配置说明。

## Git
- branch: `main`
- commit: pending (`docs: add pso commands to readme`)
