# Notes

## 需求
整理并收缩当前仓库文档体系，仅保留 README.md、AGENTS.md、docs/notes.md、docs/logs/2026-03.md，并合并旧文档中的有效信息。

## 修改文件
- README.md
- AGENTS.md
- docs/notes.md
- docs/logs/2026-03.md
- TMM/README.md

## 修改内容
- 重写 README.md，按当前代码与配置压缩项目说明、运行方式、配置入口和输出路径。
- 新增 AGENTS.md 与 docs 记录文件，并删除被吸收的 TMM/README.md。

## 验证
```bash
rg --files -g "*.md"
```

结果：通过

## Git
- branch: `docs/doc-system-shrink`
- commit: `git commit -m "docs: consolidate project documentation"`
