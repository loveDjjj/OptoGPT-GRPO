# 本次修改摘要

## 需求
- 修复 `README.md` 中文乱码，并保持当前拍平后的仓库路径说明可用。

## 实际修改
- 从历史中未损坏的 README 内容恢复中文文本，重新写为带 BOM 的 UTF-8，兼容 Windows PowerShell 和常见编辑器显示。
- 更新 README 中的目录概览、入口命令和服务器部署说明，改为当前 `_shared/`、`data_gen/`、`pretrain/`、`rl/`、`eval/`、`pso/`、`ga/` 拍平结构。
- 保留当前配置仍在使用的 `outputs/our_work/...` 产物目录名，仅清理旧源码包路径和旧 `our_work/` 目录说明。

## 验证
- `python` 按 `utf-8-sig` 读取 README，乱码特征计数为 0。
- `Get-Content README.md -TotalCount 35` 可直接显示中文。
- 扫描 README 中非输出目录的 `our_work` 残留，无命中。

## Git
- branch: our_work
- commit: pending
