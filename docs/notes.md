# 本次修改摘要

## 需求
- 将可获得完整 nk 数据的材料扩展到 25 um，并新增 2-25 um 六波段 PSO 多任务配置。

## 实际修改
- 更新 `our_work/_shared/database/` 中 17 种材料的无表头三列 Excel nk 表：
  - 每种材料使用 refractiveindex.info 中单一、可追溯且完整覆盖 2-25 um 的数据集。
  - 未找到完整覆盖数据的材料保持原文件不变，不进入新 PSO 配置。
  - 新增 `SOURCES_2_25.md` 记录数据集、样品形态和数据库 commit。
- `our_work/pso/targets.py` 新增配置化六波段二元目标生成：
  - 支持自定义波段边界、最大相邻跳变次数、排除全低目标。
  - 新配置默认生成 31 个任务。
- 新增 `our_work/pso/configs/pso_2_25_binary_bands.yaml`：
  - 波段为 2-3、3-5、5-8、8-13、13-16、16-25 um。
  - TMM 范围为 2-25 um、1024 点。
  - 显式限定 17 种完整覆盖材料。
- 补充 PSO 目标构造和配置入口测试，并更新 README。

## 验证
- 17 张 Excel 回读、波长严格递增和 2-25 um 覆盖检查通过。
- `pytest tests/our_work/pso/test_targets.py -q`：5 passed。
- Python 语法编译通过。
- 本地 `oneday` 环境未安装 torch，因此完整 PSO/TMM 测试和服务器烟测待确认。

## Git
- branch: main
- commit: pending
