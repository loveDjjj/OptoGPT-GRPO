# 本次修改摘要

## 需求
- 给 `our_work` 补一份可直接用于服务器部署与运行的 README 指南。
- 检查 `our_work` 代码里长耗时阶段的进度反馈，给缺失的地方补上进度条。

## 实际修改
- `README.md`
  - 新增 `our_work 服务器部署与运行` 章节。
  - 覆盖必须同步的目录、依赖安装、关键配置约束、从零部署步骤、只评测已有 checkpoint 的步骤、常见报错与每一步的终端输出/产物说明。
- `our_work/data_gen/pipeline/build_dataset.py`
  - 新增按 layer bucket 显示的 `tqdm` 进度条。
  - 在进度条 postfix 中显示当前层数、生成样本数、有效样本数和累计保留条数。
- `our_work/data_gen/scripts/run_build_dataset.py`
  - 新增从 YAML `logging.show_progress_bar` 读取数据生成进度条开关。
- `our_work/data_gen/configs/dataset_v1.yaml`
  - 新增 `logging.show_progress_bar: true` 配置项。
- `our_work/pretrain/scripts/run_eval.py`
  - 新增按样本数显示的评测 `tqdm` 进度条。
  - 新增 `--disable-progress` 选项。
  - 在进度条 postfix 中显示当前有效生成数、精确匹配数与最近一个样本的 RMSE。
- `docs/notes.md`
  - 覆盖为本次 README 与进度条修改摘要。
- `docs/logs/2026-04.md`
  - 追加本次记录。

## 说明
- `run_pretrain.py` 训练阶段本身依赖 `transformers.Trainer`，已有默认训练进度条，因此本次未重复包一层。
- 本次主要补齐的是数据生成和独立评测这两条此前没有显式进度反馈的长耗时主循环。
- `our_work/_shared/database/` 当前仍是未跟踪目录，本次未修改。
- `.tmp_pytest` 目录仍有本机权限异常，本次未清理。

## 验证
- `python -c "import tqdm; print(tqdm.__version__)"`
  - 结果：通过
- `python -m compileall README.md our_work tests/our_work`
  - 结果：通过
- 内联 smoke：
  - 调用 `build_small_dataset(..., show_progress=False)` 生成临时数据集
  - 检查 `splits/split_manifest.json` 与 `vocab/vocab.json`
  - 结果：通过

## Git
- branch: `docs/our-work-server-guide`
- commit: pending
