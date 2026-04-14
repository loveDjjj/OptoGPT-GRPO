# 本次修改摘要

## 需求
- 给 `our_work/pretrain` 增加训练过程中的实时可视化与持久化指标记录。
- 目标方案：
  - TensorBoard 实时查看
  - 持续落盘 `metrics/*.jsonl`、`metrics/*.csv`
  - 训练结束和每次验证后自动生成 PNG 曲线图
- 要求尽量只记录标量和已预处理 token ids，避免把大 logits 从 GPU 搬回 CPU。

## 实际修改
- `our_work/pretrain/monitoring.py`
  - 新增 `PretrainVisualizationCallback`
  - 功能：
    - 训练/验证标量写 TensorBoard
    - 持续写 `metrics/train_metrics.jsonl`
    - 持续写 `metrics/eval_metrics.jsonl`
    - 同步维护 `train_metrics.csv` / `eval_metrics.csv`
    - 每次 eval 后和训练结束后自动生成：
      - `train_loss.png`
      - `learning_rate.png`
      - `grad_norm.png`
      - `eval_loss.png`
      - `eval_token_accuracy.png`
      - `overview.png`
  - `SummaryWriter` 改成懒加载，避免模块导入阶段就拉起重依赖。
- `our_work/pretrain/scripts/run_pretrain.py`
  - `Trainer(...)` 新增挂载 `PretrainVisualizationCallback`
  - 从 YAML 的 `monitoring.*` 读取开关与 flush 参数
- `our_work/pretrain/configs/train/base_train.yaml`
  - 新增 `monitoring:` 配置段：
    - `tensorboard`
    - `jsonl`
    - `csv`
    - `save_plots`
    - `plot_every_eval`
    - `flush_secs`
- `tests/our_work/pretrain/test_training_smoke.py`
  - 新增 callback 回归测试，验证会写出 train/eval JSONL 与 CSV
- `README.md`
  - 补充默认监控配置说明
  - 补充训练产物目录：
    - `tensorboard/`
    - `metrics/*.jsonl`
    - `metrics/*.csv`
    - `plots/*.png`
  - 补充 `tensorboard --logdir ...` 使用命令
  - 补充 `tensorboard` 依赖说明

## 说明
- 实时可视化的主入口是 TensorBoard。
- 为减少 CPU/GPU 间搬运，训练和验证阶段只记录：
  - 标量
  - 已经预处理过的 token ids 级 metrics
- 不在可视化路径里传输完整 logits。

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe -m compileall our_work/pretrain tests/our_work/pretrain README.md`
  - 通过
- 手工 smoke
  - 构造 `PretrainVisualizationCallback`
  - 写入一次 train log 和一次 eval metrics
  - 结果：通过，输出 `pretrain-monitoring-smoke-ok`
- `D:\\anaconda\\envs\\oneday\\python.exe -c "import yaml, pathlib; cfg=yaml.safe_load(pathlib.Path('our_work/pretrain/configs/train/base_train.yaml').read_text(encoding='utf-8')); print(cfg['monitoring'])"`
  - 结果：`{'tensorboard': True, 'jsonl': True, 'csv': True, 'save_plots': True, 'plot_every_eval': True, 'flush_secs': 10}`

## Git
- branch: `feat/pretrain-live-monitoring`
- commit: pending
