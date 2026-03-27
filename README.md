# OptoGPT-GRPO

## 项目简介
`OptoGPT-GRPO` 是一个基于 OptoGPT checkpoint 和本地 TMM 求解器的多目标薄膜结构强化学习项目。当前主流程围绕 `gate_dataset` 合成吸收谱目标，生成 `train/eval` 目标集，加载预训练策略并用 GRPO 做数据集级训练，输出评估指标、对比图和 checkpoint。

## 项目结构
- `run_grpo.py`：训练入口。
- `configs/grpo_base.yaml`：主配置文件。
- `data/`：合成目标谱生成。
- `policy/`：OptoGPT checkpoint 封装与采样。
- `trainers/`：GRPO 训练与评估主循环。
- `rollouts/`：采样去重与分组。
- `rewards/`：TMM reward 计算。
- `TMM/`：传输矩阵求解与材料读取。
- `nk/`：材料光学常数 CSV 数据。
- `utils/`：配置、结构转换、日志与绘图工具。
- `core/`：旧版 OptoGPT checkpoint 兼容层。

## 关键文件说明
- `run_grpo.py`：读取 YAML、生成目标集、创建运行目录并启动训练。
- `configs/grpo_base.yaml`：集中定义路径、目标构造、采样、TMM、reward、GRPO、evaluation、plotting、checkpoints 和 logging。
- `data/spectrum_generator.py`：生成 `gate_dataset` 目标谱，当前支持 `pass` / `stop` 两类任务。
- `policy/optogpt_policy.py`：加载旧版 OptoGPT checkpoint，负责 batched sampling、logprob 重算和 checkpoint 导出。
- `trainers/grpo_trainer.py`：执行数据集级 GRPO、周期评估、CSV 指标写出、绘图和 checkpoint 保存。
- `rewards/tmm_reward.py`：把结构 token 转成 TMM 配置并按当前 reward 指标打分。
- `TMM/optical_calculator.py`、`TMM/TMM.py`：材料数据加载、折射率插值和 TMM 求解。
- `utils/structure.py`、`utils/logging.py`：结构解析、padding、JSONL/CSV 和运行目录输出。
- `core/transformer.py`、`core/train.py`：旧 checkpoint 兼容依赖，当前 GRPO 流程不直接训练这里的逻辑，但加载历史 checkpoint 仍会用到。

## 运行方法
主命令：

```bash
python run_grpo.py --config configs/grpo_base.yaml
```

运行前请先确认：
- `configs/grpo_base.yaml` 中 `paths.optogpt_checkpoint` 指向可用 checkpoint。当前默认值是 `model/optogpt.pt`，但仓库内未见 `model/` 目录，需自行补齐或改配置。
- 仓库内未见单独的依赖清单；按当前导入，至少需要 `torch`、`numpy`、`PyYAML`、`pandas`、`scipy`，保存图像时还需要 `matplotlib`。
- `TMM/demo.py` 是独立演示脚本，但脚本默认读取 `TMM/database`，当前仓库未见该目录，辅助运行方式待确认。

## 配置说明
- 主配置文件是 `configs/grpo_base.yaml`。
- `experiment` 控制实验名和随机种子。
- `paths` 控制 checkpoint、输出目录和材料库目录；当前材料库目录为 `nk/`。
- `target` 定义 `gate_dataset` 的波长范围、采样数量、门宽范围、平滑边缘和 `pass/stop` 家族。
- `sampling`、`tmm`、`reward`、`grpo` 分别控制候选采样、物理求解、奖励计算和策略更新。
- `evaluation`、`plotting`、`checkpoints`、`logging` 控制周期评估、图片输出、权重保存和日志文件名。
- `target.num_points` 与 `tmm.num_points` 需要保持一致；当前代码和配置都按这一约束组织。

## 输出说明
- 每次运行会创建 `outputs/<experiment.name>_<timestamp>/`。
- 关键输出包括 `config.snapshot.yaml`、`train_targets.jsonl`、`eval_targets.jsonl`、`summary.csv`。
- `metrics/` 下会写入 `before_eval.csv`、`after_eval.csv`、`train_metrics.csv`、`eval_metrics.csv`；若启用训练集对比，还会写 `before_train_compare.csv` 和 `after_train_compare.csv`。
- `plots/` 下会写 `eval_mean_error.png` 以及 before/after 对比图。
- `checkpoints/` 下会写 `best_eval.pt` 和 `final.pt`。
- `rollouts.jsonl` 只在 `logging.save_rollouts: true` 时生成。

## 阅读顺序
1. `README.md`
2. `configs/grpo_base.yaml`
3. `run_grpo.py`
4. `data/spectrum_generator.py`
5. `policy/optogpt_policy.py`
6. `trainers/grpo_trainer.py`
7. `rewards/tmm_reward.py`
8. `TMM/optical_calculator.py`
9. `TMM/TMM.py`
