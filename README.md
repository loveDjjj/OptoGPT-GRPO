# OptoGPT-GRPO 项目说明

这是一个基于 OptoGPT 的多目标薄膜结构强化学习项目。当前主线不再是“单个目标光谱单独优化”，而是：

1. 先生成一批门函数型目标光谱，构成 `train / eval` 数据集。
2. 加载一次 OptoGPT 作为初始策略。
3. 在 `train` 目标集上执行标准 GRPO 强化学习。
4. 在固定的 `eval` 目标集上做训练前、训练中、训练后的采样评估。
5. 输出前后对比曲线、评估指标、checkpoint 和 summary。

## 当前目标类型

当前目标模式是 `gate_dataset`，也就是一系列门函数型吸收率目标。

支持两类任务：

- `pass`
  指定门区间内吸收率高，门外吸收率低。

- `stop`
  指定门区间内吸收率低，门外吸收率高。

例如：

- `600-900nm` 吸收率为 1，其余为 0
- `600-900nm` 吸收率为 0，其余为 1

当前实现支持：

- 可变门宽
- 可变门位置
- `pass / stop` 两类任务混合采样
- 可选平滑边缘

## 当前目标构造方式

虽然优化目标是吸收率曲线 `A_target`，但 OptoGPT 仍然接收 `R/T` 光谱作为条件输入。

当前采用标准不透射目标假设：

- `T_target = 0`
- `R_target = 1 - A_target`

因此：

- `A_target = 1 - R_target - T_target`
- 模型输入仍然是 `71` 个反射点 + `71` 个透射点，共 `142` 维

## 当前奖励函数

当前 reward 只用吸收率 RMSE：

- 先由 `A = 1 - R - T` 得到预测吸收率
- 再和目标吸收率计算 `RMSE`
- 最终 `reward = -RMSE(A_pred, A_target)`

非法结构和非物理光谱仍然使用固定 penalty。

## 当前训练流程

### 1. 数据集生成

在 `configs/grpo_base.yaml` 中配置：

- `target.train_count`
- `target.eval_count`
- `target.width_min_nm`
- `target.width_max_nm`
- `target.edge_smooth_nm`
- `target.families`

程序会自动生成：

- `train_targets`
- `eval_targets`

### 2. 训练前评估

在固定 `eval_targets` 上：

- 对每个 target 先采样
- 去重
- TMM 计算
- 记录训练前最优误差和最优结构

### 3. 标准 GRPO 训练

每个训练 step：

1. 从 `train_targets` 中随机抽一个 `target_batch`
2. 对 batch 内每个 target：
   - oversample 候选结构
   - 按结构串去重
   - 保留前 `unique_candidates`
3. 用 TMM 计算 reward
4. 每个 target 内单独计算 group advantage
5. 聚合多个 target 的 loss，执行一次参数更新

### 4. 周期评估

每隔 `grpo.eval_interval` 步：

- 在固定 `eval_targets` 上重新评估
- 记录 `mean / median / min / max` 误差
- 如有更优的 eval mean error，则保存 `best_eval.pt`

### 5. 训练后评估

训练结束后再次对固定 `eval_targets` 进行采样和评估，并与训练前结果对比。

## 当前提速策略

### 1. 采样批量化

OptoGPT 采样已经改成 batch 解码，不再逐条样本前向。

相关配置：

- `sampling.batch_size`

### 2. logprob 批量化

GRPO 更新时，current/reference logprob 重算已改成 batch 版本。

相关配置：

- `grpo.logprob_batch_size`

### 3. TMM 只算 unique 结构

每个 target 先 oversample，再按结构串精确去重，只对 unique 结构运行 TMM。

### 4. pad 到同层数后统一跑 TMM

每个 TMM batch 内的结构会 pad 到当前批次的最大层数，pad 层使用：

- `Air`
- `thickness = 0`

相关配置：

- `tmm.batch_size`
- `tmm.pad_to_max_layers`
- `tmm.pad_material`

## 三个不同的 batch

当前项目里有 3 个不同位置的 batch 配置，含义不同：

- `sampling.batch_size`
  OptoGPT 采样时的并行 batch。

- `tmm.batch_size`
  TMM 计算时的 batch。

- `grpo.logprob_batch_size`
  GRPO 更新时，重算 token logprob 的 batch。

## 主要输出

运行后会在 `outputs/<实验名>_<时间戳>/` 下生成：

- `config.snapshot.yaml`
  本次运行配置快照

- `train_targets.jsonl`
  训练目标集

- `eval_targets.jsonl`
  评估目标集

- `rollouts.jsonl`
  所有 rollout 记录

- `metrics/before_eval.csv`
  训练前评估结果

- `metrics/after_eval.csv`
  训练后评估结果

- `metrics/train_metrics.csv`
  每个训练 step 的训练指标

- `metrics/eval_metrics.csv`
  每次 eval 的整体指标

- `summary.csv`
  训练前后对比汇总

- `plots/eval_target_*_before_after.png`
  每个 eval target 的训练前后对比图

- `plots/eval_mean_error.png`
  eval mean error 曲线

- `checkpoints/best_eval.pt`
  当前最佳 eval mean error 对应的 checkpoint

- `checkpoints/final.pt`
  训练结束时的 checkpoint

## 运行方式

```powershell
python run_grpo.py --config configs/grpo_base.yaml
```

## 依赖提醒

如果要保存图像，需要运行环境里安装 `matplotlib`。
