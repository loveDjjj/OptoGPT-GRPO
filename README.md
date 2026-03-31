# OptoGPT Spectral GRPO

## 项目定位
本项目当前保留两条主线：

- `光谱评测`
- `基于光谱 reward 的 spectral GRPO 训练`

基座模型是 [model/optogpt.pt](/O:/Optics%20Code/OptoGPT-GRPO/model/optogpt.pt)，它本身已经是用 `CE/SFT` 预训练好的 OptoGPT。  
当前训练路径在该基座上继续做基于目标光谱的 group-relative policy optimization。

## 当前目录
- `configs/eval/`
  光谱评测配置。
- `configs/grpo/`
  光谱 GRPO 训练配置。
- `configs/sft/`
  兼容旧路径的训练配置副本，建议改用 `configs/grpo/`。
- `runners/`
  运行入口；`run_grpo.py` 为当前训练主入口。
- `models/optogpt/`
  基座模型加载、生成、policy 定义、teacher forcing / policy-aware 打分、checkpoint 导出。
- `datasets/`
  光谱-结构成对数据集、切分与分布式 sampler。
- `evaluators/`
  光谱评测逻辑与指标聚合。
- `trainers/`
  GRPO 训练器与兼容封装。
- `losses/`
  序列损失、GRPO 目标与光谱损失。
- `physics/`
  原 `TMM/` 模块整体迁移后的物理计算代码。
- `data/materials/`
  材料库。
- `dataset/`
  当前使用的 `Spectrum_*.npy` 与 `Structure_*.npy`。
- `core/`
  旧 checkpoint 兼容层，保留但不扩展新逻辑。

## 数据说明
当前默认使用：

- 训练集：
  [dataset/Spectrum_train.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Spectrum_train.npy)
  [dataset/Structure_train.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Structure_train.npy)
- 验证集：
  [dataset/Spectrum_test.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Spectrum_test.npy)
  [dataset/Structure_test.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Structure_test.npy)

如果后续需要严格划分 `train/val/test`，可以：

- 直接新增独立 `val` 文件
- 或在配置里启用 `data.val_ratio`

## 入口
### 1. 光谱评测
功能：

- 输入目标光谱
- 生成结构
- 计算真实结构的序列损失
- 计算生成结构对应的光谱损失
- 输出样本级结果与汇总统计

命令：

```bash
python runners/run_spectrum_eval.py --config configs/eval/spectrum_eval.yaml
```

多卡：

```bash
torchrun --nproc_per_node=4 runners/run_spectrum_eval.py --config configs/eval/spectrum_eval.yaml
```

### 2. 光谱 GRPO 训练
功能：

- 对每条目标光谱 rollout 采样一组结构候选
- 用同一 policy 定义记录 old logprobs
- 用 TMM 计算每个候选结构的光谱 loss，并转成 reward
- 在同一 target spectrum 的组内做 reward 中心化 / 标准化 advantage
- 用 PPO-style clipped objective 更新模型

命令：

```bash
python runners/run_grpo.py --config configs/grpo/spectral_grpo.yaml
```

多卡：

```bash
torchrun --nproc_per_node=4 runners/run_grpo.py --config configs/grpo/spectral_grpo.yaml
```

兼容旧入口：

```bash
python runners/run_spectral_sft.py --config configs/sft/spectral_sft.yaml
```

## 多卡建议
当前模型规模不大，最合适的并行方式是 `DDP 数据并行`，不是模型并行。

- 开发调试：`1-2 卡`
- 正式训练：`4 卡`通常最均衡
- 大规模评测：`4-8 卡`都可以
- 训练阶段默认跳过 `Structure_train.npy` 加载，避免每个 rank 重复占用大块主机内存
- rollout / scoring / TMM 都尽量按大 batch 批处理，优先提高 GPU 利用率与吞吐
- 如果要长期跑 `4-8 卡`，更推荐 `Linux + NCCL`；当前 Windows 环境会退回到 `Gloo`

## 输出目录
光谱评测输出：

- `outputs/eval/<experiment>_<timestamp>/config.snapshot.yaml`
- `outputs/eval/<experiment>_<timestamp>/metrics/*.csv`
- `outputs/eval/<experiment>_<timestamp>/samples/*.jsonl`
- `outputs/eval/<experiment>_<timestamp>/plots/<split>/rankXX/*.png`
- `outputs/eval/<experiment>_<timestamp>/plots/summary/<split>_distribution.png`

光谱 GRPO 训练输出：

- `outputs/grpo/<experiment>_<timestamp>/config.snapshot.yaml`
- `outputs/grpo/<experiment>_<timestamp>/metrics/*.csv`
- `outputs/grpo/<experiment>_<timestamp>/checkpoints/best.pt`
- `outputs/grpo/<experiment>_<timestamp>/checkpoints/final.pt`

## 依赖
运行前请确认以下依赖可用：

- `python`
- `torch`
- `PyYAML`
- `numpy`
- `scipy`

可选：

- `matplotlib`
  仅在需要画图时使用。

## 说明
- 当前 `physics/` 直接复用原 TMM 模块，不另起一套实现。
- 当前训练目标不是传统 teacher forcing CE，而是基于目标光谱 reward 的 GRPO。
- rollout 与 update 现在共用同一 policy 定义，不再出现 “filtered rollout / raw scoring” 的不一致。
- 当前默认的光谱误差是 `R/T` 直接误差，即比较拼接后的 `[R..., T...]` 光谱。
- `core/` 保留的主要目的，是兼容旧 OptoGPT checkpoint 的加载。
