# OptoGPT Spectral SFT

## 项目定位
本项目当前只保留两条主线：

- `光谱测评`
- `基于光谱损失的 spectral SFT`

这里的基座模型是 [model/optogpt.pt](/O:/Optics%20Code/OptoGPT-GRPO/model/optogpt.pt)，它本身已经是别人用 `CE/SFT` 预训练好的 OptoGPT。  
当前仓库不再包含 `GRPO / PPO / Reward` 训练流程。

## 当前目录
- `configs/eval/`
  光谱测评配置。
- `configs/sft/`
  光谱损失微调配置。
- `runners/`
  运行入口。
- `models/optogpt/`
  基座模型加载、生成、teacher forcing 打分、checkpoint 导出。
- `datasets/`
  光谱-结构成对数据集、切分与分布式 sampler。
- `evaluators/`
  光谱测评逻辑与指标聚合。
- `trainers/`
  基于光谱损失的训练器。
- `losses/`
  序列损失与光谱损失。
- `physics/`
  原 `TMM/` 模块整体迁移后的物理计算代码。
- `data/materials/`
  原 `nk/` 材料库。
- `dataset/`
  当前使用的 `Spectrum_*.pkl` 与 `Structure_*.pkl`。
- `core/`
  旧 checkpoint 兼容层，保留但不扩展新逻辑。

## 数据说明
当前默认使用：

- 训练集：
  [dataset/Spectrum_train.pkl](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Spectrum_train.pkl)
  [dataset/Structure_train.pkl](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Structure_train.pkl)
- 验证集：
  [dataset/Spectrum_test.pkl](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Spectrum_test.pkl)
  [dataset/Structure_test.pkl](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Structure_test.pkl)

如果后续需要严格划分 `train/val/test`，可以：

- 直接新增独立 `val` 文件
- 或在配置里启用 `data.val_ratio`

## 两个入口
### 1. 光谱测评
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

### 2. 光谱损失微调
功能：

- 读取训练集光谱
- 用当前模型生成结构
- 用 TMM 计算生成结构的光谱损失
- 用去中心化后的光谱损失加权生成序列的对数概率，更新模型

命令：

```bash
python runners/run_spectral_sft.py --config configs/sft/spectral_sft.yaml
```

多卡：

```bash
torchrun --nproc_per_node=4 runners/run_spectral_sft.py --config configs/sft/spectral_sft.yaml
```

## 多卡建议
当前模型规模不大，最合适的并行方式是 `DDP 数据并行`，不是模型并行。

- 开发调试：`1-2 卡`
- 正式训练：`4 卡`最合适
- 大规模评测：`4-8 卡`都可以

如果没有 NVLink，也仍然可以跑；只是对当前任务来说，`4 卡`通常比 `8 卡`更均衡。

## 输出目录
光谱测评输出：

- `outputs/eval/<experiment>_<timestamp>/config.snapshot.yaml`
- `outputs/eval/<experiment>_<timestamp>/metrics/*.csv`
- `outputs/eval/<experiment>_<timestamp>/samples/*.jsonl`

光谱损失微调输出：

- `outputs/sft/<experiment>_<timestamp>/config.snapshot.yaml`
- `outputs/sft/<experiment>_<timestamp>/metrics/*.csv`
- `outputs/sft/<experiment>_<timestamp>/checkpoints/best.pt`
- `outputs/sft/<experiment>_<timestamp>/checkpoints/final.pt`

## 依赖
运行前请确认以下依赖可用：

- `python`
- `torch`
- `PyYAML`
- `numpy`
- `scipy`

可选：

- `matplotlib`
  仅在后续需要画图时使用。

## 说明
- 当前 `physics/` 直接复用原 TMM 模块，不另起一套实现。
- 当前训练目标不是传统 teacher forcing CE，而是基于生成结构光谱误差的微调。
- `core/` 保留的唯一目的，是兼容旧 OptoGPT checkpoint 的加载。
