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
- `runners/`
  运行入口；`run_grpo.py` 为当前训练主入口。
- `models/optogpt/`
  基座模型加载、生成、policy 定义、teacher forcing / policy-aware 打分、checkpoint 导出。
- `datasets/`
  光谱-结构成对数据集、切分与分布式 sampler。
- `evaluators/`
  光谱评测逻辑与指标聚合。
- `trainers/`
  GRPO 训练器。
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

## our_work 服务器部署与运行
本节对应仓库根目录下的 [our_work](/O:/Optics%20Code/OptoGPT-GRPO/our_work) 独立数据生成、预训练与评测链路。当前默认配置已经改成服务器可直接运行的版本，不再依赖必须从仓库根目录启动；不过为了排查日志和产物更直观，仍然建议先 `cd` 到仓库根目录再执行。

### 1. 必须同步的目录
- 仓库代码本身
  - `git clone` 或 `git pull` 即可，`our_work/` 已经在主工作区根目录。
- `database/`
  - 这是材料库，当前不在 git 中。
  - 服务器只 `git clone` 不够，必须单独同步。

### 2. 服务器依赖
运行 `our_work` 链路前请确认以下 Python 依赖可用：

- `torch`
- `PyYAML`
- `numpy`
- `scipy`
- `pandas`
- `pyarrow`
- `openpyxl`
- `transformers`
- `safetensors`
- `Pillow`

推荐安装示例：

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas pyarrow openpyxl pyyaml pillow transformers safetensors
```

安装完成后可执行：

```bash
python -c "import torch,yaml,pandas,scipy,numpy,PIL,transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

典型终端输出：

```text
torch 2.x.x cuda True
```

### 3. 默认配置与关键约束
默认服务器配置文件：

- 数据生成：[dataset_v1.yaml](/O:/Optics%20Code/OptoGPT-GRPO/our_work/data_gen/configs/dataset_v1.yaml)
- 训练：[base_train.yaml](/O:/Optics%20Code/OptoGPT-GRPO/our_work/pretrain/configs/train/base_train.yaml)
- 模型：[base_gpt.yaml](/O:/Optics%20Code/OptoGPT-GRPO/our_work/pretrain/configs/model/base_gpt.yaml)

当前默认值：

- `dataset_v1.yaml`
  - `paths.database_dir: database`
  - `paths.output_dir: outputs/our_work/data_gen/v1`
  - `data.layer_counts: [5, 6, 7, 8, 9, 10]`
  - `data.samples_per_bucket: 500000`
  - `data.thickness_range_nm: {min: 10, max: 500, step: 10}`
  - `tmm.num_points: 1024`
- `base_train.yaml`
  - `data.dataset_dir: outputs/our_work/data_gen/v1`
  - `data.vocab_path: outputs/our_work/data_gen/v1/vocab/vocab.json`
  - `training.output_dir: outputs/our_work/pretrain/base_train`
  - `training.max_steps: null`
  - `training.num_train_epochs: 5`
- `base_gpt.yaml`
  - `model.spectrum_dim: 2048`

关键约束：

- `model.spectrum_dim` 必须等于 `2 * tmm.num_points`
- 如果你把 `num_points` 改成不是 `1024`，就必须同步修改 `model.spectrum_dim`
- 现在 YAML 内的相对路径都会自动按仓库根目录解析

### 4. 从零开始部署与运行
以下步骤假设服务器部署目录为 `/srv/OptoGPT-GRPO`，并且真实数据输出保存在仓库根目录的 `outputs/` 下。

#### Step 1: 拉代码并进入仓库

```bash
cd /srv
git clone <your-repo-url> OptoGPT-GRPO
cd /srv/OptoGPT-GRPO
git checkout main
```

典型终端输出：

```text
Cloning into 'OptoGPT-GRPO'...
Already on 'main'
```

此时你应能看到：

- [our_work](/O:/Optics%20Code/OptoGPT-GRPO/our_work)
- [README.md](/O:/Optics%20Code/OptoGPT-GRPO/README.md)

#### Step 2: 同步材料库

把本地 `database/` 同步到服务器仓库根目录，例如：

```bash
scp -r database user@server:/srv/OptoGPT-GRPO/
```

同步完成后服务器上应存在：

- `/srv/OptoGPT-GRPO/database/*.csv`
- 或 `/srv/OptoGPT-GRPO/database/*.xlsx`

你可以执行：

```bash
ls /srv/OptoGPT-GRPO/database | head
```

典型终端输出：

```text
Ag.xlsx
Al.xlsx
Ge.xlsx
SiO2.xlsx
...
```

#### Step 3: 安装依赖

```bash
cd /srv/OptoGPT-GRPO
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas pyarrow openpyxl pyyaml pillow transformers safetensors
```

典型终端输出：

```text
Successfully installed ...
```

#### Step 4: 生成数据集

```bash
cd /srv/OptoGPT-GRPO
python our_work/data_gen/scripts/run_build_dataset.py --config our_work/data_gen/configs/dataset_v1.yaml
```

典型终端输出：

```text
data_gen buckets:  17%|█▋        | 1/6 [00:xx<00:xx, ... bucket/s, layer_count=5, generated=500000, valid=..., kept=...]
```

该步骤完成后应出现：

- `outputs/our_work/data_gen/v1/shards/shard-00000.parquet`
- `outputs/our_work/data_gen/v1/splits/split_manifest.json`
- `outputs/our_work/data_gen/v1/vocab/vocab.json`

你可以检查：

```bash
ls outputs/our_work/data_gen/v1/shards | head
cat outputs/our_work/data_gen/v1/splits/split_manifest.json
```

#### Step 5: 启动预训练

```bash
cd /srv/OptoGPT-GRPO
python our_work/pretrain/scripts/run_pretrain.py \
  --model-config our_work/pretrain/configs/model/base_gpt.yaml \
  --train-config our_work/pretrain/configs/train/base_train.yaml
```

典型终端输出：

```text
{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
{'eval_loss': ..., 'eval_token_accuracy': ..., 'eval_runtime': ..., 'epoch': ...}
100%|██████████| ...
```

该步骤完成后应出现：

- `outputs/our_work/pretrain/base_train/checkpoint-*`
- `outputs/our_work/pretrain/base_train/checkpoint-*/config.json`
- `outputs/our_work/pretrain/base_train/checkpoint-*/model.safetensors`
- `outputs/our_work/pretrain/base_train/checkpoint-*/vocab.json`

你可以检查：

```bash
ls outputs/our_work/pretrain/base_train
ls outputs/our_work/pretrain/base_train/checkpoint-1
```

#### Step 6: 运行独立评测

```bash
cd /srv/OptoGPT-GRPO
python our_work/pretrain/scripts/run_eval.py \
  --checkpoint-dir outputs/our_work/pretrain/base_train \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --database-dir database \
  --split val \
  --max-samples 256 \
  --max-new-tokens 10 \
  --output-dir outputs/our_work/eval \
  --output-json outputs/our_work/eval/latest_eval.json
```

典型终端输出：

```text
our_work eval:  42%|████▏     | 108/256 [00:xx<00:xx, ... sample/s, valid=..., exact=..., last_rmse=...]
{
  "summary": {
    "sample_count": 256,
    "valid_generation_count": ...,
    ...
  },
  "results": [...],
  "run_dir": ".../outputs/our_work/eval/base_train/eval_runs/2026..."
}
```

该步骤完成后应出现：

- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/summary.json`
- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/results.jsonl`
- `outputs/our_work/eval/latest_eval.json`

如果不加 `--disable-plots`，还会出现：

- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/plots/*.png`
- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/samples/*.png`

### 5. 只部署已有 checkpoint 做评测
如果你不想在服务器上重训，只想评测已有模型，需要同步：

- 仓库代码
- `database/`
- 已生成的数据集目录，例如 `outputs/our_work/data_gen/v1`
- 已训练的 checkpoint 目录，例如 `outputs/our_work/pretrain/base_train`

然后直接运行：

```bash
cd /srv/OptoGPT-GRPO
python our_work/pretrain/scripts/run_eval.py \
  --checkpoint-dir outputs/our_work/pretrain/base_train \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --database-dir database \
  --split val \
  --max-samples 256 \
  --max-new-tokens 10 \
  --output-dir outputs/our_work/eval \
  --output-json outputs/our_work/eval/latest_eval.json
```

### 6. 常见问题
- `database_path must point to an existing directory`
  - 原因：服务器没同步 `database/`
  - 检查：`ls database`
- `No checkpoint directory found under ...`
  - 原因：训练目录里没有 `checkpoint-*`
  - 检查：`ls outputs/our_work/pretrain/base_train`
- `spectrum_dim mismatch`
  - 原因：`base_gpt.yaml` 里的 `model.spectrum_dim` 与数据集 `2 * num_points` 不一致
- `num_points mismatch`
  - 原因：`run_eval.py` 命令行传入的 `--num-points` 与 checkpoint 的 `spectrum_dim` 不一致
- `read_excel` / parquet 相关报错
  - 原因：缺少 `openpyxl` 或 `pyarrow`

## 说明
- 当前 `physics/` 直接复用原 TMM 模块，不另起一套实现。
- 当前训练目标不是传统 teacher forcing CE，而是基于目标光谱 reward 的 GRPO。
- rollout 与 update 现在共用同一 policy 定义，不再出现 “filtered rollout / raw scoring” 的不一致。
- 当前默认的光谱误差是 `R/T` 直接误差，即比较拼接后的 `[R..., T...]` 光谱。
- `core/` 保留的主要目的，是兼容旧 OptoGPT checkpoint 的加载。
