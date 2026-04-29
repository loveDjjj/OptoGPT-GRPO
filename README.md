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
- `tqdm`
  `our_work/pso` 的补充数据集搜索会用它显示 target/layer 级别的进度条；未安装时会退化为普通输出。
- `tensorboard`
  `our_work/pretrain` 的实时损失、学习率、梯度范数等可视化默认通过 TensorBoard 查看。
  `our_work/data_gen` 自动分析和评测绘图都会用到。

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
- `tqdm`

推荐安装示例：

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas pyarrow openpyxl pyyaml pillow transformers safetensors tqdm
```

安装完成后可执行：

```bash
python -c "import torch,yaml,pandas,scipy,numpy,PIL,transformers,tqdm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

典型终端输出：

```text
torch 2.x.x cuda True
```

### 3. 默认配置与关键约束
默认服务器配置文件：

- 数据生成：[dataset_v1.yaml](/O:/Optics%20Code/OptoGPT-GRPO/our_work/data_gen/configs/dataset_v1.yaml)
- 数据生成（4 卡）：`our_work/data_gen/configs/a100_4gpu.yaml`
- 数据生成（8 卡）：`our_work/data_gen/configs/a100_8gpu.yaml`
- 训练：[base_train.yaml](/O:/Optics%20Code/OptoGPT-GRPO/our_work/pretrain/configs/train/base_train.yaml)
- 训练（4 卡）：`our_work/pretrain/configs/train/a100_4gpu.yaml`
- 训练（8 卡）：`our_work/pretrain/configs/train/a100_8gpu.yaml`
- 模型：[base_gpt.yaml](/O:/Optics%20Code/OptoGPT-GRPO/our_work/pretrain/configs/model/base_gpt.yaml)
- 强化学习（基础）：`our_work/rl/configs/grpo/base_grpo.yaml`
- 强化学习（4 卡）：`our_work/rl/configs/grpo/a100_4gpu.yaml`
- 强化学习（8 卡）：`our_work/rl/configs/grpo/a100_8gpu.yaml`
- PSO 补充数据集：`our_work/pso/configs/pso_supplement.yaml`
- GA 优秀解族补充数据集：`our_work/ga/configs/ga_seeded_absorbers.yaml`

当前默认值（单卡 A100 80G + 16 CPU）：

- `dataset_v1.yaml`
  - `paths.database_dir: database`
  - `paths.output_dir: outputs/our_work/data_gen/v1`
  - `data.layer_counts: [5, 6, 7, 8, 9, 10]`
  - `data.samples_per_bucket: 500000`
  - `data.thickness_range_nm: {min: 10, max: 500, step: 10}`
  - `sampling.device: auto`
  - `sampling.batch_size: 65536`
  - `sampling.max_duplicate_retry: 1000`
  - `tmm.device: auto`
  - `tmm.cpu_threads: 16`
  - `tmm.batch_size: 4096`
  - `tmm.num_points: 1024`
  - `analysis.enabled: true`
  - `analysis.auto_after_build: true`
  - `analysis.scopes: [all]`
  - `analysis.spectrum.pca_components: 8`
  - `analysis.spectrum.cluster_count: 16`
- `base_train.yaml`
  - `data.dataset_dir: outputs/our_work/data_gen/v1`
  - `data.vocab_path: outputs/our_work/data_gen/v1/vocab/vocab.json`
  - `data.num_workers: 8`
  - `data.prefetch_factor: 4`
  - `data.pin_memory: true`
  - `data.persistent_workers: true`
  - `training.output_dir: outputs/our_work/pretrain/base_train`
  - `training.per_device_train_batch_size: 16`
  - `training.per_device_eval_batch_size: 64`
  - `training.gradient_accumulation_steps: 2`
  - `training.max_steps: null`
  - `training.num_train_epochs: 5`
  - `training.learning_rate: 1e-4`
  - `training.bf16: true`
  - `training.tf32: true`
  - `training.logging_steps: 1000`
  - `training.eval_steps: 100000`
  - `training.save_steps: 50000`
  - `monitoring.tensorboard/jsonl/csv/save_plots: true`
  - 评估路径会先把 logits 预处理成 `argmax token ids` 再做 metrics，避免完整收集 `[batch, seq_len, vocab]` 级别的大张量
  - `distributed.*` 只有在 `torchrun --nproc_per_node=...` 的真实多卡环境下才会生效；单进程 `python run_pretrain.py ...` 会忽略这部分并清理脏的 DDP 环境变量
- `base_gpt.yaml`
  - `model.spectrum_dim: 2048`
  - `model.prefix_length: 8`
  - `model.n_embd: 1024`
  - `model.n_layer: 6`
  - `model.n_head: 16`
- `base_grpo.yaml`
  - `training.per_device_batch_size: 16`
  - `rollout.group_size: 4`
  - `rollout.batch_size: 512`
  - `scoring.batch_size: 1024`
  - `reward.tmm.batch_size: 4096`
  - `monitoring.tensorboard/jsonl/csv/save_plots: true`
- `pso_supplement.yaml`
  - `paths.database_dir: our_work/_shared/database`
  - `paths.output_dir: outputs/our_work/data_gen/pso_supplement`
  - `data.layer_counts: [5, 6, 7, 8, 9, 10]`
  - `data.thickness_range_nm: {min: 10, max: 500, step: 10}`
  - `targets.include_fixed: true`
  - `targets.include_lorentzian: true`
  - `targets.lorentzian.center_min_um: 2.1`
  - `targets.lorentzian.center_max_um: 14.9`
  - `targets.lorentzian.center_step_um: 0.1`
  - `targets.lorentzian.fwhm_um: 0.02`
  - `search.population_size: 8192`
  - `search.iterations: 50`
  - `search.batch_size: 2048`
  - `search.max_accepted_per_target_layer: 1000`
  - `search.acceptance_mse_threshold: 0.01`
  - `tmm.wavelength_range_um: [2.0, 15.0]`
  - `tmm.num_points: 1024`
  - `tmm.batch_size: 2048`
- `ga_seeded_absorbers.yaml`
  - `paths.database_dir: our_work/_shared/database`
  - `paths.output_dir: outputs/our_work/data_gen/ga_seeded_absorbers`
  - `data.target_sample_count: 100`
  - `data.thickness_range_nm: {min: 10, max: 500, step: 10}`
  - `data.include_seed_thickness_values: true`
  - `targets.include_ids: null`
  - `search.population_size: 4096`
  - `search.generations: 80`
  - `search.batch_size: 1024`
  - `search.acceptance_mse_threshold: 0.005`
  - `search.max_restarts: 20`
  - `tmm.wavelength_range_um: [2.0, 15.0]`
  - `tmm.num_points: 1024`
  - `visualization.enabled: true`

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

运行前，建议先确认数据生成配置里确实启用了分块采样和分块 TMM：

```bash
cd /srv/OptoGPT-GRPO
python -c "import yaml, pathlib; cfg=yaml.safe_load(pathlib.Path('our_work/data_gen/configs/dataset_v1.yaml').read_text(encoding='utf-8')); print('sampling =', cfg['sampling']); print('tmm.batch_size =', cfg['tmm']['batch_size'])"
```

典型终端输出：

```text
sampling = {'device': 'auto', 'batch_size': 65536, 'max_duplicate_retry': 1000}
tmm.batch_size = 4096
```

默认配置片段如下：

```yaml
data:
  layer_counts: [5, 6, 7, 8, 9, 10]
  samples_per_bucket: 500000
  thickness_range_nm:
    min: 10
    max: 500
    step: 10

sampling:
  device: auto
  batch_size: 65536
  max_duplicate_retry: 1000

tmm:
  wavelength_range_um: [2.0, 15.0]
  num_points: 1024
  incident_angle: 0.0
  polarization: 0
  tolerance: 0.001
  complex_dtype: complex128
  batch_size: 4096

analysis:
  enabled: true
  auto_after_build: true
  batch_size: 8192
  scopes: [all]
  structure:
    enabled: true
  spectrum:
    enabled: true
    device: auto
    engine: rapids
    pca_components: 8
    pca_fit_samples: 50000
    cluster_count: 16
    cluster_fit_samples: 50000
    cluster_iterations: 20
    scatter_max_points: 20000
    save_split_analysis: false
```

说明：
- 默认自动分析只跑 `all`，避免生成完成后再把 `train / val / test` 重复扫一遍。
- 当 `analysis.spectrum.engine: rapids` 时，`run_build_dataset.py` 会在独立子进程里调用 `run_analyze_dataset.py`，避免同一 Python 进程里混用 `torch` 和 RAPIDS 的 CUDA 运行时栈。

```bash
cd /srv/OptoGPT-GRPO
python our_work/data_gen/scripts/run_build_dataset.py --config our_work/data_gen/configs/dataset_v1.yaml
```

典型终端输出：

```text
data_gen buckets:  17%|█▋        | 1/6 [00:xx<00:xx, ... bucket/s, layer_count=5, bucket_kept=98304, bucket_target=500000, sample_batch=65536, tmm_batch=4096, duplicates_skipped=..., valid_kept=...]
```

该步骤完成后应出现：

- `outputs/our_work/data_gen/v1/shards/shard-00000.parquet`
- `outputs/our_work/data_gen/v1/splits/split_manifest.json`
- `outputs/our_work/data_gen/v1/vocab/vocab.json`
- `outputs/our_work/data_gen/v1/analysis/all/structure_material_by_layer.png`
- `outputs/our_work/data_gen/v1/analysis/all/structure_thickness_by_layer.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_mean_std.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_pca_scatter.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_cluster_sizes.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_cluster_representatives.png`

说明：

- 结构候选现在按 `sampling.batch_size` 在 GPU/CPU 上分块生成。
- TMM 光谱计算按 `tmm.batch_size` 分批执行，不会再把整 bucket 一次性送进显存/内存。
- bucket 内仍然保持全局严格唯一；重复结构会被丢弃并自动补采。
- 数据生成结束后默认只自动跑 `all` 分析；`train/val/test` 建议通过独立 CLI 按需补跑。
- 当前默认只自动分析 `all`，避免对 `train/val/test` 重复扫描导致耗时过长。
- 光谱分析使用拼接后的 `[R..., T...]` 做标准化、PCA 和聚类；结构分析会把材料和厚度拆开统计。
- 光谱分析优先走 RAPIDS（`cudf + cuml`），把 PCA / 聚类主路径放在 GPU 上。

你可以检查：

```bash
ls outputs/our_work/data_gen/v1/shards | head
cat outputs/our_work/data_gen/v1/splits/split_manifest.json
ls outputs/our_work/data_gen/v1/analysis/all
```

如果你切到多卡，直接使用专用配置：

4 卡 A100 正式命令：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 our_work/data_gen/scripts/run_build_dataset.py --config our_work/data_gen/configs/a100_4gpu.yaml
```

8 卡 A100 正式命令：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=8 our_work/data_gen/scripts/run_build_dataset.py --config our_work/data_gen/configs/a100_8gpu.yaml
```

说明：

- 当前多卡数据生成先按 `layer bucket` 在 rank 之间分配，保证 bucket 内全局唯一不会被跨 rank 破坏。
- `4` 卡时 6 个 bucket 会分到 4 个 rank。
- `8` 卡时会有空闲 rank，这是当前版本为了保证唯一性和正确性做的保守实现。

#### Step 4.1: 单独运行数据集分析

如果你已经有现成数据集，也可以不重新生成，直接单独跑分析：

```bash
cd /srv/OptoGPT-GRPO
python our_work/data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --split all \
  --wavelength-min 2.0 \
  --wavelength-max 15.0 \
  --device auto
```

典型终端输出：

```text
# 命令本身默认安静执行，完成后会在 analysis 目录下写出 PNG / JSON 结果
```

如果只分析某些 shard 文件：

```bash
cd /srv/OptoGPT-GRPO
python our_work/data_gen/scripts/run_analyze_dataset.py \
  --shard-path outputs/our_work/data_gen/v1/shards/shard-00000.parquet \
  --shard-path outputs/our_work/data_gen/v1/shards/shard-00001.parquet \
  --output-dir outputs/our_work/data_gen/custom_analysis \
  --wavelength-min 2.0 \
  --wavelength-max 15.0 \
  --device cpu
```

如果你还想单独分析 `train / val / test`，直接改 `--split` 即可，例如：

```bash
cd /srv/OptoGPT-GRPO
python our_work/data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --split train \
  --wavelength-min 2.0 \
  --wavelength-max 15.0 \
  --device auto
```

该步骤完成后应出现：

- `outputs/our_work/data_gen/v1/analysis/analysis_manifest.json`
- `outputs/our_work/data_gen/v1/analysis/<scope>/structure_analysis.json`
- `outputs/our_work/data_gen/v1/analysis/<scope>/spectrum_analysis.json`
- 对应 scope 下的结构分布和谱形分析 PNG

#### Step 4.2: 转换并分析旧 `.npy` 数据集

旧数据集文件：

- `dataset/Spectrum_train.npy`
- `dataset/Spectrum_test.npy`
- `dataset/Structure_train.npy`
- `dataset/Structure_test.npy`

不能直接传给 `our_work/data_gen/scripts/run_analyze_dataset.py`，需要先转换成 `our_work/data_gen` 的 parquet schema。

转换命令：

```bash
cd /srv/OptoGPT-GRPO
python -m our_work.data_gen.scripts.convert_legacy_npy_dataset \
  --spectrum-train dataset/Spectrum_train.npy \
  --structure-train dataset/Structure_train.npy \
  --spectrum-test dataset/Spectrum_test.npy \
  --structure-test dataset/Structure_test.npy \
  --output-dir outputs/legacy_npy_parquet \
  --records-per-shard 50000 \
  --num-workers 8
```

该步骤完成后应出现：

- `outputs/legacy_npy_parquet/shards/train-shard-00000.parquet`
- `outputs/legacy_npy_parquet/shards/test-shard-00000.parquet`
- `outputs/legacy_npy_parquet/splits/split_manifest.json`
- `outputs/legacy_npy_parquet/vocab/vocab.json`
- `outputs/legacy_npy_parquet/stats/summary.json`

说明：

- `Spectrum_*.npy` 会按行复制到 `spectrum_rt` 字段。
- `Structure_*.npy` 会从 `Material_ThicknessNm` token 拆出 `materials` 和 `thickness_nm`。
- 旧数据集的光谱维度通常是 `142 = R(71) + T(71)`，对应旧配置 `0.4-1.1 um`、`71` 个波长点。
- `Structure_train.npy` 是 object array，NumPy 不能内存映射；转换脚本会一次只加载一个 split，但转换 train 时仍需要服务器有足够内存容纳该 object 数组。
- `--num-workers` 默认为 `1`。当设置为大于 `1` 时，脚本会按 shard 多进程并行写 parquet；并行前会先扫描结构 token 构建稳定 vocab。
- 多进程模式下，每个 worker 进程都会加载当前 split 的 `Structure_*.npy` object array，因此主机内存占用会近似随 `num_workers` 放大；如果内存紧张，先从 `4` 或更小值开始。

如果只想先小规模验证转换流程，可以加采样上限：

```bash
cd /srv/OptoGPT-GRPO
python -m our_work.data_gen.scripts.convert_legacy_npy_dataset \
  --spectrum-train dataset/Spectrum_train.npy \
  --structure-train dataset/Structure_train.npy \
  --spectrum-test dataset/Spectrum_test.npy \
  --structure-test dataset/Structure_test.npy \
  --output-dir outputs/legacy_npy_parquet_smoke \
  --records-per-shard 50000 \
  --max-train-samples 10000 \
  --max-test-samples 10000 \
  --num-workers 2
```

转换后运行分析：

```bash
cd /srv/OptoGPT-GRPO
python our_work/data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/legacy_npy_parquet \
  --scope train \
  --scope test \
  --output-dir outputs/legacy_npy_analysis \
  --wavelength-min 0.4 \
  --wavelength-max 1.1 \
  --engine rapids \
  --device auto
```

如果服务器没有 RAPIDS / cudf / cuml，只分析结构分布：

```bash
cd /srv/OptoGPT-GRPO
python our_work/data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/legacy_npy_parquet \
  --scope train \
  --scope test \
  --output-dir outputs/legacy_npy_analysis_structure_only \
  --wavelength-min 0.4 \
  --wavelength-max 1.1 \
  --disable-spectrum-analysis
```

#### Step 4.3: 生成 PSO 补充数据集

PSO 补充数据集用于围绕指定目标吸收谱搜索相近结构，作为随机生成数据集之外的定向补充数据。默认目标包括：

- `broad_3_13`：`3-13 um` 吸收为 1，其余为 0。
- `band_5_8`：`5-8 um` 吸收为 1，其余为 0。
- `dual_3_5_8_13`：`3-5 um` 和 `8-13 um` 吸收为 1，其余为 0。
- `notch_3_5`：`3-5 um` 吸收为 0，其余为 1。
- 洛伦兹窄带目标：`2.1-14.9 um`，中心步长 `0.1 um`，半高宽 `0.02 um`。

单进程运行：

```bash
cd /srv/OptoGPT-GRPO
python -m our_work.pso.scripts.run_pso_dataset --config our_work/pso/configs/pso_supplement.yaml
```

该步骤完成后应出现：

- `outputs/our_work/data_gen/pso_supplement/shards/shard-00000.parquet`
- `outputs/our_work/data_gen/pso_supplement/splits/split_manifest.json`
- `outputs/our_work/data_gen/pso_supplement/vocab/vocab.json`
- `outputs/our_work/data_gen/pso_supplement/targets/target_manifest.json`
- `outputs/our_work/data_gen/pso_supplement/stats/summary.json`
- `outputs/our_work/data_gen/pso_supplement/stats/search_summary.json`

说明：

- 安装 `tqdm` 后，运行过程中会显示 `pso rank <rank>/<world_size>` 的 target/layer 级别进度条。
- PSO 结构参数与主数据生成链路保持一致：`5-10` 层、`10-500 nm`、厚度步长 `10 nm`、材料来自 `our_work/_shared/database/`。
- 输出光谱仍然是 `[R..., T...]`，共 `2048` 维；目标吸收谱只用于 PSO 搜索时计算 MSE。
- 只有 `absorption MSE < search.acceptance_mse_threshold` 的结构会被写入数据集。
- 写出前会按完整 `structure_tokens` 做全局去重。
- 该补充数据默认写入独立目录，不会自动混入随机数据集；后续训练混合比例需要在训练数据加载侧单独定义。

多进程拆分运行：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 -m our_work.pso.scripts.run_pso_dataset --config our_work/pso/configs/pso_supplement.yaml
```

使用多进程前，需要先把 `our_work/pso/configs/pso_supplement.yaml` 里的 `distributed.enabled` 改成 `true`。多进程会按 `target/layer` work items 拆分任务，并分别写到：

- `outputs/our_work/data_gen/pso_supplement/rank00`
- `outputs/our_work/data_gen/pso_supplement/rank01`
- `outputs/our_work/data_gen/pso_supplement/rankXX`

当前版本还没有内置跨 rank 合并与二次去重脚本；正式混入训练前，建议先对各 `rankXX` 目录做合并和全局去重。

#### Step 4.4: 分析 PSO 补充数据集

PSO 数据集生成完成后，可以单独运行分析和可视化：

```bash
cd /srv/OptoGPT-GRPO
python -m our_work.pso.analysis.run_analyze_pso \
  --dataset-dir outputs/our_work/data_gen/pso_supplement \
  --output-dir outputs/our_work/pso_analysis/pso_supplement \
  --split all \
  --wavelength-min-um 2.0 \
  --wavelength-max-um 15.0 \
  --top-k 8 \
  --max-spectrum-groups 100
```

如果要给所有 `target/layer` 组合都画光谱图，把 `--max-spectrum-groups` 改成 `-1`：

```bash
cd /srv/OptoGPT-GRPO
python -m our_work.pso.analysis.run_analyze_pso \
  --dataset-dir outputs/our_work/data_gen/pso_supplement \
  --output-dir outputs/our_work/pso_analysis/pso_supplement_full \
  --split all \
  --max-spectrum-groups -1
```

该步骤完成后应出现：

- `outputs/our_work/pso_analysis/pso_supplement/summary.json`
- `outputs/our_work/pso_analysis/pso_supplement/analysis_manifest.json`
- `outputs/our_work/pso_analysis/pso_supplement/tables/target_layer_stats.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/search_efficiency.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/material_stats.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/diversity_stats.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/best_samples.csv`
- `outputs/our_work/pso_analysis/pso_supplement/figures/mse_by_target.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/accepted_count_heatmap.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/structures/material_frequency.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/spectra/<target_id>/layer_<layer_count>_topk.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/spectra/<target_id>/layer_<layer_count>_mean_band.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/lorentzian/center_vs_best_mse.png`

#### Step 4.5: 生成 GA 优秀解族补充数据集

GA 补充数据集用于从已知优秀结构出发做局部变异和交叉，搜索满足阈值的相近优秀解族。当前只包含三类目标：

- `broad_3_13_high`：`3-13 um` 高吸收，其他波段不参与 loss。种子结构：`YbF3(870) / ZnS(480) / Si(280) / Bi(20) / Ge(130) / Bi(820) / Au(100)`。
- `mid_5_8_high`：`3-5 um` 低吸收、`5-8 um` 高吸收、`8-13 um` 低吸收，其他波段不参与 loss。种子结构：`Si(250) / SiO2(120) / Ge(500) / MgF2(850) / Ge(110) / MgF2(500) / Bi(130) / Au(100)`。
- `dual_3_5_8_13_high`：`3-5 um` 高吸收、`5-8 um` 低吸收、`8-13 um` 高吸收，其他波段不参与 loss。种子结构：`SiO2(150) / MgF2(500) / Si(500) / ZnS(450) / Ge(490) / MgF2(280) / Si(320) / Bi(250) / Au(100)`。

单进程运行：

```bash
cd /srv/OptoGPT-GRPO
python -m our_work.ga.scripts.run_ga_dataset --config our_work/ga/configs/ga_seeded_absorbers.yaml
```

该步骤完成后应出现：

- `outputs/our_work/data_gen/ga_seeded_absorbers/shards/shard-00000.parquet`
- `outputs/our_work/data_gen/ga_seeded_absorbers/splits/split_manifest.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/vocab/vocab.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/targets/target_manifest.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/stats/summary.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/stats/search_summary.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/*_accepted_absorption_topk.png`
- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/*_mse_hist.png`

说明：

- GA 固定使用每个种子结构的层数，在同层数内做材料变异、厚度变异、精英保留、锦标赛选择和 layer-wise crossover。
- 接受条件是 masked absorption MSE `< 0.005`，每个目标默认收集 `100` 条全局去重结构。
- 默认材料集合与 PSO 一致，使用 `database_dir` 下的全部材料；如果只想围绕已知优秀解的材料局部搜索，可在 YAML 里显式写 `materials`。
- 已知优秀解包含 `820/850/870 nm` 层；默认配置会把这些 seed 厚度额外加入可选厚度集合。若要严格限制到 `10-500 nm`，将 `data.include_seed_thickness_values` 改为 `false`，但种子会被近邻厚度裁剪，搜索质量可能下降。
- 输出光谱仍然是 `[R..., T...]`，共 `2048` 维；目标吸收谱只用于 GA 搜索时计算 masked MSE。

多进程拆分运行：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=3 -m our_work.ga.scripts.run_ga_dataset --config our_work/ga/configs/ga_seeded_absorbers.yaml
```

使用多进程前，需要先把 `our_work/ga/configs/ga_seeded_absorbers.yaml` 里的 `distributed.enabled` 改成 `true`。三个 target 会按 rank 拆分，并分别写到：

- `outputs/our_work/data_gen/ga_seeded_absorbers/rank00`
- `outputs/our_work/data_gen/ga_seeded_absorbers/rank01`
- `outputs/our_work/data_gen/ga_seeded_absorbers/rank02`

如果只想从某个 parquet shard 里随机抽样画图，例如从 `shard-00000.parquet` 随机抽 10 条 `3-13 um` 目标光谱：

```bash
cd /srv/OptoGPT-GRPO
python -m our_work.ga.scripts.plot_random_parquet_spectra \
  --shard-path outputs/our_work/data_gen/ga_seeded_absorbers/shards/shard-00000.parquet \
  --output-path outputs/our_work/data_gen/ga_seeded_absorbers/figures/random_10_broad_3_13_absorption.png \
  --sample-count 10 \
  --seed 42 \
  --target-id broad_3_13_high
```

该命令会同时写出：

- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/random_10_broad_3_13_absorption.png`
- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/random_10_broad_3_13_absorption.selected.json`

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
- `outputs/our_work/pretrain/base_train/tensorboard/`
- `outputs/our_work/pretrain/base_train/metrics/train_metrics.jsonl`
- `outputs/our_work/pretrain/base_train/metrics/eval_metrics.jsonl`
- `outputs/our_work/pretrain/base_train/metrics/train_metrics.csv`
- `outputs/our_work/pretrain/base_train/metrics/eval_metrics.csv`
- `outputs/our_work/pretrain/base_train/plots/train_loss.png`
- `outputs/our_work/pretrain/base_train/plots/learning_rate.png`
- `outputs/our_work/pretrain/base_train/plots/grad_norm.png`
- `outputs/our_work/pretrain/base_train/plots/eval_loss.png`
- `outputs/our_work/pretrain/base_train/plots/eval_token_accuracy.png`
- `outputs/our_work/pretrain/base_train/plots/overview.png`

你可以检查：

```bash
ls outputs/our_work/pretrain/base_train
ls outputs/our_work/pretrain/base_train/checkpoint-1
```

TensorBoard 实时查看命令：

```bash
cd /srv/OptoGPT-GRPO
tensorboard --logdir outputs/our_work/pretrain/base_train/tensorboard --bind_all
```

4 卡 A100 正式命令：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 our_work/pretrain/scripts/run_pretrain.py \
  --model-config our_work/pretrain/configs/model/base_gpt.yaml \
  --train-config our_work/pretrain/configs/train/a100_4gpu.yaml
```

4 卡默认训练配置要点：

- 读取数据集：`outputs/our_work/data_gen/a100_4gpu`
- `per_device_train_batch_size: 512`
- `per_device_eval_batch_size: 512`
- `num_train_epochs: 100`
- `learning_rate: 1e-4`
- `lr_scheduler_type: cosine`
- `warmup_ratio: 0.01`
- `max_grad_norm: 1.0`
- `logging_steps: 1000`
- `eval_steps: 5000`
- `save_steps: 5000`
- `save_total_limit: 3`

8 卡 A100 正式命令：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=8 our_work/pretrain/scripts/run_pretrain.py \
  --model-config our_work/pretrain/configs/model/base_gpt.yaml \
  --train-config our_work/pretrain/configs/train/a100_8gpu.yaml
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

#### Step 7: 运行 our_work 轻量 GRPO

`our_work/rl` 当前是一个轻量、训练就绪的 GRPO 子系统，接口风格尽量贴近 `Transformers + torchrun`，但没有引入重型外部 RL 平台。

单机单卡 smoke：

```bash
cd /srv/OptoGPT-GRPO
python our_work/rl/scripts/run_grpo.py --config our_work/rl/configs/grpo/base_grpo.yaml
```

基础 RL 配置要点：

- `model.checkpoint_dir: outputs/our_work/pretrain/base_train`
- `data.dataset_dir: outputs/our_work/data_gen/v1`
- `per_device_batch_size: 16`
- `rollout.batch_size: 512`
- `scoring.batch_size: 1024`
- `reward.tmm.batch_size: 4096`
- `training.lr_scheduler_type: cosine`
- `training.warmup_ratio: 0.01`

4 卡 A100 正式命令：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 our_work/rl/scripts/run_grpo.py --config our_work/rl/configs/grpo/a100_4gpu.yaml
```

4 卡 RL 配置要点：

- `model.checkpoint_dir: outputs/our_work/pretrain/a100_4gpu`
- `data.dataset_dir: outputs/our_work/data_gen/a100_4gpu`
- `data.num_workers: 0`
- `per_device_batch_size: 32`
- `gradient_accumulation_steps: 1`
- `rollout.batch_size: 128`
- `scoring.batch_size: 256`
- `reward.tmm.batch_size: 128`
- `training.lr_scheduler_type: cosine`
- `training.warmup_ratio: 0.01`
- `training.eval_steps: 1000`
- `training.save_steps: 1000`

8 卡 A100 正式命令：

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=8 our_work/rl/scripts/run_grpo.py --config our_work/rl/configs/grpo/a100_8gpu.yaml
```

8 卡 RL 配置要点：

- `model.checkpoint_dir: outputs/our_work/pretrain/a100_8gpu`
- `data.dataset_dir: outputs/our_work/data_gen/a100_8gpu`
- `data.num_workers: 0`
- `per_device_batch_size: 32`
- `gradient_accumulation_steps: 1`
- `rollout.batch_size: 128`
- `scoring.batch_size: 256`
- `reward.tmm.batch_size: 128`
- `training.lr_scheduler_type: cosine`
- `training.warmup_ratio: 0.01`
- `training.eval_steps: 1000`
- `training.save_steps: 1000`

典型终端输出：

```text
our_work grpo: 100%|██████████| ... [loss=..., reward=..., valid=...]
```

该步骤完成后应出现：

- `outputs/our_work/rl/<run-name>/metrics/train_metrics.jsonl`
- `outputs/our_work/rl/<run-name>/metrics/eval_metrics.jsonl`
- `outputs/our_work/rl/<run-name>/metrics/train_metrics.csv`
- `outputs/our_work/rl/<run-name>/metrics/eval_metrics.csv`
- `outputs/our_work/rl/<run-name>/plots/train_loss.png`
- `outputs/our_work/rl/<run-name>/plots/train_mean_reward.png`
- `outputs/our_work/rl/<run-name>/plots/train_valid_ratio.png`
- `outputs/our_work/rl/<run-name>/plots/eval_mean_reward.png`
- `outputs/our_work/rl/<run-name>/plots/overview.png`
- `outputs/our_work/rl/<run-name>/tensorboard/`
- `outputs/our_work/rl/<run-name>/checkpoints/checkpoint-*`

如需实时查看 RL 标量，可执行：

```bash
tensorboard --logdir outputs/our_work/rl/<run-name>/tensorboard --bind_all
```

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
## our_work Eval Suite

用途：

- 加载 `our_work/pretrain` 训练好的 checkpoint
- 同时评估 `train + val`
- 各 split 随机抽样固定数量样本
- 批量生成预测结构
- 批量回算预测结构光谱
- 计算目标光谱与预测光谱误差
- 输出汇总 JSON / JSONL
- 输出最好 / 最差 / 接近均值误差样本的序列与光谱对比图

运行方式：

```bash
python our_work/eval/scripts/run_eval_suite.py --config our_work/eval/configs/base_eval.yaml
```

默认配置文件：

- `our_work/eval/configs/base_eval.yaml`

主要配置项：

- `paths.checkpoint_dir`
- `paths.dataset_dir`
- `paths.database_dir`
- `paths.output_dir`
- `data.splits`
- `data.sample_mode`
- `data.max_samples_per_split`
- `data.max_shards_per_split`
- `inference.batch_size`
- `inference.max_new_tokens`
- `tmm.batch_size`
- `plots.worst_count`
- `plots.best_count`
- `plots.mean_count`

采样模式：

- `random`
  - 扫描整个 split 的所有 shard，用 reservoir sampling 做严格随机抽样
- `head_shards`
  - 只扫描前若干个 shard，速度最快，但样本可能有顺序偏差
- `shard_subset_random`
  - 先随机选若干个 shard，再只扫描这些 shard，速度和代表性折中

输出内容：

- `summary.json`
- `split_summaries.json`
- `selected_samples.json`
- `results/train.jsonl`
- `results/val.jsonl`
- `plots/train/*.png`
- `plots/val/*.png`
- `plots/comparison/*.png`
- `samples/train/best/*.png`
- `samples/train/worst/*.png`
- `samples/train/mean/*.png`
- `samples/val/best/*.png`
- `samples/val/worst/*.png`
- `samples/val/mean/*.png`
