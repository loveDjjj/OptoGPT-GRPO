# 2-15um 光谱数据生成与预训练设计

## 目标
- 在仓库根目录下新增独立工作区 `our_work/`，用于承载新的数据集生成与预训练体系。
- 新体系不依赖根目录现有评测、强化学习、GRPO 代码路径，不要求与当前 checkpoint 或训练接口兼容。
- 目标任务是薄膜逆设计：输入 `2-15um`、`1024` 个波长点的 `R/T` 光谱，输出多层结构 token 序列。
- 预训练框架采用 `Hugging Face Transformers + Trainer`。
- 后续强化学习会单独重写，本设计只要求为未来的 `generation + teacher forcing scoring + HF checkpoint` 留出清晰接口。

## 已确认约束
- 光谱波段：`2-15um`
- 波长点数：`1024`
- 光谱输入形式：`R + T` 拼接，总输入维度 `2048`
- 结构层数：`5/6/7/8/9/10` 层
- 数据规模：每个层数 `50w` 条，总计 `300w` 条
- 边界条件：`Air | 可变多层薄膜 | Air`
- 材料来源：仓库内 `database/` 文件夹
- 相邻层允许使用相同材料
- 结构 token 形式：`材料_厚度`
- 厚度离散规则：`10-500nm`，步长 `10nm`
- 模型范式：`decoder-only`
- 训练框架：`Transformers + Trainer`

## 总体方案
新体系拆分为三个部分：

1. `our_work/_shared`
   - 迁移并维护与新体系直接相关的底层共用代码。
   - 主要包含物理求解、结构序列化、基础 I/O、配置读取与少量工具函数。
   - 不 import 根目录现有 RL/评测模块，避免耦合。

2. `our_work/data_gen`
   - 负责结构采样、光谱模拟、数据质量控制、数据分片与 manifest 生成。
   - 输出给 `our_work/pretrain` 直接消费的分片数据集与 tokenizer/vocab 元信息。

3. `our_work/pretrain`
   - 负责 tokenizer、HF 数据集封装、自定义 decoder-only 条件语言模型、Trainer 训练与评测。
   - 输出标准 HF 风格 checkpoint。

## 可迁移与参考边界
### 直接迁移到 `our_work/_shared`
- `physics/optical_calculator.py`
- `physics/TMM.py`
- `physics/structure.py`
- 必要时迁移 `physics/spectrum.py` 中的光谱辅助函数

### 只参考工程写法
- `utils/config.py`
- `utils/dist.py`
- `datasets/optogpt_dataset.py`
- `datasets/collator.py`

### 只参考思路，不继承接口
- `prework_optogpt/optogpt/core/datasets/datasets.py`
- `prework_optogpt/optogpt/core/trains/train.py`
- `prework_optogpt/optogpt/run_optogpt.py`

原因：
- 现有根目录代码与旧光谱维度、旧 checkpoint 兼容逻辑、旧 GRPO 训练方式强绑定。
- `prework_optogpt` 虽可作为 token 序列化与旧预训练行为参考，但不满足“标准 HF 生态接口”的目标。

## 目录设计
```text
our_work/
  _shared/
    physics/
    serialization/
    io/
    utils/
  data_gen/
    configs/
    pipeline/
    scripts/
    outputs/
  pretrain/
    configs/
    dataset/
    model/
    trainer/
    scripts/
```

## 数据生成设计
### 数据生成目标
- 对每个层数桶 `L in {5, 6, 7, 8, 9, 10}` 独立生成 `50w` 条样本。
- 总样本数为 `300w`。
- 每条样本独立生成，不使用子母结构派生逻辑。

### 结构采样规则
- 每层材料从 `database/` 对应的可用材料集合中有放回随机采样。
- 每层厚度从 `{10, 20, ..., 500}` nm 中随机采样。
- 单层 token 直接编码为 `Material_Thickness`。
- 对完全重复的结构做去重；若去重后数量不足，继续补采，直到每个层数桶达到目标数。

### 光谱模拟规则
- 使用迁移后的 `our_work/_shared/physics/optical_calculator.py` 和 `TMM.py` 批量计算光谱。
- 固定波长范围为 `2-15um`，波长点数为 `1024`。
- 输出 `R` 与 `T` 两段曲线，并在数据落盘时拼接成 `2048` 维 `float32` 向量。

### 质量控制
- 结构合法性检查：
  - token 可解析为材料与厚度
  - 层数位于 `5-10`
  - 厚度位于 `10-500nm`
- 光谱数值检查：
  - `R/T` 必须有限
  - `R/T` 必须位于合理容差范围内
  - `R + T` 不得明显超过 `1 + tolerance`
- 不合格样本直接丢弃并补采。

### 存储格式
不再使用单个超大 `.npy + object array` 作为主格式，改为分片数据集。

每条样本至少包含：
- `sample_id`
- `layer_count`
- `structure_tokens`
- `token_ids`
- `spectrum_rt`
- `materials`
- `thickness_nm`

输出目录建议：
```text
our_work/data_gen/outputs/v1/
  manifests/
  shards/
    shard-00000.parquet
    shard-00001.parquet
    ...
  vocab/
    vocab.json
    tokenizer_config.json
  stats/
    layer_count_distribution.json
    token_frequency.json
    spectrum_stats.json
  splits/
    split_manifest.json
```

推荐 `Parquet/Arrow shards` 的原因：
- 与 Hugging Face `datasets` 对接顺畅
- 支持按列读取
- 便于断点续跑和版本管理
- 对 `300w x 2048` 的数值数据更容易做分片管理

### 数据切分
- 每个层数桶分别随机切分，再合并成全局 split。
- 原始数据分片保持中性命名，split 信息由 `splits/split_manifest.json` 单独维护。
- 推荐比例：
  - `train: 98%`
  - `val: 1%`
  - `test: 1%`
- 这样能保证所有层数桶在 train/val/test 中都有覆盖。

### 运行节奏
全量生成前分三阶段验证：
- 阶段 1：`1k` 样本做 pipeline smoke test
- 阶段 2：`10w` 样本验证数据 schema、吞吐与存储格式
- 阶段 3：确认无误后生成全量 `300w`

## 预训练设计
### 建模目标
- 输入：`2048` 维 `R+T` 连续光谱
- 输出：结构 token 序列
- 框架：`decoder-only causal LM`

### 条件注入方式
不将光谱离散为 token，而是采用连续条件前缀：

1. 将 `2048` 维光谱送入 `spectrum projector`
2. projector 输出固定长度 `K` 的前缀 embedding
3. 将这些 prefix embeddings 拼接到结构 token embeddings 前面
4. 用标准 causal LM 方式预测结构 token 序列

训练序列形态：
```text
[prefix_1, ..., prefix_K, BOS, token_1, ..., token_n, EOS]
```

loss 规则：
- prefix 位置的 label 全部为 `-100`
- 可选地将 `BOS` 位置 label 也设为 `-100`
- 从第一个真实结构 token 到 `EOS` 参与 causal LM loss

### 为什么采用该方案
- 保留光谱连续信息，不引入额外量化误差
- 保持 decoder-only 结构，符合已确认方向
- 与 HF 生态兼容，便于后续 SFT / PEFT / RL
- 比“把 2048 维光谱强行变成离散 token 序列”更自然

### tokenizer 设计
- 词表独立维护，不复用根目录旧 checkpoint 词表
- token 形式：`Material_Thickness`
- special tokens：
  - `[PAD]`
  - `[BOS]`
  - `[EOS]`
  - `[UNK]`
- 最大结构长度：
  - `BOS + 10 层 token + EOS = 12`

### projector 设计
- 第一版使用简单稳定的 `MLP(2048 -> K * hidden_size)`，再 reshape 为 `(K, hidden_size)`
- `K` 设计为可配置，默认建议从 `8` 开始
- 第一版不引入额外的 CNN / Perceiver / Resampler 结构，优先保证可训、可复现、易调试

## `pretrain` 模块边界
```text
our_work/pretrain/
  configs/
    model/
    data/
    train/
  dataset/
    hf_dataset.py
    collator.py
    tokenizer.py
  model/
    configuration_spectral_gpt.py
    modeling_spectral_gpt.py
    projector.py
    generation.py
  trainer/
    trainer.py
    metrics.py
  scripts/
    run_pretrain.py
    run_eval.py
```

职责划分：
- `dataset/tokenizer.py`
  - 管理 `材料_厚度` 词表
  - 提供 encode/decode
- `dataset/hf_dataset.py`
  - 从 `data_gen` 分片读取样本
- `dataset/collator.py`
  - 组装 `spectra + input_ids + attention_mask + labels`
- `model/projector.py`
  - 连续光谱到 prefix embeddings 的投影
- `model/modeling_spectral_gpt.py`
  - 自定义 HF `PreTrainedModel`
- `model/generation.py`
  - 推理与采样封装
- `scripts/run_pretrain.py`
  - 唯一训练入口

## 模型实现建议
建议实现一个自定义 HF 模型族，而不是强行魔改纯文本 GPT：

- `SpectralGPTConfig`
- `SpectralGPTForCausalLM`

模型 `forward()` 至少支持：
- `spectra`
- `input_ids`
- `attention_mask`
- `labels`
- `past_key_values`

原因：
- 条件输入不是纯 token ids
- 训练和生成都需要显式处理 `spectra`
- 需要把 projector 与 tokenizer 一起纳入标准 checkpoint
- 未来 RL rollout 也需要统一的条件注入接口

## 训练样本格式
collator 输出至少包含：
- `spectra: (batch, 2048)`
- `input_ids: (batch, seq_len)`
- `attention_mask: (batch, prefix_len + seq_len)`
- `labels: (batch, prefix_len + seq_len)`

其中：
- prefix 区域 label 为 `-100`
- 真实结构 token 与 `EOS` 参与损失

## checkpoint 规范
使用 Hugging Face 风格保存，不沿用根目录当前 checkpoint 格式：

```text
checkpoint_dir/
  config.json
  generation_config.json
  tokenizer.json / vocab.json
  special_tokens_map.json
  model.safetensors
  trainer_state.json
```

这样便于：
- 继续预训练
- SFT
- LoRA / PEFT
- 后续新 GRPO / policy optimization

## 与未来 RL 的接口约束
虽然本阶段不实现新的 GRPO，但预训练模型必须保证未来可提供三类能力：

1. conditioned generation
   - 输入 `spectra`
   - 采样输出结构 token 序列

2. teacher-forcing scoring
   - 输入 `spectra + token_ids`
   - 返回每个 token 的 logprob

3. HF checkpoint compatibility
   - policy / reference / finetuned model 使用同一套 checkpoint 规范

只要预训练阶段把这三类接口做干净，后续 RL 可以单独重建，而无需重新定义基础模型。

## 错误处理与可观测性
- 数据生成阶段记录：
  - 已生成样本数
  - 丢弃样本数
  - TMM 失败数
  - 非物理解样本数
  - 各层数桶完成进度
- 预训练阶段记录：
  - train/val loss
  - token-level perplexity
  - 按层数桶切分的验证 loss
  - 生成样本的长度分布与非法 token 比例

## 非目标
- 不要求与根目录现有 GRPO checkpoint 兼容
- 不要求复用当前根目录训练入口
- 不在本阶段实现新的 GRPO
- 不在第一版中实现复杂多任务头、连续厚度回归或 surrogate simulator

## 风险与缓解
### 风险 1：数据量大，生成成本高
- 缓解：采用分片、断点续跑、先 `1k` 和 `10w` 验证，再跑全量

### 风险 2：材料空间过大导致词表与分布复杂
- 缓解：由 `database/` 扫描结果生成材料 registry，并输出频率统计；必要时后续可增加白名单配置

### 风险 3：decoder-only 条件建模不稳定
- 缓解：第一版采用简单 MLP projector + 小规模样本验证；prefix 长度 `K` 配置化

### 风险 4：数据 schema 返工
- 缓解：在全量生成前，先固定 shard schema、manifest、tokenizer 元信息，再启动正式生成

## 最终结论
- 新体系采用 `our_work/` 独立工作区
- `data_gen` 负责 `5-10` 层、每层 `50w` 的独立结构采样与 TMM 光谱生成
- `pretrain` 负责基于 `Transformers + Trainer` 的 decoder-only 连续条件建模
- 光谱通过 projector 变为 prefix embeddings，结构序列通过 causal LM 学习
- 全部 checkpoint、tokenizer、config 采用标准 HF 风格保存
- 后续新的强化学习系统围绕这套模型接口单独实现
