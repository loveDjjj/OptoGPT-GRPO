# our_work data_gen GPU 分块采样与分块 TMM 设计

## 目标
- 保持 `our_work/data_gen` 的输出 schema 不变，但把正式数据生成流程改成可支撑 `5-10` 层、每 bucket `50w` 条样本的实现。
- 结构生成改为 GPU 批量采样，不再逐条在 CPU 上随机生成 token 序列。
- TMM 计算改为按 chunk 分批执行，不再把整 bucket 一次性塞进显存或主存。
- bucket 内结构继续保持全局严格唯一；重复结构要自动丢弃并补采，直到达到目标条数。

## 当前问题
- 当前 `sample_unique_bucket(...)` 在 CPU 上一次性生成整个 bucket 的结构序列。
- 当前 `build_small_dataset(...)` 会把 `samples_per_bucket` 条结构整批传给 `simulate_structure_batch(...)`。
- 这导致 `50w / bucket` 时：
  - CPU 采样成为前置瓶颈
  - 唯一性集合、结构 token 列表、TMM 输入配置会在内存里堆成超大对象
  - TMM 侧 `batch_size = len(structure_configs)`，会尝试把整个 bucket 一次性送进 GPU

## 选定方案
采用“GPU 分块采样 + GPU 分块 TMM + bucket 内全局去重补采”。

### 原则
- GPU 负责高频、规则化、可张量化的随机采样。
- CPU 仅保留最小必要的全局去重状态和最终落盘记录。
- TMM 每次只吃一个显式可配的 chunk。
- bucket 内全局唯一以最终保留结构为准，而不是单个 chunk 内局部唯一。

## 配置设计
在 `our_work/data_gen/configs/dataset_v1.yaml` 中新增两个配置段。

### `sampling`
- `device`
  - 值：`auto | cpu | cuda | cuda:0`
  - 默认：`auto`
- `batch_size`
  - 含义：单次 GPU 采样的候选结构数量
  - 默认建议：`65536`
- `max_duplicate_retry`
  - 含义：若连续多轮补采后去重增量极低，达到该上限后直接报错，避免死循环
  - 默认建议：`1000`

### `tmm`
- 保留现有：
  - `wavelength_range_um`
  - `num_points`
  - `incident_angle`
  - `polarization`
  - `tolerance`
  - `complex_dtype`
- 新增：
  - `batch_size`
    - 含义：单次送入 TMM 的结构数量
    - 默认建议：`2048`

## 模块设计

### 1. `our_work/data_gen/pipeline/sampler.py`
新增 GPU 采样入口，保留现有 CPU 接口用于测试和回退。

#### 新职责
- 把材料和厚度候选编码为整数 id
- 在 GPU 上按 `(batch_size, layer_count)` 形状采样材料 id 和厚度 id
- 合成结构 token id 表示
- 返回：
  - `material_ids`
  - `thickness_ids`
  - `structure_tokens`
  - 用于去重的 `structure_key`

#### 去重策略
- chunk 内先在 GPU/张量层面生成候选
- 转到 CPU 后用 `set[tuple[str, ...]]` 做 bucket 内全局唯一过滤
- 只保留首次出现的结构
- 若本轮新增不足，就继续按 `sampling.batch_size` 补采

### 2. `our_work/data_gen/pipeline/build_dataset.py`
把当前“一次性生成整个 bucket”的流程改成“双层循环”。

#### 外层：bucket 级循环
- 遍历 `layer_counts`
- 为每个 bucket 维护：
  - `target_count = samples_per_bucket`
  - `seen_structures`
  - `accepted_records`
  - `generated_candidate_count`
  - `accepted_valid_count`

#### 内层：采样 / TMM chunk 循环
- 每次先按 `sampling.batch_size` 在 GPU 上生成一批候选结构
- 对候选做 bucket 内全局去重
- 把去重后的结构按 `tmm.batch_size` 再切块
- 每个 TMM chunk 调用一次 `simulate_structure_batch(...)`
- 只把 `ok_mask=True` 的样本落入最终记录
- 直到当前 bucket 的有效保留条数达到 `samples_per_bucket`

#### 进度条
- 保留现有 bucket 级 `tqdm`
- 新增 chunk 级 postfix 信息：
  - `bucket_kept`
  - `bucket_target`
  - `sample_batch`
  - `tmm_batch`
  - `duplicates_skipped`
  - `valid_kept`
- 不新增第二根常驻进度条，避免终端刷屏；统一在 bucket 级进度条上更新 postfix

### 3. `our_work/data_gen/pipeline/simulator.py`
接口保持单次“传入一个结构组列表，返回一个结构组结果列表”的语义不变。

#### 调整点
- 不再假设上游会把整个 bucket 一次传进来
- 保持 `simulate_structure_batch(...)` 的单批职责，只负责：
  - token 转 TMM config
  - 调用批量 TMM
  - 做物理合法性检查

### 4. `our_work/data_gen/scripts/run_build_dataset.py`
扩展 YAML 读取和参数传递。

#### 调整点
- 读取 `sampling.device`
- 读取 `sampling.batch_size`
- 读取 `sampling.max_duplicate_retry`
- 读取 `tmm.batch_size`
- 把这些配置透传给 `build_small_dataset(...)`

## 数据流
单个 bucket 的正式执行顺序：

1. 从材料 registry 构造材料 id 映射
2. 从厚度区间构造厚度值表与厚度 id 映射
3. GPU 批量采样 `(sampling.batch_size, layer_count)` 候选结构
4. CPU 侧做 bucket 内全局唯一过滤
5. 唯一候选按 `tmm.batch_size` 分块
6. 每个 TMM chunk 批量计算光谱
7. 对非法光谱样本丢弃
8. 合法样本转 record 并累计
9. 达到 `samples_per_bucket` 后进入下一个 bucket
10. 所有 bucket 完成后再做 split / shard / vocab 落盘

## 错误处理
- `sampling.batch_size <= 0`：直接报错
- `tmm.batch_size <= 0`：直接报错
- `tmm.batch_size > sampling.batch_size`：允许，但等效为本轮唯一候选数上限
- `sampling.device=cuda*` 但无 CUDA：回退到 CPU 并打印显式告警
- 连续补采达到 `max_duplicate_retry` 且仍未凑满 bucket：抛出带 bucket 信息的异常
- 某个 TMM chunk 全部非法：继续补采，不中断整体流程

## 测试设计

### 单元测试
- `tests/our_work/data_gen/test_sampler.py`
  - GPU/CPU 统一采样接口的形状与 token 合法性
  - chunk 内候选可转 token
  - 全局去重补采逻辑
- `tests/our_work/data_gen/test_build_dataset.py`
  - 小样本下按 `sampling.batch_size` + `tmm.batch_size` 分块仍能生成完整 manifest
  - `samples_per_bucket` 大于 `tmm.batch_size` 时不再一次性送完整 bucket
  - 进度开关不影响输出
- `tests/our_work/data_gen/test_simulator.py`
  - 分块输入下返回结构数量与 mask 一致

### 运行态 smoke
- 用极小配置验证：
  - `sampling.batch_size=8`
  - `tmm.batch_size=2`
  - `samples_per_bucket=4`
- 从非仓库根目录启动 `run_build_dataset.py`
- 确认：
  - 输出目录结构正常
  - shard / manifest / vocab 正常
  - 终端可见 bucket 进度与 chunk postfix

## 兼容性
- 输出 schema 不变：
  - `sample_id`
  - `layer_count`
  - `structure_tokens`
  - `token_ids`
  - `materials`
  - `thickness_nm`
  - `spectrum_rt`
- 现有 `pretrain` 数据加载与 tokenizer 逻辑无需改动
- 保留现有 CPU 采样函数用于测试和回退，不强制删除旧接口

## 不做的事
- 不在这次改动里改动根目录 GRPO / eval 路径
- 不把去重集合做成磁盘外排或布隆过滤器
- 不引入多进程/多卡数据生成编排
- 不在这次改动里重写 TMM 内核

## 建议默认值
- `sampling.device: auto`
- `sampling.batch_size: 65536`
- `sampling.max_duplicate_retry: 1000`
- `tmm.batch_size: 2048`

## 自检
- 方案只覆盖 `our_work/data_gen` 一个子系统，没有把 `pretrain` 一起卷进来，范围可控。
- 全局唯一、GPU 采样、分块 TMM、进度条、YAML 配置四个核心要求都已覆盖。
- 没有留 `TODO/TBD/待确认` 占位。
