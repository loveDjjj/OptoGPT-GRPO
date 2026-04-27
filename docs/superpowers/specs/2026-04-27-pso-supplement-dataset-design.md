# PSO 补充数据集设计

## 目标
在 `our_work/pso` 下构建一个基于粒子群算法的补充数据集生成器。它负责搜索吸收谱接近指定目标形状的多层膜结构，并把满足条件的结构写成与现有 `our_work/data_gen` parquet 数据格式兼容的训练样本。

## 范围
- PSO 补充数据生成流程独立于现有随机数据生成流程。
- 物理问题定义必须与主数据集一致：
  - 边界条件：`Air | stack | Air`
  - 波段范围：`2.0-15.0 um`
  - 波长点数：`1024`
  - 保存光谱：`[R..., T...]`，长度 `2048`
  - 层数范围：`5, 6, 7, 8, 9, 10`
  - 厚度取值：`10-500 nm`，步长 `10 nm`
  - 结构 token 格式：`Material_ThicknessNm`
- 吸收谱只作为 PSO 优化目标：
  - `A = 1 - R - T`
  - `loss = mse(A, target_A)`
- 接受样本写盘时保存真实仿真的 `R/T`，不能用目标吸收谱替代真实光谱。

## 目标光谱
所有目标都在同一个 `2.0-15.0 um` 波长网格上生成。

固定带状目标：
- `broad_3_13`：`3.0-13.0 um` 吸收率为 `1`，其余为 `0`。
- `band_5_8`：`5.0-8.0 um` 吸收率为 `1`，其余为 `0`。
- `dual_3_5_8_13`：`3.0-5.0 um` 和 `8.0-13.0 um` 吸收率为 `1`，其余为 `0`。
- `notch_3_5`：`3.0-5.0 um` 吸收率为 `0`，其余为 `1`。

洛伦兹窄带目标：
- 中心波长：`2.1, 2.2, ..., 14.9 um`
- 半高宽：`0.02 um`
- 峰值归一化到 `1`
- 目标 id 格式：`lorentz_fwhm_0p02_center_<center>`

目标总数：`133` 个。

## 推荐模块结构
```text
our_work/pso/
  configs/
    pso_supplement.yaml
  targets.py
  search.py
  dataset_writer.py
  scripts/
    run_pso_dataset.py
```

现有 `PSO_lisan.py` 可以作为原型参考，但补充数据集生成器不建议继续堆在这个大脚本里，应拆成更小的模块。

## 数据流程
1. 读取配置，并从 `database/` 构建材料注册表。
2. 使用主数据一致的材料名和厚度取值构建 token vocab。
3. 生成全部目标吸收谱。
4. 对每个目标和每个层数分别运行 PSO：
   - 用指定 seed 初始化粒子群。
   - 把粒子转成结构，批量运行 TMM，计算吸收谱，并与目标吸收谱计算 MSE。
   - 保留所有 `MSE < acceptance_mse_threshold` 的结构。
   - 使用 `tuple(structure_tokens)` 做全局去重。
   - 持续搜索，直到该目标需要的结构数量达到要求。
5. 如果 PSO 连续 `max_stagnant_iterations` 轮没有产生新的合格结构，就切换随机种子重新开始。
6. 当目标/层数桶达到样本数量，或达到 `max_restarts` 后仍无法继续产生新样本，就停止该桶。
7. 将接受的样本写成 parquet shards，并写出 manifest 和统计信息。

## 接受样本格式
样本必须包含现有训练链路需要的字段：
- `sample_id`
- `layer_count`
- `structure_tokens`
- `token_ids`
- `materials`
- `thickness_nm`
- `spectrum_rt`

补充数据特有的元数据字段：
- `target_id`
- `target_family`
- `target_center_um`
- `target_fwhm_um`
- `target_mse`
- `acceptance_mse_threshold`
- `pso_seed`
- `pso_restart_index`

这些新增列不应影响现有预训练读取逻辑，因为预训练 loader 只消费已有字段。

## 输出目录
```text
outputs/our_work/data_gen/pso_supplement/
  shards/
    shard-00000.parquet
    shard-00001.parquet
  splits/
    split_manifest.json
  vocab/
    vocab.json
  targets/
    target_manifest.json
  stats/
    summary.json
```

补充数据集不直接写入随机数据集目录。训练时的数据混合后续通过 merge 脚本或多数据集 loader 处理。

## 停止与去重策略
- 按完整有序 token 序列去重：`tuple(structure_tokens)`。
- 统计每个目标和全局的重复数量。
- 统计 stagnant iterations 和 stagnant restarts。
- 如果某个目标/层数桶持续无法产生新的合格样本，就停止该桶，并在 `stats/summary.json` 中记录缺口。
- 不能通过复制重复样本来补齐数量。

## 兼容性说明
- 当前 `PSO_lisan_config.yaml` 使用固定 Au 底层和较小材料集合。补充数据流程必须移除固定底层，并从主 `database/` 读取材料集合。
- 当前脚本依赖从 `our_work/pso` 目录启动，因为它使用了 `from TMM ...` 和相对路径。新入口应按 `our_work/data_gen/scripts/run_build_dataset.py` 的模式，从仓库根目录解析路径。
- 当前脚本会保存吸收谱图，但补充数据流程应优先生成 parquet 数据；图和统计摘要是辅助产物。

## 验证计划
- 单元测试目标光谱生成：
  - 固定带状目标 mask
  - 洛伦兹中心数量和半高宽行为
- 单元测试粒子转 token：
  - 材料索引裁剪
  - 厚度离散到 `10 nm`
  - 层数保持正确
- 单元测试接受样本序列化，确保符合现有 data_gen schema。
- 小规模 smoke 测试完整流程：
  - `2` 个目标
  - `1` 个层数
  - 小 population 和 iteration
  - 低样本数量要求
  - 输出 parquet 能被现有 pretrain dataset loader 读取。
