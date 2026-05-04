# 本次修改摘要

## 需求
- 修复 4 卡评测时 NCCL `Duplicate GPU detected` 报错。

## 实际修改
- `our_work/eval/pipeline.py`
  - 在分布式初始化前，新增 `torch.cuda.set_device(local_rank)`。
  - 确保每个 rank 在 NCCL collective 前绑定到唯一本地 GPU。

## 结果
- 避免多个 rank 被 NCCL 识别到同一 CUDA 设备导致初始化失败。

## Git
- branch: main
- commit: pending
