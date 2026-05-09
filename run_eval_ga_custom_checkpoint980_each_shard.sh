#!/usr/bin/env bash
set -euo pipefail

# 4-GPU eval via official eval entry
# Run from repo root: OptoGPT-GRPO

torchrun --nproc_per_node=4 eval/scripts/run_eval_suite.py \
  --config eval/configs/ga_custom_checkpoint980_4gpu.yaml
