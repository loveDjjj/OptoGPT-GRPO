#!/usr/bin/env bash
set -euo pipefail

# 4-GPU eval via official our_work/eval entry
# Run from repo root: OptoGPT-GRPO

torchrun --nproc_per_node=4 our_work/eval/scripts/run_eval_suite.py \
  --config our_work/eval/configs/ga_custom_checkpoint980_4gpu.yaml
