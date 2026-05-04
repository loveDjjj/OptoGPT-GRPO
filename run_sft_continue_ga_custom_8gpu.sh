#!/usr/bin/env bash
set -euo pipefail

# 8-GPU continue SFT on ga_custom_tasks from an existing checkpoint
# Run from repo root: OptoGPT-GRPO

INIT_CKPT="outputs/our_work/pretrain/a100_4gpu/checkpoint-131900"
DATASET_DIR="outputs/our_work/data_gen/ga_custom_tasks"
VOCAB_PATH="${DATASET_DIR}/vocab/vocab.json"

# Note: output_dir still comes from a100_8gpu.yaml.
# If needed, adjust training.output_dir in that yaml before running.

torchrun --nproc_per_node=8 our_work/pretrain/scripts/run_pretrain.py \
  --model-config our_work/pretrain/configs/model/base_gpt.yaml \
  --train-config our_work/pretrain/configs/train/a100_8gpu.yaml \
  --init-checkpoint-dir "${INIT_CKPT}" \
  --dataset-dir "${DATASET_DIR}" \
  --vocab-path "${VOCAB_PATH}" \
  --train-split train \
  --eval-split train
