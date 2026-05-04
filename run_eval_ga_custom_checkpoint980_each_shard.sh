#!/usr/bin/env bash
set -euo pipefail

# Evaluate checkpoint on each ga_custom_tasks shard separately with per-shard plots.
# Run from repo root: OptoGPT-GRPO

CHECKPOINT_DIR="outputs/our_work/pretrain/a100_4gpu/checkpoint-980"
SHARDS_DIR="outputs/our_work/data_gen/ga_custom_tasks/shards"
DATABASE_DIR="our_work/_shared/database"
OUTPUT_ROOT="outputs/our_work/eval/ga_custom_tasks_checkpoint980_each_shard"

python our_work/pretrain/scripts/eval_each_shard.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --shards-dir "${SHARDS_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-new-tokens 10 \
  --wavelength-min 2.0 \
  --wavelength-max 15.0 \
  --incident-angle 0.0 \
  --polarization 0 \
  --tolerance 1e-3 \
  --complex-dtype complex128
