#!/bin/bash
# SeeCal v5 full training run — mlx-vlm 0.6.7 stack (ml/.venv, no patches).
#
# BEFORE running, the validation ladder must be green (see CLAUDE.md Toolchain v2):
#   .venv/bin/python smoke_test.py --data finetune_data_v2/train.jsonl
#   (then the 32-dish overfit + probe gates if the stack or data changed)
#
# Notes:
# - finetune_data_v2 has the top-level "images" field 0.6.7 requires.
# - LoRA scale = alpha/rank = 2 on this stack (standard convention);
#   LR 1e-4 validated by the overfit/probe gates on 2026-07-26.
# - completion masking is prefix-based; no --assistant-id needed.
# - Resume after interruption: add --adapter-path adapters_v5 and reduce --epochs.
# - NO --grad-checkpoint: +15% throughput (benchmarked 2026-07-26) but peaks
#   ~47GB Metal memory. Do NOT load other models (LM Studio!) or run evals
#   while training — Metal memory doesn't swap; co-tenancy = OOM-killed run.
#   If the machine must stay usable during a run, add --grad-checkpoint back
#   (drops to ~13GB for ~15% slower).
set -euo pipefail
cd "$(dirname "$0")"

.venv/bin/python -m mlx_vlm.lora \
  --model-path /Users/jevgenikabanov/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit \
  --dataset finetune_data_v2 \
  --train-mode sft \
  --train-on-completions \
  --lora-rank 16 \
  --lora-alpha 32 \
  --batch-size 1 \
  --max-seq-length 2048 \
  --learning-rate 1e-4 \
  --epochs 2 \
  --steps-per-report 50 \
  --steps-per-save 500 \
  --output-path adapters_v5 \
  2>&1 | tee adapters_v5_train.log
