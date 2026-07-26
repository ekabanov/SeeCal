#!/bin/bash
# SeeCal full training run (LoRA fine-tune) — mlx-vlm 0.6.7 stack (ml/.venv, no patches).
#
# BEFORE running, the validation ladder must be green (see CLAUDE.md Toolchain v2
# and ./prep.sh --help):
#   ./prep.sh                      # select_images -> prepare_finetune -> smoke_test
#   (then the 32-dish overfit + probe gates if the stack or data changed)
#
# Notes:
# - finetune_data_v2 has the top-level "images" field 0.6.7 requires.
# - LoRA scale = alpha/rank = 2 on this stack (standard convention);
#   LR 1e-4 validated by the overfit/probe gates on 2026-07-26.
# - completion masking is prefix-based; no --assistant-id needed.
# - NO --grad-checkpoint: +15% throughput (benchmarked 2026-07-26) but peaks
#   ~47GB Metal memory. Do NOT load other models (LM Studio!) or run evals
#   while training — Metal memory doesn't swap; co-tenancy = OOM-killed run.
#   If the machine must stay usable during a run, add --grad-checkpoint back
#   (drops to ~13GB for ~15% slower).
#
# Usage:
#   ./train.sh [--help]
# Configuration is via environment variables (no positional args):
#   MODEL_PATH    Base model directory (default: ~/models/Qwen3.5-4B-MLX-4bit,
#                 set up by ./download_model.sh).
#   OUTPUT_PATH   Adapter output directory (default: adapters_v5). PICK AN
#                 EXPLICIT, VERSIONED NAME (adapters_v6, adapters_v7, ...) for
#                 any new run — the default is only correct if you are
#                 deliberately re-running/resuming v5. This script warns (but
#                 does not refuse) if OUTPUT_PATH already has checkpoints in it.
#   DATASET       Training dataset directory (default: finetune_data_v2).
#   ADAPTER_PATH  Optional: resume from an existing adapter checkpoint (see
#                 the resume example below). Unset by default (fresh run).
#
# Resume-after-interruption example (reduce --epochs to just the remainder,
# and point ADAPTER_PATH + OUTPUT_PATH at the same versioned run so
# checkpoints keep accumulating in one place instead of forking a new dir):
#   ADAPTER_PATH=adapters_v6 OUTPUT_PATH=adapters_v6 ./train.sh
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage: ./train.sh [--help]

Runs the mlx-vlm LoRA fine-tune (2 epochs, LR 1e-4, rank 16/alpha 32) against
the configured base model and dataset. Takes no positional arguments —
configure via environment variables:

  MODEL_PATH    Base model directory.
                Default: ~/models/Qwen3.5-4B-MLX-4bit (./download_model.sh's
                output location).
  OUTPUT_PATH   Adapter output directory. Default: adapters_v5.
                IMPORTANT: pick an explicit versioned name (adapters_v6,
                adapters_v7, ...) for any run that isn't a deliberate
                resume/re-run of v5 — this script warns if the directory
                already has a checkpoint in it, but does not block you from
                overwriting one.
  DATASET       Training JSONL directory. Default: finetune_data_v2.
  ADAPTER_PATH  Resume from this existing adapter checkpoint instead of a
                fresh LoRA init. Unset by default.

Resume-after-interruption example:
  ADAPTER_PATH=adapters_v6 OUTPUT_PATH=adapters_v6 ./train.sh

Training log is tee'd to <OUTPUT_PATH>_train.log in this directory.

Expected timing (M3 Ultra, no --grad-checkpoint): ~3.5h for the full 2-epoch
run, peaking ~47-58GB Metal memory. Do NOT run anything else GPU-heavy
(LM Studio, evals, another training run) at the same time — Metal memory
does not swap, so co-tenancy silently OOM-kills the run instead of erroring
cleanly.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "train.sh: unknown argument: $arg (this script takes no positional args — see --help)" >&2
      exit 1
      ;;
  esac
done

# MODEL_PATH convention (set by ml/download_model.sh; override for a local copy).
# Maintainer's local override example (LM Studio's own cache of the same repo,
# used during v4/v5/v6 development on this machine — NOT a portable default):
#   MODEL_PATH=/Users/jevgenikabanov/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-4B-MLX-4bit}"
OUTPUT_PATH="${OUTPUT_PATH:-adapters_v5}"
DATASET="${DATASET:-finetune_data_v2}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

if [ -f "$OUTPUT_PATH/adapters.safetensors" ] && [ -z "$ADAPTER_PATH" ]; then
  echo "WARNING: $OUTPUT_PATH/adapters.safetensors already exists and ADAPTER_PATH is unset —" >&2
  echo "this run will overwrite it. If this isn't a deliberate resume/re-run, set an explicit," >&2
  echo "unused OUTPUT_PATH (e.g. OUTPUT_PATH=adapters_v7 ./train.sh)." >&2
fi

train_args=(
  --model-path "$MODEL_PATH"
  --dataset "$DATASET"
  --train-mode sft
  --train-on-completions
  --lora-rank 16
  --lora-alpha 32
  --batch-size 1
  --max-seq-length 2048
  --learning-rate 1e-4
  --epochs 2
  --steps-per-report 50
  --steps-per-save 500
  --output-path "$OUTPUT_PATH"
)
if [ -n "$ADAPTER_PATH" ]; then
  train_args+=(--adapter-path "$ADAPTER_PATH")
fi

.venv/bin/python -m mlx_vlm.lora "${train_args[@]}" 2>&1 | tee "${OUTPUT_PATH}_train.log"
