#!/bin/bash
# SeeCal full training run (LoRA fine-tune) — mlx-vlm 0.6.7 stack (ml/.venv, no patches).
#
# BEFORE running, the validation ladder must be green (see AGENTS.md Toolchain v2
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
#   MODEL_PATH    Base model directory (default: ~/models/mlx-community/Qwen3.5-4B-MLX-4bit,
#                 set up by ./download_model.sh).
#   OUTPUT_PATH   Adapter output directory (default: adapters_v5). PICK AN
#                 EXPLICIT, VERSIONED NAME (adapters_v6, adapters_v7, ...) for
#                 any new run — the default is only correct if you are
#                 deliberately re-running/resuming v5. This script warns (but
#                 does not refuse) if OUTPUT_PATH already has checkpoints in it.
#   DATASET       Training dataset directory (default: finetune_data_v2).
#   ADAPTER_PATH  Unsupported. Resume is disabled because mlx-vlm 0.6.7 can
#                 silently expose the wrong trainable parameter set.
#   EXPECTED_TRAINABLE_MILLIONS
#                 Fail-fast LoRA guard. Default: 32.464896.
#   MAX_SEQ_LENGTH Token ceiling (default: 2048).
#   EPOCHS         Number of training epochs (default: 2).
#
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage: ./train.sh [--help]

Runs the mlx-vlm LoRA fine-tune (2 epochs, LR 1e-4, rank 16/alpha 32) against
the configured base model and dataset. Takes no positional arguments —
configure via environment variables:

  MODEL_PATH    Base model directory.
                Default: ~/models/mlx-community/Qwen3.5-4B-MLX-4bit (./download_model.sh's
                output location).
  OUTPUT_PATH   Adapter output directory. Default: adapters_v5.
                IMPORTANT: pick an explicit versioned name (adapters_v6,
                adapters_v7, ...) for any run that isn't a deliberate
                resume/re-run of v5 — this script warns if the directory
                already has a checkpoint in it, but does not block you from
                overwriting one.
  DATASET       Training JSONL directory. Default: finetune_data_v2.
  ADAPTER_PATH  Unsupported: adapter resume is disabled until mlx-vlm resume
                semantics are characterized.
  EXPECTED_TRAINABLE_MILLIONS
                Expected LoRA trainable count. Default: 32.464896.
  MAX_SEQ_LENGTH
                Token ceiling. Default: 2048.
  EPOCHS        Number of epochs. Default: 2.

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
#   MODEL_PATH=~/models/mlx-community/Qwen3.5-4B-MLX-4bit
MODEL_PATH="${MODEL_PATH:-$HOME/models/mlx-community/Qwen3.5-4B-MLX-4bit}"
OUTPUT_PATH="${OUTPUT_PATH:-adapters_v5}"
DATASET="${DATASET:-finetune_data_v2}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EPOCHS="${EPOCHS:-2}"
EXPECTED_TRAINABLE_MILLIONS="${EXPECTED_TRAINABLE_MILLIONS:-32.464896}"

if [ -n "$ADAPTER_PATH" ]; then
  echo "train.sh: ADAPTER_PATH resume is disabled: mlx-vlm 0.6.7 exposed 366.9M" >&2
  echo "trainable parameters in a factored-track resume instead of the expected 32.5M." >&2
  echo "Start a fresh run with an unused OUTPUT_PATH." >&2
  exit 2
fi

if [ -f "$OUTPUT_PATH/adapters.safetensors" ]; then
  echo "WARNING: $OUTPUT_PATH/adapters.safetensors already exists and ADAPTER_PATH is unset —" >&2
  echo "this run will overwrite it. If this isn't a deliberate resume/re-run, set an explicit," >&2
  echo "unused OUTPUT_PATH (e.g. OUTPUT_PATH=adapters_v7 ./train.sh)." >&2
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

train_args=(
  --model-path "$MODEL_PATH"
  --dataset "$DATASET"
  --train-mode sft
  --train-on-completions
  --lora-rank 16
  --lora-alpha 32
  --batch-size 1
  --max-seq-length "$MAX_SEQ_LENGTH"
  --learning-rate 1e-4
  --epochs "$EPOCHS"
  --steps-per-report 50
  --steps-per-save 500
  --output-path "$OUTPUT_PATH"
)
.venv/bin/python verified_lora_train.py \
  --expected-millions "${EXPECTED_TRAINABLE_MILLIONS:-32.464896}" \
  --log-file "${OUTPUT_PATH}_train.log" \
  -- \
  .venv/bin/python -m mlx_vlm.lora "${train_args[@]}"
