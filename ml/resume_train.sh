#!/bin/bash
# Guarded LoRA warm-start for mlx-vlm 0.6.7.
#
# This is not an exact optimizer/dataloader resume: mlx-vlm checkpoints contain
# adapter weights and config, but not Adam state or RNG/shuffle position.
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage:
  ADAPTER_PATH=<saved-adapter-dir> \
  OUTPUT_PATH=<new-output-dir> \
  ITERS=<continuation-updates> \
  ./resume_train.sh

Optional environment variables:
  MODEL_PATH    Base Qwen model directory.
  DATASET       Training dataset directory.
  LEARNING_RATE Continuation learning rate (default: 1e-4).
  MAX_SEQ_LENGTH
  EXPECTED_TRAINABLE_MILLIONS

The source adapter is never overwritten. Training is blocked unless the model
reports exactly the expected LoRA trainable-parameter count.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "resume_train.sh: unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

MODEL_PATH="${MODEL_PATH:-$HOME/models/mlx-community/Qwen3.5-4B-MLX-4bit}"
DATASET="${DATASET:-finetune_data_v2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EXPECTED_TRAINABLE_MILLIONS="${EXPECTED_TRAINABLE_MILLIONS:-32.464896}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
ITERS="${ITERS:-}"

if [ -z "$ADAPTER_PATH" ] || [ -z "$OUTPUT_PATH" ] || [ -z "$ITERS" ]; then
  usage >&2
  exit 2
fi
if [ ! -f "$ADAPTER_PATH/adapters.safetensors" ] ||
   [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
  echo "resume_train.sh: incomplete source adapter: $ADAPTER_PATH" >&2
  exit 2
fi
if [ "$ADAPTER_PATH" = "$OUTPUT_PATH" ]; then
  echo "resume_train.sh: OUTPUT_PATH must differ from ADAPTER_PATH" >&2
  exit 2
fi
if [ -e "$OUTPUT_PATH/adapters.safetensors" ]; then
  echo "resume_train.sh: refusing to overwrite existing output: $OUTPUT_PATH" >&2
  exit 2
fi
case "$ITERS" in
  ''|*[!0-9]*|0)
    echo "resume_train.sh: ITERS must be a positive integer" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUTPUT_PATH")"

train_args=(
  --model-path "$MODEL_PATH"
  --adapter-path "$ADAPTER_PATH"
  --output-path "$OUTPUT_PATH"
  --dataset "$DATASET"
  --train-mode sft
  --train-on-completions
  --batch-size 1
  --iters "$ITERS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --learning-rate "$LEARNING_RATE"
  --steps-per-report 50
  --steps-per-save 500
)

.venv/bin/python verified_lora_train.py \
  --expected-millions "$EXPECTED_TRAINABLE_MILLIONS" \
  --log-file "${OUTPUT_PATH}_train.log" \
  -- \
  .venv/bin/python safe_lora_warm_start.py "${train_args[@]}"
