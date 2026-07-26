#!/usr/bin/env bash
# prep.sh — data-pipeline prep: select images -> build fine-tune JSONL ->
# smoke test. Wraps select_images.py, prepare_finetune.py, and
# smoke_test.py into a single idempotent command.
#
# THE VALIDATION LADDER (CLAUDE.md "Toolchain v2"): never start a
# multi-hour training run without climbing this ladder in order:
#   1. THIS SCRIPT — select_images.py + prepare_finetune.py, then
#      smoke_test.py (~1 minute, no model weights loaded beyond the
#      processor/config) on the generated train.jsonl. Catches every
#      silent failure mode that previously cost multi-day training runs:
#      missing top-level "images" column, pixel_values absent (vision
#      encoder never running), collapsed/truncated sequences, image-token
#      count mismatches, and a completion mask that's empty or covers
#      image tokens.
#   2. A 32-dish overfit run (not automated by this script — confirms the
#      model can memorize a tiny set, i.e. gradients and masking are wired
#      correctly end-to-end through actual weight updates).
#   3. A short (~500-iteration) probe run, evaluated with infer.py against
#      the un-adapted base model as a baseline.
# Only after all three are green should ./train.sh run unattended for
# hours. Skipping straight to a long run is exactly how the v4 adapter
# died mid-run and went unnoticed for most of a day (see CLAUDE.md, Run 4).
#
# Steps this script runs:
#   1. select_images.py --max-images 1 [--with-depth] [--src DIR] [--dst DIR]
#   2. prepare_finetune.py --only-overhead [--depth-mode MODE] [--out-dir DIR]
#   3. smoke_test.py --data <out-dir>/train.jsonl [--model-dir DIR]  (unless --skip-smoke)
#
# Usage:
#   ./prep.sh [--depth-mode {none,text,image}] [--with-depth]
#             [--src DIR] [--dst DIR] [--out-dir DIR] [--model-dir DIR]
#             [--skip-smoke] [--help]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEPTH_MODE="none"
WITH_DEPTH=0
SRC=""
DST=""
OUT_DIR=""
MODEL_DIR="${MODEL_PATH:-$HOME/models/mlx-community/Qwen3.5-4B-MLX-4bit}"
SKIP_SMOKE=0

usage() {
  cat <<'EOF'
Usage: ./prep.sh [--depth-mode {none,text,image}] [--with-depth]
                 [--src DIR] [--dst DIR] [--out-dir DIR] [--model-dir DIR]
                 [--skip-smoke] [--help]

Runs the data half of the validation ladder in one command:
  1. select_images.py --max-images 1 (overhead-only, matches deployment)
  2. prepare_finetune.py --only-overhead
  3. smoke_test.py on the freshly generated train.jsonl

  --depth-mode {none,text,image}
                Forwarded to prepare_finetune.py. 'none' (default) is
                byte-identical to the v5 dataset. 'text'/'image' generate
                the depth-augmented variants (design brief sec. (f)) and
                REQUIRE depth_raw.png in dataset_clean/ — this script
                auto-enables --with-depth for you when --depth-mode is not
                'none', unless you already passed it explicitly.
  --with-depth  Forwarded to select_images.py: backfill depth_raw.png into
                dataset_clean/<dish>/ for dishes already selected.
  --src DIR     Forwarded to select_images.py --src (Nutrition5K root;
                default: ./Nutrition5K).
  --dst DIR     Forwarded to select_images.py --dst AND used as
                prepare_finetune.py --clean-dir (default: ./dataset_clean).
  --out-dir DIR Forwarded to prepare_finetune.py --out-dir. Default depends
                on --depth-mode: none -> finetune_data_v2, text ->
                finetune_data_v2d_txt, image -> finetune_data_v2d_img
                (mirrors prepare_finetune.py's own defaults).
  --model-dir DIR
                Model directory smoke_test.py loads a processor/config
                from (NOT the full training run — no weights beyond the
                processor are used). Default: $MODEL_PATH env var, or
                ~/models/mlx-community/Qwen3.5-4B-MLX-4bit.
  --skip-smoke  Skip the smoke_test.py step (selecting images + building
                JSONL only). Not recommended — this is rung 1 of the
                validation ladder for a reason.
  -h, --help    Show this help and exit.

Idempotent: select_images.py and prepare_finetune.py are safe to re-run
(they overwrite their own outputs deterministically); smoke_test.py has no
side effects.

Requires ml/.venv (run ./setup.sh first) and dataset_clean/'s source
imagery (run ./download_dataset.sh first).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --depth-mode)
      DEPTH_MODE="$2"
      shift 2
      ;;
    --depth-mode=*)
      DEPTH_MODE="${1#*=}"
      shift
      ;;
    --with-depth)
      WITH_DEPTH=1
      shift
      ;;
    --src)
      SRC="$2"
      shift 2
      ;;
    --src=*)
      SRC="${1#*=}"
      shift
      ;;
    --dst)
      DST="$2"
      shift 2
      ;;
    --dst=*)
      DST="${1#*=}"
      shift
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --out-dir=*)
      OUT_DIR="${1#*=}"
      shift
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --model-dir=*)
      MODEL_DIR="${1#*=}"
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "prep.sh: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$DEPTH_MODE" in
  none|text|image) ;;
  *)
    echo "prep.sh: --depth-mode must be one of none|text|image, got: $DEPTH_MODE" >&2
    exit 1
    ;;
esac

if [ "$DEPTH_MODE" != "none" ] && [ "$WITH_DEPTH" -eq 0 ]; then
  echo "note: --depth-mode $DEPTH_MODE needs depth_raw.png in dataset_clean/ — auto-enabling --with-depth"
  WITH_DEPTH=1
fi

# Resolve the effective prepare_finetune.py out-dir so we know what to feed
# smoke_test.py — must mirror prepare_finetune.py's own DEPTH_MODE_OUT_DIRS.
if [ -n "$OUT_DIR" ]; then
  EFFECTIVE_OUT_DIR="$OUT_DIR"
else
  case "$DEPTH_MODE" in
    none)  EFFECTIVE_OUT_DIR="finetune_data_v2" ;;
    text)  EFFECTIVE_OUT_DIR="finetune_data_v2d_txt" ;;
    image) EFFECTIVE_OUT_DIR="finetune_data_v2d_img" ;;
  esac
fi

PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found — run ./setup.sh first." >&2
  exit 1
fi

echo "=== Step 1/3: select_images.py ==="
select_args=(--max-images 1)
if [ -n "$SRC" ]; then
  select_args+=(--src "$SRC")
fi
if [ -n "$DST" ]; then
  select_args+=(--dst "$DST")
fi
if [ "$WITH_DEPTH" -eq 1 ]; then
  select_args+=(--with-depth)
fi
"$PY" select_images.py "${select_args[@]}"

echo
echo "=== Step 2/3: prepare_finetune.py ==="
prep_args=(--only-overhead --depth-mode "$DEPTH_MODE")
if [ -n "$DST" ]; then
  prep_args+=(--clean-dir "$DST")
fi
if [ -n "$OUT_DIR" ]; then
  prep_args+=(--out-dir "$OUT_DIR")
fi
"$PY" prepare_finetune.py "${prep_args[@]}"

if [ "$SKIP_SMOKE" -eq 1 ]; then
  echo
  echo "--skip-smoke passed — NOT running smoke_test.py. Data is ready in $EFFECTIVE_OUT_DIR/,"
  echo "but you have not cleared rung 1 of the validation ladder. Run smoke_test.py yourself"
  echo "before training."
  exit 0
fi

echo
echo "=== Step 3/3: smoke_test.py ==="
"$PY" smoke_test.py --data "$EFFECTIVE_OUT_DIR/train.jsonl" --model-dir "$MODEL_DIR"

echo
echo "Rung 1 of the validation ladder is green: $EFFECTIVE_OUT_DIR/ is ready for training."
echo "Next: a 32-dish overfit run, then a ~500-iteration probe evaluated against the base"
echo "model with infer.py, before starting a full ./train.sh run (see CLAUDE.md)."
