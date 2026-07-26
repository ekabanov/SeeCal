#!/usr/bin/env bash
# eval.sh — wraps infer.py --test-set for adapter evaluation.
#
# --limit has NO default and is REQUIRED: infer.py's own --limit defaults to
# 20, which once silently truncated what was meant to be a full overnight
# eval down to 20 samples with no warning (nobody noticed until the numbers
# looked suspiciously clean). This wrapper refuses to run until you say how
# many samples you actually want, so that mistake can't repeat silently.
#
# Usage:
#   ./eval.sh ADAPTER_PATH --limit N [--test-set FILE] [--model-path DIR]
#             [--out-json FILE] [--max-tokens N] [--help]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MODEL_PATH="~/models/mlx-community/Qwen3.5-4B-MLX-4bit"
DEFAULT_TEST_SET="finetune_data_v2/test.jsonl"

usage() {
  cat <<EOF
Usage: ./eval.sh ADAPTER_PATH --limit N [--test-set FILE] [--model-path DIR]
                 [--out-json FILE] [--max-tokens N] [--help]

Evaluates a LoRA adapter against a JSONL test set via infer.py, computing
MAE/median error for calories, protein, fat, and carbs.

  ADAPTER_PATH   REQUIRED, first positional argument. Path to the adapter
                 directory (e.g. adapters_v5).
  --limit N      REQUIRED. Number of test-set samples to evaluate. There is
                 NO default — infer.py's own default (20) once silently
                 truncated a full eval overnight; this wrapper will not
                 guess for you. Pass the full test-set size (e.g. 325) for
                 a complete run.
  --test-set FILE
                 Test-set JSONL. Default: $DEFAULT_TEST_SET
  --model-path DIR
                 Base model directory.
                 Default: $DEFAULT_MODEL_PATH
  --out-json FILE
                 Where to write the eval summary JSON. Default: auto-named
                 under runs/, as runs/eval_<adapter-name>_<timestamp>.json.
  --max-tokens N Forwarded to infer.py (default: infer.py's own default,
                 1536 — sized for the ~700-token p90 ground-truth JSON).
  -h, --help     Show this help and exit.

Example (full 325-dish test set):
  ./eval.sh adapters_v5 --limit 325
Example (quick 20-sample sanity check, explicit about the truncation):
  ./eval.sh adapters_v5 --limit 20 --out-json runs/eval_v5_quick20.json
EOF
}

ADAPTER_PATH=""
LIMIT=""
TEST_SET="$DEFAULT_TEST_SET"
MODEL_PATH="$DEFAULT_MODEL_PATH"
OUT_JSON=""
MAX_TOKENS=""

# First positional argument (before any flags) is the adapter path.
if [ $# -gt 0 ]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      ;;
    *)
      ADAPTER_PATH="$1"
      shift
      ;;
  esac
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --limit=*)
      LIMIT="${1#*=}"
      shift
      ;;
    --test-set)
      TEST_SET="$2"
      shift 2
      ;;
    --test-set=*)
      TEST_SET="${1#*=}"
      shift
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --out-json)
      OUT_JSON="$2"
      shift 2
      ;;
    --out-json=*)
      OUT_JSON="${1#*=}"
      shift
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --max-tokens=*)
      MAX_TOKENS="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "eval.sh: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "$ADAPTER_PATH" ]; then
  echo "eval.sh: ERROR: missing required first argument ADAPTER_PATH." >&2
  echo >&2
  usage >&2
  exit 1
fi

if [ -z "$LIMIT" ]; then
  cat >&2 <<EOF
eval.sh: ERROR: --limit is REQUIRED (no default).

infer.py's own --limit defaults to 20 samples. That default once silently
truncated what was meant to be a full overnight evaluation run down to 20
samples with no warning — the failure was only caught after the fact, when
the reported numbers looked too good. This wrapper refuses to guess: pass
--limit explicitly (e.g. --limit 325 for the full test set, or --limit 20
if you really do just want a quick sanity check).
EOF
  exit 1
fi

if [ -z "$OUT_JSON" ]; then
  mkdir -p runs
  adapter_name="$(basename "$ADAPTER_PATH")"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  OUT_JSON="runs/eval_${adapter_name}_${timestamp}.json"
fi
mkdir -p "$(dirname "$OUT_JSON")"

infer_args=(
  --test-set "$TEST_SET"
  --limit "$LIMIT"
  --model-path "$MODEL_PATH"
  --adapter-path "$ADAPTER_PATH"
  --out-json "$OUT_JSON"
)
if [ -n "$MAX_TOKENS" ]; then
  infer_args+=(--max-tokens "$MAX_TOKENS")
fi

echo "Evaluating adapter: $ADAPTER_PATH"
echo "Test set           : $TEST_SET (limit $LIMIT)"
echo "Output summary     : $OUT_JSON"
echo

.venv/bin/python infer.py "${infer_args[@]}"
