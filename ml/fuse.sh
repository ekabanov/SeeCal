#!/usr/bin/env bash
# fuse.sh — fuse a trained LoRA adapter into the base model offline, producing a
# standalone MLX model directory (no adapter needed at load). Thin wrapper around
# fuse_model.py — see its docstring for the full story.
#
# By default the fused model is RE-QUANTIZED to the base model's 4-bit format
# (~2.3 GB, on-device ready). Use --dequantize for fp16.
#
# Usage:
#   ./fuse.sh ADAPTER_PATH --out-path DIR [--base DIR] [--dequantize]
#             [--upload-repo REPO] [--help]
#
# Examples:
#   ./fuse.sh adapters_v5 --out-path fused_v5
#   ./fuse.sh adapters_v5 --out-path fused_v5 --upload-repo <user>/seecal-qwen3.5-4b-v5
#
# Publishing (--upload-repo) pushes to Hugging Face and requires you to be logged
# in first (`hf auth login`). A public repo is free to host. Do not publish
# without intending to — it is a public release.
#
# GPU note: fusing loads the full model and dequantizes/requantizes every layer.
# Metal memory doesn't swap — don't run this alongside train.sh/eval.sh or LM
# Studio (co-tenancy silently OOM-kills).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_BASE="$HOME/models/mlx-community/Qwen3.5-4B-MLX-4bit"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

ADAPTER_PATH=""
OUT_PATH=""
BASE="$DEFAULT_BASE"
DEQUANTIZE=""
UPLOAD_REPO=""

if [[ $# -gt 0 && "$1" != --* ]]; then
  ADAPTER_PATH="$1"; shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-path) OUT_PATH="$2"; shift 2 ;;
    --out-path=*) OUT_PATH="${1#*=}"; shift ;;
    --base) BASE="$2"; shift 2 ;;
    --base=*) BASE="${1#*=}"; shift ;;
    --dequantize) DEQUANTIZE="--dequantize"; shift ;;
    --upload-repo) UPLOAD_REPO="$2"; shift 2 ;;
    --upload-repo=*) UPLOAD_REPO="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "fuse.sh: unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$ADAPTER_PATH" ]]; then
  echo "fuse.sh: ERROR: missing ADAPTER_PATH (first argument)." >&2
  usage >&2
  exit 1
fi
if [[ -z "$OUT_PATH" ]]; then
  echo "fuse.sh: ERROR: --out-path is required." >&2
  usage >&2
  exit 1
fi

args=(--base "$BASE" --adapter-path "$ADAPTER_PATH" --out-path "$OUT_PATH")
[[ -n "$DEQUANTIZE" ]] && args+=("$DEQUANTIZE")
[[ -n "$UPLOAD_REPO" ]] && args+=(--upload-repo "$UPLOAD_REPO")

exec .venv/bin/python fuse_model.py "${args[@]}"
