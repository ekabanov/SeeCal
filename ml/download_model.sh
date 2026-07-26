#!/usr/bin/env bash
# download_model.sh — fetch the SeeCal base model from Hugging Face.
#
# Downloads mlx-community/Qwen3.5-4B-MLX-4bit into MODELS_DIR (default
# ~/models), producing MODELS_DIR/Qwen3.5-4B-MLX-4bit/. This is the model
# ml/train.sh and ml/infer.py use via the MODEL_PATH convention.
#
# Idempotent: uses `hf download --local-dir`, which only fetches
# missing/changed files and resumes partial downloads — safe to re-run.
#
# Usage:
#   ./download_model.sh [--dir DIR] [--help]
# Or:
#   MODELS_DIR=/custom/path ./download_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

REPO_ID="mlx-community/Qwen3.5-4B-MLX-4bit"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

usage() {
  cat <<EOF
Usage: ./download_model.sh [--dir DIR] [--help]

Downloads $REPO_ID from Hugging Face into
DIR/Qwen3.5-4B-MLX-4bit/ (DIR defaults to \$MODELS_DIR, or ~/models if
that env var is unset).

  --dir DIR    Override the models directory (same as setting MODELS_DIR).
  -h, --help   Show this help and exit.

Resolution order for downloading:
  1. system 'hf' on PATH             (modern huggingface_hub CLI)
  2. system 'huggingface-cli' on PATH (deprecated alias, still works)
  3. ml/.venv's huggingface_hub, invoked directly (installed there already
     as a transitive dependency of transformers/datasets; if somehow
     missing, this script pip-installs it into ml/.venv). Requires
     ml/.venv to exist — run ./setup.sh first if it doesn't.

Idempotent: re-running skips files already downloaded intact and resumes
any partial transfer (this is 'hf download --local-dir' behavior, not
something this script implements itself).

After downloading, this script prints the resolved model path and the
MODEL_PATH environment variable to set for train.sh / eval.sh (wired up by
a later task in the open-source reorg — for now this only prints the path).
EOF
}

DIR_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dir)
      DIR_OVERRIDE="$2"
      shift 2
      ;;
    --dir=*)
      DIR_OVERRIDE="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "download_model.sh: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done
if [ -n "$DIR_OVERRIDE" ]; then
  MODELS_DIR="$DIR_OVERRIDE"
fi

MODEL_DIR="$MODELS_DIR/mlx-community/Qwen3.5-4B-MLX-4bit"

# ---------------------------------------------------------------------------
# Pick a way to run the 'hf' CLI.
# ---------------------------------------------------------------------------

# Runs `hf <args...>` via whichever mechanism is available. Defined as a
# function (not a resolved path) because the ml/.venv fallback invokes the
# CLI's Python entry point directly rather than through a shebang script —
# console-script shebangs embed an absolute interpreter path recorded at
# venv-creation time, which breaks if the venv directory is ever moved or
# renamed (as happened once already in this repo's history) even though the
# venv itself still works fine via `python -m`/direct import.
run_hf() {
  if command -v hf >/dev/null 2>&1; then
    hf "$@"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli "$@"
  else
    if [ ! -x "$VENV_PY" ]; then
      echo "ERROR: no system 'hf'/'huggingface-cli' found, and ml/.venv does not exist." >&2
      echo "Run ./setup.sh first, or install the Hugging Face CLI system-wide:" >&2
      echo "  pip install -U 'huggingface_hub[cli]'" >&2
      exit 1
    fi
    if ! "$VENV_PY" -m pip show huggingface_hub >/dev/null 2>&1; then
      echo "huggingface_hub not found in ml/.venv — installing it there..."
      "$VENV_PY" -m pip install -U "huggingface_hub[cli]"
    fi
    "$VENV_PY" -c '
import sys
from huggingface_hub.cli.hf import main
sys.argv = ["hf"] + sys.argv[1:]
sys.exit(main())
' "$@"
  fi
}

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

if [ -f "$MODEL_DIR/config.json" ]; then
  echo "Model directory already has a config.json — likely already downloaded."
  echo "Verifying/completing via hf download (idempotent, resumes if needed)..."
fi

mkdir -p "$MODEL_DIR"

echo "Downloading $REPO_ID -> $MODEL_DIR ..."
run_hf download "$REPO_ID" --local-dir "$MODEL_DIR"

echo
echo "Model ready at: $MODEL_DIR"
echo
echo "To use it with train.sh / infer.py (MODEL_PATH convention):"
echo "  export MODEL_PATH=\"$MODEL_DIR\""
