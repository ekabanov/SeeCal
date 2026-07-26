#!/usr/bin/env bash
# setup.sh — create (or verify) ml/.venv with the pinned SeeCal training stack.
#
# Idempotent: if ml/.venv already exists this script ONLY verifies the pinned
# package versions and reports status — it never deletes or recreates an
# existing venv. Delete ml/.venv yourself first if you want a clean rebuild.
#
# Pinned stack (verified against the venv actually used for the v5/v6 training
# runs — see CLAUDE.md "Toolchain v2"):
#   mlx-vlm 0.6.7, mlx 0.32.0, mlx-lm 0.31.3, transformers 5.14.1,
#   datasets 5.0.0, torch 2.13.0, torchvision 0.28.0, pytest (any recent 3.x/9.x)
#
# IMPORTANT — torch/torchvision are installed EXPLICITLY, as a separate pip
# step, deliberately. mlx-vlm's own extras (e.g. the "train" extra) do NOT
# pull in torch/torchvision. The Qwen3VL processor imports torchvision at
# init time; without it, `mlx_vlm.lora` fails ONLY once it reaches the
# processor step — well after argument parsing and dataset loading look fine,
# so the failure mode reads as "it's hanging" or a confusing late traceback
# rather than a clear missing-dependency error. Always install them yourself.
#
# Usage (run from anywhere; the script cds to its own directory):
#   ./setup.sh [--help]
#
# Requires python3.12 or newer available on PATH (as python3.12, python3.13,
# python3.14, ... or a python3 that itself resolves to >= 3.12). This script
# does not install Python itself — on macOS, `brew install python@3.12` (or
# newer) if none is found.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--help]

Creates ml/.venv with the pinned SeeCal training stack if it does not already
exist. If ml/.venv already exists, verifies the installed package versions
against the pins and reports status (does NOT recreate or modify the venv).

Pinned versions:
  mlx-vlm       0.6.7
  mlx           0.32.0
  mlx-lm        0.31.3
  transformers  5.14.1
  datasets      5.0.0
  torch         2.13.0
  torchvision   0.28.0
  pytest        (latest available; version not pinned)

Requirements:
  - python3.12 or newer on PATH (checked as python3.12/python3.13/python3.14/...
    or a python3 whose version is already >= 3.12)
  - macOS + Apple Silicon (mlx / mlx-vlm are Metal-only)

Notes:
  - torch and torchvision are installed as an explicit, separate pip step.
    mlx-vlm's extras do not pull them in, and the Qwen3VL processor needs
    torchvision at import time — training fails (late, confusingly) without it.
  - After setup, run:
      ./download_dataset.sh   # Nutrition5K subset into ml/Nutrition5K/
      ./download_model.sh     # base model into ~/models (or $MODELS_DIR)
    then see ml/README.md for the rest of the pipeline.

Exit codes: 0 on success (including the "already set up" fast path), 1 on
any failure (missing Python, failed installs, failed smoke check).
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "setup.sh: unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Pinned versions (single source of truth for both install and verification).
# Plain "name:version" lines rather than an associative array — macOS ships
# bash 3.2 as /bin/bash (no associative array support), and this script must
# run under whatever bash `env` finds first, not just a homebrew bash 4+.
# ---------------------------------------------------------------------------
PINNED_LIST="
mlx-vlm:0.6.7
mlx:0.32.0
mlx-lm:0.31.3
transformers:5.14.1
datasets:5.0.0
torch:2.13.0
torchvision:0.28.0
"
# pytest is intentionally unpinned (any recent version is fine); we only check
# that it is importable.

pinned_version() {
  # Prints the pinned version for dist name "$1", or empty if not found.
  local dist="$1"
  echo "$PINNED_LIST" | awk -F: -v d="$dist" '$1 == d {print $2}'
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Find a Python >= 3.12 on PATH. Prints the interpreter path on stdout.
find_python() {
  local candidate
  for candidate in python3.14 python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
        command -v "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

# Installed version of a distribution inside the venv, or empty string.
installed_version() {
  local dist="$1"
  "$VENV_DIR/bin/python" -m pip show "$dist" 2>/dev/null | awk -F': ' '/^Version:/ {print $2}'
}

# ---------------------------------------------------------------------------
# Fast path: venv already exists — verify pins, report, exit 0.
# ---------------------------------------------------------------------------
if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
  echo "ml/.venv already exists — verifying pinned versions (not recreating)."
  echo

  mismatch=0
  for dist in mlx-vlm mlx mlx-lm transformers datasets torch torchvision; do
    want="$(pinned_version "$dist")"
    got="$(installed_version "$dist")"
    if [ -z "$got" ]; then
      printf '  %-14s MISSING (want %s)\n' "$dist" "$want"
      mismatch=1
    elif [ "$got" != "$want" ]; then
      printf '  %-14s %s (pinned: %s) — MISMATCH\n' "$dist" "$got" "$want"
      mismatch=1
    else
      printf '  %-14s %s OK\n' "$dist" "$got"
    fi
  done

  if "$VENV_DIR/bin/python" -m pip show pytest >/dev/null 2>&1; then
    printf '  %-14s %s OK (unpinned)\n' "pytest" "$(installed_version pytest)"
  else
    echo "  pytest         MISSING"
    mismatch=1
  fi

  echo
  if [ "$mismatch" -ne 0 ]; then
    echo "Status: ml/.venv exists but does NOT match the pinned stack exactly."
    echo "Not recreating automatically — delete ml/.venv and rerun this script"
    echo "for a clean install if you want the exact pins restored."
  else
    echo "Status: ml/.venv matches the pinned stack."
  fi

  echo
  echo "Import smoke check:"
  "$VENV_DIR/bin/python" -c "
import mlx_vlm, torch, torchvision, PIL, numpy
print('  mlx_vlm     ', mlx_vlm.__version__)
print('  torch       ', torch.__version__)
print('  torchvision ', torchvision.__version__)
print('  PIL         ', PIL.__version__)
print('  numpy       ', numpy.__version__)
"

  echo
  echo "Next steps:"
  echo "  ./download_dataset.sh   # Nutrition5K subset into ml/Nutrition5K/"
  echo "  ./download_model.sh     # base model into ~/models (or \$MODELS_DIR)"
  echo "  see ml/README.md for the rest of the pipeline"
  exit 0
fi

# ---------------------------------------------------------------------------
# Slow path: create the venv from scratch.
# ---------------------------------------------------------------------------
echo "ml/.venv not found — creating it with the pinned stack."

PYTHON_BIN="$(find_python)" || {
  echo "ERROR: no python3.12+ interpreter found on PATH." >&2
  echo "Checked: python3.14, python3.13, python3.12, python3." >&2
  echo "On macOS: brew install python@3.12  (or a newer python@3.1x)" >&2
  exit 1
}
echo "Using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PY="$VENV_DIR/bin/python"

echo "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip

echo "Installing pinned mlx-vlm stack..."
"$VENV_PY" -m pip install \
  "mlx-vlm==$(pinned_version mlx-vlm)" \
  "mlx==$(pinned_version mlx)" \
  "mlx-lm==$(pinned_version mlx-lm)" \
  "transformers==$(pinned_version transformers)" \
  "datasets==$(pinned_version datasets)" \
  "pytest"

# torch/torchvision installed SEPARATELY and EXPLICITLY — see header comment.
# mlx-vlm's extras do not pull these in; the Qwen3VL processor needs
# torchvision at import time and training fails silently/late without it.
echo "Installing torch/torchvision explicitly (NOT pulled in by mlx-vlm)..."
"$VENV_PY" -m pip install \
  "torch==$(pinned_version torch)" \
  "torchvision==$(pinned_version torchvision)"

echo
echo "Import smoke check:"
"$VENV_PY" -c "
import mlx_vlm, torch, torchvision, PIL, numpy
print('  mlx_vlm     ', mlx_vlm.__version__)
print('  torch       ', torch.__version__)
print('  torchvision ', torchvision.__version__)
print('  PIL         ', PIL.__version__)
print('  numpy       ', numpy.__version__)
"

echo
echo "ml/.venv created successfully."
echo
echo "Next steps:"
echo "  ./download_dataset.sh   # Nutrition5K subset into ml/Nutrition5K/"
echo "  ./download_model.sh     # base model into ~/models (or \$MODELS_DIR)"
echo "  see ml/README.md for the rest of the pipeline"
