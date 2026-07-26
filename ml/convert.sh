#!/usr/bin/env bash
# convert.sh — wraps convert_adapter_for_swift.py: converts an mlx-vlm LoRA
# adapter directory into the format mlx-swift-lm's LoRAContainer loads, and
# stamps the adapter version into the output adapter_config.json's
# "seecal_adapter_version" key (the key iOS's ModelInfoResolver looks for
# first — see ios/SeeCal/Sources/SeeCalInference/ModelInfoResolver.swift).
#
# Usage:
#   ./convert.sh ADAPTER_PATH [--output-path DIR] [--version VERSION] [--help]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage: ./convert.sh ADAPTER_PATH [--output-path DIR] [--version VERSION] [--help]

Converts an mlx-vlm LoRA adapter (adapter_config.json + adapters.safetensors)
into the format mlx-swift-lm's LoRAContainer expects, and stamps a
"seecal_adapter_version" key into the output adapter_config.json.

  ADAPTER_PATH     REQUIRED, first positional argument. mlx-vlm adapter
                   directory, e.g. adapters_v5.
  --output-path DIR
                   Output directory. Default: "<ADAPTER_PATH>_swift"
                   (e.g. adapters_v5 -> adapters_v5_swift).
  --version VERSION
                   Explicit version string to stamp (e.g. "v5"). Default:
                   derived from a trailing "_v<N>" in ADAPTER_PATH's
                   directory name — see convert_adapter_for_swift.py's
                   docstring for the exact rule and the iOS-side key name
                   it must match.
  -h, --help       Show this help and exit.

Example:
  ./convert.sh adapters_v5
  ./convert.sh adapters_v6 --output-path adapters_v6_swift --version v6
EOF
}

ADAPTER_PATH=""
OUTPUT_PATH=""
VERSION=""

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
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --output-path=*)
      OUTPUT_PATH="${1#*=}"
      shift
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --version=*)
      VERSION="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "convert.sh: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "$ADAPTER_PATH" ]; then
  echo "convert.sh: ERROR: missing required first argument ADAPTER_PATH." >&2
  echo >&2
  usage >&2
  exit 1
fi

if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
  echo "convert.sh: ERROR: $ADAPTER_PATH/adapter_config.json not found." >&2
  exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
  OUTPUT_PATH="${ADAPTER_PATH%/}_swift"
fi

convert_args=(--adapter-path "$ADAPTER_PATH" --output-path "$OUTPUT_PATH")
if [ -n "$VERSION" ]; then
  convert_args+=(--version "$VERSION")
fi

.venv/bin/python convert_adapter_for_swift.py "${convert_args[@]}"
