#!/bin/bash
# Xcode build phase: bundle model weights into the app.
#
# Invoked by the "Bundle model weights" run-script phase declared in
# project.yml. Weights are NOT in git; they are copied at build time from a
# directory you provide via the MODELS_DIR build setting (scripts/build.sh and
# the release scripts pass it through from the environment).
#
# Required MODELS_DIR layout (matches ModelAssetResolver's bundled paths):
#   $MODELS_DIR/mlx-community/Qwen3.5-4B-MLX-4bit/   base model (~2.3 GB)
#   $MODELS_DIR/adapters/                            converted LoRA adapter
#                                                    (optional, ~35 MB; output
#                                                    of ml/convert.sh)
#
# Escape hatch: SEECAL_ALLOW_NO_WEIGHTS=1 skips bundling entirely. This is
# for simulator development only — ModelAssetResolver falls back to local
# paths on the simulator, and the app cannot run inference on device without
# bundled (or side-loaded) weights.
set -euo pipefail

MODEL_FOLDER="mlx-community/Qwen3.5-4B-MLX-4bit"
DEST="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/Models"

if [[ -z "${MODELS_DIR:-}" ]]; then
    # Simulator builds never need bundled weights: ModelAssetResolver falls back
    # to local host paths there. Auto-skip so "open in Xcode → Run" just works.
    if [[ "${PLATFORM_NAME:-}" == "iphonesimulator" ]]; then
        echo "note: simulator build without MODELS_DIR — skipping weight bundling."
        rm -rf "${DEST}"
        exit 0
    fi
    if [[ "${SEECAL_ALLOW_NO_WEIGHTS:-0}" == "1" ]]; then
        echo "note: SEECAL_ALLOW_NO_WEIGHTS=1 — skipping weight bundling (simulator dev mode)."
        rm -rf "${DEST}"
        exit 0
    fi
    cat >&2 <<'EOF'
error: MODELS_DIR is not set — SeeCal cannot bundle model weights.

The app bundles the base model and LoRA adapter at build time; they are not
in git. Point MODELS_DIR at a directory with this layout:

  $MODELS_DIR/mlx-community/Qwen3.5-4B-MLX-4bit/   base model (ml/download_model.sh)
  $MODELS_DIR/adapters/                            converted adapter (ml/convert.sh), optional

Then build with:

  MODELS_DIR=/path/to/models scripts/build.sh

For a simulator-only build without weights (ModelAssetResolver has simulator
fallbacks), set SEECAL_ALLOW_NO_WEIGHTS=1:

  SEECAL_ALLOW_NO_WEIGHTS=1 scripts/build.sh
EOF
    exit 1
fi

MODEL_SRC="${MODELS_DIR}/${MODEL_FOLDER}"
if [[ ! -f "${MODEL_SRC}/config.json" ]]; then
    echo "error: '${MODEL_SRC}/config.json' not found." >&2
    echo "MODELS_DIR must contain '${MODEL_FOLDER}/' with the MLX model files" >&2
    echo "(download with ml/download_model.sh)." >&2
    exit 1
fi
if ! compgen -G "${MODEL_SRC}/*.safetensors" > /dev/null; then
    echo "error: no .safetensors files in '${MODEL_SRC}' — model download incomplete?" >&2
    exit 1
fi

mkdir -p "${DEST}/${MODEL_FOLDER%/*}"
echo "Bundling base model from ${MODEL_SRC} (rsync, incremental)..."
rsync -a --delete "${MODEL_SRC}/" "${DEST}/${MODEL_FOLDER}/"

ADAPTER_SRC="${MODELS_DIR}/adapters"
if [[ -f "${ADAPTER_SRC}/adapter_config.json" && -f "${ADAPTER_SRC}/adapters.safetensors" ]]; then
    echo "Bundling LoRA adapter from ${ADAPTER_SRC}..."
    rsync -a --delete "${ADAPTER_SRC}/" "${DEST}/adapters/"
else
    echo "warning: no adapter at '${ADAPTER_SRC}' (need adapter_config.json + adapters.safetensors) — app will run the base model."
    rm -rf "${DEST}/adapters"
fi

echo "Model weights bundled into ${DEST}"
