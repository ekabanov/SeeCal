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
#
# The LoRA adapter is NOT expected in MODELS_DIR: it is a build artifact of the
# ml/ pipeline and is sourced from ml/<SHIPPING_ADAPTER> in the repo by default
# (see the adapter step below). Staging one at $MODELS_DIR/adapters/ is an
# optional override that takes precedence when present.
#
# Escape hatch: SEECAL_ALLOW_NO_WEIGHTS=1 skips bundling entirely. This is
# for simulator development only — ModelAssetResolver falls back to local
# paths on the simulator, and the app cannot run inference on device without
# bundled (or side-loaded) weights.
set -euo pipefail

MODEL_FOLDER="mlx-community/Qwen3.5-4B-MLX-4bit"
DEST="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/Models"

# Simulator builds never bundle weights: the app runs the mock inference engine
# there (ProductionRootView) and ModelAssetResolver falls back to local host
# paths. Skip unconditionally so "open in Xcode → Run" on a simulator just works
# — with or without MODELS_DIR.
if [[ "${PLATFORM_NAME:-}" == "iphonesimulator" ]]; then
    echo "note: simulator build — skipping weight bundling."
    rm -rf "${DEST}"
    exit 0
fi

# Explicit opt-out for a device build without weights (it builds/installs but
# cannot run inference).
if [[ "${SEECAL_ALLOW_NO_WEIGHTS:-0}" == "1" ]]; then
    echo "note: SEECAL_ALLOW_NO_WEIGHTS=1 — skipping weight bundling."
    rm -rf "${DEST}"
    exit 0
fi

# Device build. Resolve MODELS_DIR in priority order:
#   1. explicit environment — scripts/build.sh, the release scripts, or an
#      Xcode build setting all export it this way (wins).
#   2. .secrets/release.env (written by scripts/release-setup.sh). The CLI
#      scripts source this themselves; a plain Xcode GUI Run does not, so we
#      source it HERE too — otherwise the configured MODELS_DIR is invisible to
#      Xcode and the build fails even though .secrets has it.
#   3. the canonical ~/models default (ml/download_model.sh's install location).
if [[ -z "${MODELS_DIR:-}" ]]; then
    SECRETS_ENV="${PROJECT_DIR}/../../.secrets/release.env"
    if [[ -f "${SECRETS_ENV}" ]]; then
        echo "note: MODELS_DIR not in environment — sourcing ${SECRETS_ENV}"
        # shellcheck source=/dev/null
        source "${SECRETS_ENV}"
    fi
fi
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
echo "Bundling weights from MODELS_DIR=${MODELS_DIR}"

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

# Adapter: the shipping fine-tune. It is a build artifact of the ml/ pipeline
# (output of ml/convert.sh), so by default it is sourced straight from the repo
# — no manual staging into MODELS_DIR is required. To ship a different adapter,
# either update SHIPPING_ADAPTER below or drop one at $MODELS_DIR/adapters/
# (which takes precedence when present).
SHIPPING_ADAPTER="adapters_v7b_swift"  # ml/<this> — change when the shipping adapter changes
REPO_ADAPTER="${PROJECT_DIR}/../../ml/${SHIPPING_ADAPTER}"

ADAPTER_SRC=""
if [[ -f "${MODELS_DIR}/adapters/adapter_config.json" && -f "${MODELS_DIR}/adapters/adapters.safetensors" ]]; then
    ADAPTER_SRC="${MODELS_DIR}/adapters"
elif [[ -f "${REPO_ADAPTER}/adapter_config.json" && -f "${REPO_ADAPTER}/adapters.safetensors" ]]; then
    ADAPTER_SRC="${REPO_ADAPTER}"
fi

if [[ -n "${ADAPTER_SRC}" ]]; then
    echo "Bundling LoRA adapter from ${ADAPTER_SRC}..."
    rsync -a --delete "${ADAPTER_SRC}/" "${DEST}/adapters/"
else
    echo "warning: no adapter found (looked in ${MODELS_DIR}/adapters and ${REPO_ADAPTER}) —" >&2
    echo "         app will run the BASE model, not the fine-tune. Run ml/convert.sh adapters_v7b" >&2
    echo "         to produce ml/${SHIPPING_ADAPTER}, or stage one at ${MODELS_DIR}/adapters." >&2
    rm -rf "${DEST}/adapters"
fi

echo "Model weights bundled into ${DEST}"
