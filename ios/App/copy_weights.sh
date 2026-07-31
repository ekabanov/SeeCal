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
# Optional local override:
#   $MODELS_DIR/visual-specialist/SeeCalVisualSpecialist.mlmodelc/
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
# SEECAL_MODEL_STACK selects "factored" (default) or the protected "v8"
# rollback stack.
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

# Adapter: the selected device-test fine-tune. It is a build artifact of the ml/ pipeline
# (output of ml/convert.sh), so by default it is sourced straight from the repo
# — no manual staging into MODELS_DIR is required. A matching stamped adapter
# at $MODELS_DIR/adapters/ takes precedence when present.
MODEL_STACK="${SEECAL_MODEL_STACK:-factored}"
case "${MODEL_STACK}" in
    factored)
        SHIPPING_ADAPTER="runs/factored/e3-v2-foodseg-1024-warm-start-12000-lr1e-5/adapter_swift"
        SHIPPING_ADAPTER_VERSION="factored-e3-foodseg-c1"
        ;;
    v8)
        SHIPPING_ADAPTER="adapters_v8_numeric_4b_swift"
        SHIPPING_ADAPTER_VERSION="v8-conditioned"
        ;;
    *)
        echo "error: SEECAL_MODEL_STACK must be 'factored' or 'v8', got '${MODEL_STACK}'." >&2
        exit 1
        ;;
esac
echo "Bundling model stack: ${MODEL_STACK}"
REPO_ADAPTER="${PROJECT_DIR}/../../ml/${SHIPPING_ADAPTER}"

ADAPTER_SRC=""
if [[ -f "${MODELS_DIR}/adapters/adapter_config.json" && -f "${MODELS_DIR}/adapters/adapters.safetensors" ]]; then
    ADAPTER_SRC="${MODELS_DIR}/adapters"
elif [[ -f "${REPO_ADAPTER}/adapter_config.json" && -f "${REPO_ADAPTER}/adapters.safetensors" ]]; then
    ADAPTER_SRC="${REPO_ADAPTER}"
fi

if [[ -z "${ADAPTER_SRC}" ]]; then
    echo "error: '${SHIPPING_ADAPTER_VERSION}' adapter not found (looked in ${MODELS_DIR}/adapters and ${REPO_ADAPTER})." >&2
    exit 1
fi

FOUND_ADAPTER_VERSION="$(/usr/bin/plutil -extract seecal_adapter_version raw -o - "${ADAPTER_SRC}/adapter_config.json" 2>/dev/null || true)"
if [[ "${FOUND_ADAPTER_VERSION}" != "${SHIPPING_ADAPTER_VERSION}" ]]; then
    echo "error: adapter at ${ADAPTER_SRC} is '${FOUND_ADAPTER_VERSION:-unstamped}', expected '${SHIPPING_ADAPTER_VERSION}'." >&2
    echo "Remove a stale MODELS_DIR/adapters override or convert the selected adapter." >&2
    exit 1
fi
echo "Bundling '${SHIPPING_ADAPTER_VERSION}' LoRA adapter from ${ADAPTER_SRC}..."
rsync -a --delete "${ADAPTER_SRC}/" "${DEST}/adapters/"

# Remove the inactive stack's small Core ML artifact so an incremental build
# cannot accidentally pair the factored adapter with v8's specialist or vice
# versa.
rm -rf "${DEST}/visual-specialist" "${DEST}/scale"
if [[ "${MODEL_STACK}" == "factored" ]]; then
    SCALE_NAME="SeeCalScaleC1.mlmodelc"
    STAGED_SCALE="${MODELS_DIR}/scale/${SCALE_NAME}"
    REPO_SCALE="${PROJECT_DIR}/../../ml/runs/factored/deployment/compiled/${SCALE_NAME}"
    SCALE_SRC=""
    if [[ -d "${STAGED_SCALE}" ]]; then
        SCALE_SRC="${STAGED_SCALE}"
    elif [[ -d "${REPO_SCALE}" ]]; then
        SCALE_SRC="${REPO_SCALE}"
    fi
    if [[ -z "${SCALE_SRC}" ]]; then
        echo "error: compiled factored SCALE C1 model not found." >&2
        echo "Export and compile it to ${REPO_SCALE}." >&2
        exit 1
    fi
    echo "Bundling SCALE C1 from ${SCALE_SRC}..."
    mkdir -p "${DEST}/scale"
    rsync -a --delete "${SCALE_SRC}/" "${DEST}/scale/${SCALE_NAME}/"
else
    # The v8 adapter was trained to receive this specialist block.
    SPECIALIST_NAME="SeeCalVisualSpecialist.mlmodelc"
    STAGED_SPECIALIST="${MODELS_DIR}/visual-specialist/${SPECIALIST_NAME}"
    REPO_SPECIALIST="${PROJECT_DIR}/../../ml/runs/visual-specialist/deployment/${SPECIALIST_NAME}"
    SPECIALIST_SRC=""
    if [[ -d "${STAGED_SPECIALIST}" ]]; then
        SPECIALIST_SRC="${STAGED_SPECIALIST}"
    elif [[ -d "${REPO_SPECIALIST}" ]]; then
        SPECIALIST_SRC="${REPO_SPECIALIST}"
    fi
    if [[ -z "${SPECIALIST_SRC}" ]]; then
        echo "error: compiled visual specialist not found." >&2
        echo "Export the trained checkpoint and compile it to ${REPO_SPECIALIST}." >&2
        exit 1
    fi
    echo "Bundling visual specialist from ${SPECIALIST_SRC}..."
    mkdir -p "${DEST}/visual-specialist"
    rsync -a --delete "${SPECIALIST_SRC}/" "${DEST}/visual-specialist/${SPECIALIST_NAME}/"
fi

# The correction-first review flow uses the same pruned, local USDA database
# as the factored pipeline. It is a generated artifact (gitignored), so bundle
# it from the canonical ML workspace at build time.
NUTRITION_DB="${PROJECT_DIR}/../../ml/datasets/fdc/seecal-nutrition.sqlite"
if [[ ! -f "${NUTRITION_DB}" ]]; then
    echo "error: nutrition database not found at ${NUTRITION_DB}." >&2
    echo "Regenerate it with the FDC steps in ml/README.md." >&2
    exit 1
fi
echo "Bundling local nutrition database..."
mkdir -p "${DEST}/nutrition"
cp "${NUTRITION_DB}" "${DEST}/nutrition/seecal-nutrition.sqlite"

echo "Model weights bundled into ${DEST}"
