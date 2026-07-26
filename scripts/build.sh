#!/bin/bash
# Build the SeeCal iOS app.
#
# Usage:
#   scripts/build.sh              Debug build for the iOS Simulator (no signing)
#   scripts/build.sh --device     Signed Debug build for a physical device
#   scripts/build.sh --help
#
# Environment:
#   MODELS_DIR                Directory with model weights to bundle into the
#                             app (see ios/App/copy_weights.sh for the layout).
#   SEECAL_ALLOW_NO_WEIGHTS   Set to 1 to build without weights (simulator dev
#                             mode). Without MODELS_DIR or this flag the build
#                             fails with instructions.
#   BUNDLE_ID                 Overrides the com.example.seecal placeholder
#                             (required for --device unless in .secrets/release.env).
#   DEVELOPMENT_TEAM          Apple Developer team id (required for --device
#                             unless in .secrets/release.env).
#
# The Xcode project is regenerated from ios/App/project.yml on every run
# (SeeCal.xcodeproj is not committed). Build products and DerivedData live
# under ios/App/build/ (gitignored).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${REPO_ROOT}/ios/App"
DERIVED_DATA="${APP_DIR}/build/DerivedData"

DEVICE_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --device) DEVICE_BUILD=1 ;;
        --help|-h)
            awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "error: unknown argument '$arg' (see scripts/build.sh --help)" >&2
            exit 2
            ;;
    esac
done

if ! command -v xcodegen > /dev/null 2>&1; then
    echo "error: xcodegen is not installed. Install it with:" >&2
    echo "  brew install xcodegen" >&2
    exit 1
fi

echo "==> Generating Xcode project from ios/App/project.yml"
(cd "${APP_DIR}" && xcodegen generate --quiet)

# The FMT_CONSTEVAL flag must be a command-line override so it reaches the
# SwiftPM package targets (project settings do not propagate into packages).
XCODEBUILD_ARGS=(
    -project "${APP_DIR}/SeeCal.xcodeproj"
    -scheme SeeCal
    -configuration Debug
    -derivedDataPath "${DERIVED_DATA}"
    "OTHER_CPLUSPLUSFLAGS=\$(inherited) -DFMT_CONSTEVAL="
)

# Pass weight-bundling knobs through as build settings so the
# "Bundle model weights" script phase can see them.
if [[ -n "${MODELS_DIR:-}" ]]; then
    XCODEBUILD_ARGS+=("MODELS_DIR=${MODELS_DIR}")
fi
if [[ -n "${SEECAL_ALLOW_NO_WEIGHTS:-}" ]]; then
    XCODEBUILD_ARGS+=("SEECAL_ALLOW_NO_WEIGHTS=${SEECAL_ALLOW_NO_WEIGHTS}")
fi

if [[ "${DEVICE_BUILD}" == "1" ]]; then
    # Pick up BUNDLE_ID / DEVELOPMENT_TEAM from release-setup.sh if present;
    # explicit environment variables win.
    if [[ -f "${REPO_ROOT}/.secrets/release.env" ]]; then
        _env_bundle_id="${BUNDLE_ID:-}"
        _env_team="${DEVELOPMENT_TEAM:-}"
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/.secrets/release.env"
        [[ -n "${_env_bundle_id}" ]] && BUNDLE_ID="${_env_bundle_id}"
        [[ -n "${_env_team}" ]] && DEVELOPMENT_TEAM="${_env_team}"
    fi
    if [[ -z "${DEVELOPMENT_TEAM:-}" ]]; then
        echo "error: --device needs a signing team. Either:" >&2
        echo "  - run scripts/release-setup.sh once (stores it in .secrets/release.env), or" >&2
        echo "  - export DEVELOPMENT_TEAM=<your team id> (and optionally BUNDLE_ID)." >&2
        exit 1
    fi
    XCODEBUILD_ARGS+=(
        -destination "generic/platform=iOS"
        -allowProvisioningUpdates
        "DEVELOPMENT_TEAM=${DEVELOPMENT_TEAM}"
    )
    if [[ -n "${BUNDLE_ID:-}" ]]; then
        XCODEBUILD_ARGS+=("PRODUCT_BUNDLE_IDENTIFIER=${BUNDLE_ID}")
    fi
    echo "==> Building Debug for device (signed, team ${DEVELOPMENT_TEAM})"
else
    XCODEBUILD_ARGS+=(
        -destination "generic/platform=iOS Simulator"
        CODE_SIGNING_ALLOWED=NO
    )
    echo "==> Building Debug for the iOS Simulator (unsigned)"
fi

xcodebuild "${XCODEBUILD_ARGS[@]}" build

echo "==> Build succeeded. App products under ${DERIVED_DATA}/Build/Products/"
