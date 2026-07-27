#!/bin/bash
# Regenerate ios/App/SeeCal.xcodeproj and open it in Xcode, with your signing
# team and bundle id baked in so a device Run just works.
#
# project.yml reads PRODUCT_BUNDLE_IDENTIFIER / DEVELOPMENT_TEAM from the
# environment (XcodeGen ${VAR} substitution). This script sources
# .secrets/release.env (written by scripts/release-setup.sh) and always sets
# them — so the team persists across regenerates instead of being wiped to the
# com.example placeholder every time (the "Signing requires a development team"
# error you hit).
#
# Usage:
#   scripts/gen-xcode.sh            regenerate + open in Xcode
#   scripts/gen-xcode.sh --no-open  regenerate only
#
# Without .secrets (public checkout / CI) it still works: bundle id defaults to
# com.example.seecal and the team is left empty (device signing then needs a
# team, but simulator builds are fine).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${REPO_ROOT}/ios/App"

if ! command -v xcodegen > /dev/null 2>&1; then
    echo "error: xcodegen is not installed. Install it with: brew install xcodegen" >&2
    exit 1
fi

# Defaults for a public checkout; overridden by .secrets below.
BUNDLE_ID="com.example.seecal"
DEVELOPMENT_TEAM=""
if [[ -f "${REPO_ROOT}/.secrets/release.env" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.secrets/release.env"
fi

# XcodeGen substitutes these into project.yml. They must be SET (even if empty),
# or XcodeGen leaves the literal ${...} in the project.
export SEECAL_BUNDLE_ID="${BUNDLE_ID:-com.example.seecal}"
export SEECAL_DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM:-}"

echo "==> Generating SeeCal.xcodeproj (bundle=${SEECAL_BUNDLE_ID}, team=${SEECAL_DEVELOPMENT_TEAM:-<none>})"
(cd "${APP_DIR}" && xcodegen generate)

if [[ "${1:-}" != "--no-open" ]]; then
    echo "==> Opening in Xcode"
    open "${APP_DIR}/SeeCal.xcodeproj"
fi
