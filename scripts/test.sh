#!/bin/bash
# Run the whole SeeCal monorepo test story with a single exit code:
#
#   1. ml pytest suite            (ml/tests, runs from ml/ with ml/.venv)
#   2. Swift package tests        (ios/SeeCal, swift test on macOS)
#   3. iOS app build check        (scripts/build.sh, simulator, no weights)
#
# Usage:
#   scripts/test.sh               Run everything
#   scripts/test.sh --skip-build  Skip the (slow) iOS app build check
#
# All suites run even if an earlier one fails; the summary at the end lists
# every failure and the exit code is non-zero if anything failed.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --help|-h)
            awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "error: unknown argument '$arg' (see scripts/test.sh --help)" >&2
            exit 2
            ;;
    esac
done

FAILURES=()

banner() {
    echo
    echo "=============================================================="
    echo "==> $1"
    echo "=============================================================="
}

# 1. ml pytest -----------------------------------------------------------
banner "1/3 ml pytest suite"
if [[ ! -x "${REPO_ROOT}/ml/.venv/bin/python" ]]; then
    echo "error: ml/.venv not found — run ml/setup.sh first." >&2
    FAILURES+=("ml pytest (ml/.venv missing — run ml/setup.sh)")
else
    if ! (cd "${REPO_ROOT}/ml" && .venv/bin/python -m pytest tests/); then
        FAILURES+=("ml pytest")
    fi
fi

# 2. Swift package tests -------------------------------------------------
banner "2/3 Swift package tests (ios/SeeCal)"
# -DFMT_CONSTEVAL=: Xcode 26 clang vs the fmt vendored in mlx-swift.
if ! (cd "${REPO_ROOT}/ios/SeeCal" && swift test -Xcxx -DFMT_CONSTEVAL=); then
    FAILURES+=("swift test (ios/SeeCal)")
fi

# 3. iOS app build check -------------------------------------------------
if [[ "${SKIP_BUILD}" == "1" ]]; then
    banner "3/3 iOS app build check — SKIPPED (--skip-build)"
else
    banner "3/3 iOS app build check (simulator)"
    # A build check needs no weights; honour MODELS_DIR if the caller set it.
    if [[ -n "${MODELS_DIR:-}" ]]; then
        if ! "${REPO_ROOT}/scripts/build.sh"; then
            FAILURES+=("iOS app build (scripts/build.sh)")
        fi
    else
        if ! SEECAL_ALLOW_NO_WEIGHTS=1 "${REPO_ROOT}/scripts/build.sh"; then
            FAILURES+=("iOS app build (scripts/build.sh)")
        fi
    fi
fi

# Summary ----------------------------------------------------------------
echo
echo "=============================================================="
if [[ ${#FAILURES[@]} -eq 0 ]]; then
    echo "All test suites passed."
    exit 0
fi
echo "FAILED suites:"
for f in "${FAILURES[@]}"; do
    echo "  - $f"
done
exit 1
