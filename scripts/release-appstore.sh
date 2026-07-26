#!/bin/bash
# Build, sign, and upload SeeCal to App Store Connect for App Store release.
#
# Usage:
#   MODELS_DIR=/path/to/models scripts/release-appstore.sh [--dry-run]
#
#   --dry-run   Archive and export the signed .ipa but stop before uploading;
#               prints what would be uploaded.
#
# Prerequisites (one-time): scripts/release-setup.sh — stores the App Store
# Connect API key, bundle id, and team id under .secrets/.
#
# After upload: TestFlight and App Store submissions share the same uploaded
# build. Attach it to a version and submit for review in App Store Connect
# (https://appstoreconnect.apple.com/apps).
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "error: unknown argument '$arg' (see --help)" >&2
            exit 2
            ;;
    esac
done

# shellcheck source=scripts/release-common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/release-common.sh"

release_preflight
release_archive
release_export_or_upload "${DRY_RUN}" "App Store"
