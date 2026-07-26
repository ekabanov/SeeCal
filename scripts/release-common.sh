#!/bin/bash
# Shared logic for scripts/release-testflight.sh and scripts/release-appstore.sh.
# Not executable on its own — the release scripts source this file.
#
# Upload mechanism (verified against Xcode 26.6, July 2026): `xcodebuild
# -exportArchive` with `method: app-store-connect` + `destination: upload` in
# ExportOptions.plist, authenticated with an App Store Connect API key via
# -authenticationKeyPath/-authenticationKeyID/-authenticationKeyIssuerID.
# This replaces the deprecated `altool --upload-app` flow (altool still ships
# but has known upload breakage under Xcode 26; Apple's supported CLI path is
# xcodebuild itself, or the Transporter app for manual uploads).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${REPO_ROOT}/ios/App"
RELEASE_DIR="${APP_DIR}/build/release"
ARCHIVE_PATH="${RELEASE_DIR}/SeeCal.xcarchive"
EXPORT_PATH="${RELEASE_DIR}/export"
EXPORT_PLIST="${RELEASE_DIR}/ExportOptions.plist"

release_preflight() {
    # 1. Credentials — created by scripts/release-setup.sh. Auth is EXCLUSIVELY
    # the App Store Connect API key stored under .secrets/ (passed to both the
    # archive and exportArchive steps); the account signed into Xcode is never
    # used as a fallback.
    if [[ ! -f "${REPO_ROOT}/.secrets/release.env" ]]; then
        echo "error: no release credentials found (.secrets/release.env missing)." >&2
        echo "Run scripts/release-setup.sh once to store your App Store Connect" >&2
        echo "API key, bundle id, and team id." >&2
        exit 1
    fi
    _env_models_dir="${MODELS_DIR:-}"
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.secrets/release.env"
    # An explicitly exported MODELS_DIR wins over the saved one.
    [[ -n "${_env_models_dir}" ]] && MODELS_DIR="${_env_models_dir}"
    for var in ASC_KEY_ID ASC_ISSUER_ID ASC_KEY_PATH BUNDLE_ID DEVELOPMENT_TEAM; do
        if [[ -z "${!var:-}" ]]; then
            echo "error: ${var} is missing from .secrets/release.env." >&2
            echo "Re-run scripts/release-setup.sh to repair the configuration." >&2
            exit 1
        fi
    done
    if [[ ! -f "${ASC_KEY_PATH}" ]]; then
        echo "error: App Store Connect key file not found at ${ASC_KEY_PATH}." >&2
        echo "Re-run scripts/release-setup.sh and provide the .p8 file again." >&2
        exit 1
    fi
    AUTH_ARGS=(
        -authenticationKeyPath "${ASC_KEY_PATH}"
        -authenticationKeyID "${ASC_KEY_ID}"
        -authenticationKeyIssuerID "${ASC_ISSUER_ID}"
    )

    # 2. Tooling.
    if ! command -v xcodegen > /dev/null 2>&1; then
        echo "error: xcodegen is not installed. Install it with:" >&2
        echo "  brew install xcodegen" >&2
        exit 1
    fi

    # 3. Weights — a release build must bundle them.
    if [[ -z "${MODELS_DIR:-}" ]]; then
        echo "error: MODELS_DIR is not set — a release build must bundle model weights." >&2
        echo "Re-run scripts/release-setup.sh (it stores MODELS_DIR), or:" >&2
        echo "Point MODELS_DIR at a directory containing:" >&2
        echo "  mlx-community/Qwen3.5-4B-MLX-4bit/   (ml/download_model.sh)" >&2
        echo "  adapters/                            (ml/convert.sh output, optional)" >&2
        echo "then re-run, e.g.:" >&2
        echo "  MODELS_DIR=/path/to/models scripts/release-testflight.sh" >&2
        exit 1
    fi
}

release_version() { # MARKETING_VERSION from project.yml, for messages
    awk -F'"' '/MARKETING_VERSION/ { print $2; exit }' "${APP_DIR}/project.yml"
}

release_archive() {
    echo "==> Generating Xcode project from ios/App/project.yml"
    (cd "${APP_DIR}" && xcodegen generate --quiet)

    mkdir -p "${RELEASE_DIR}"
    rm -rf "${ARCHIVE_PATH}"

    echo "==> Archiving (Release, device, team ${DEVELOPMENT_TEAM}, bundle ${BUNDLE_ID})"
    # -allowProvisioningUpdates is required for non-interactive archive/export;
    # AUTH_ARGS adds ASC API-key auth on top when .secrets are configured.
    if ! xcodebuild archive \
        -project "${APP_DIR}/SeeCal.xcodeproj" \
        -scheme SeeCal \
        -configuration Release \
        -destination "generic/platform=iOS" \
        -archivePath "${ARCHIVE_PATH}" \
        -derivedDataPath "${APP_DIR}/build/DerivedData" \
        -allowProvisioningUpdates \
        "${AUTH_ARGS[@]}" \
        "DEVELOPMENT_TEAM=${DEVELOPMENT_TEAM}" \
        "PRODUCT_BUNDLE_IDENTIFIER=${BUNDLE_ID}" \
        "MODELS_DIR=${MODELS_DIR}" \
        "OTHER_CPLUSPLUSFLAGS=\$(inherited) -DFMT_CONSTEVAL="; then
        echo >&2
        echo "error: archive failed. If the failure is signing-related, check that:" >&2
        echo "  - your Apple Developer Program membership is active for team ${DEVELOPMENT_TEAM}," >&2
        echo "  - the bundle id ${BUNDLE_ID} is yours (registered or free to auto-register)," >&2
        echo "  - the ASC API key from release-setup.sh has App Manager access" >&2
        echo "    (automatic signing uses it via -allowProvisioningUpdates)." >&2
        echo "Alternatively open ios/App/SeeCal.xcodeproj in Xcode once, select your" >&2
        echo "team under Signing & Capabilities, and resolve any signing prompts." >&2
        exit 1
    fi
}

write_export_plist() { # <destination: export|upload>
    cat > "${EXPORT_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key>
	<string>app-store-connect</string>
	<key>destination</key>
	<string>$1</string>
	<key>signingStyle</key>
	<string>automatic</string>
	<key>teamID</key>
	<string>${DEVELOPMENT_TEAM}</string>
	<key>manageAppVersionAndBuildNumber</key>
	<true/>
	<key>uploadSymbols</key>
	<true/>
</dict>
</plist>
EOF
}

release_export_or_upload() { # <dry_run: 0|1> <track label, e.g. "TestFlight">
    local dry_run="$1" label="$2" destination verb
    if [[ "${dry_run}" == "1" ]]; then
        destination="export"
        verb="Exporting signed .ipa (dry run — no upload)"
    else
        destination="upload"
        verb="Uploading to App Store Connect (${label})"
    fi
    write_export_plist "${destination}"
    rm -rf "${EXPORT_PATH}"

    echo "==> ${verb}"
    # -allowProvisioningUpdates is required here too: without it,
    # "destination: upload" fails with "exportArchive Failed to Use Accounts".
    if ! xcodebuild -exportArchive \
        -archivePath "${ARCHIVE_PATH}" \
        -exportOptionsPlist "${EXPORT_PLIST}" \
        -exportPath "${EXPORT_PATH}" \
        -allowProvisioningUpdates \
        "${AUTH_ARGS[@]}"; then
        echo >&2
        echo "error: export/upload failed. Common causes:" >&2
        echo "  - the app record for ${BUNDLE_ID} does not exist yet in App Store" >&2
        echo "    Connect (create it at https://appstoreconnect.apple.com/apps)," >&2
        echo "  - the ASC API key lacks App Manager access," >&2
        echo "  - a build with the same version/build number was already uploaded." >&2
        exit 1
    fi

    local version
    version="$(release_version)"
    if [[ "${dry_run}" == "1" ]]; then
        echo
        echo "Dry run complete. Would upload v${version} (${BUNDLE_ID}) to App Store Connect (${label}):"
        find "${EXPORT_PATH}" -name '*.ipa' -exec du -h {} \;
        echo "Re-run without --dry-run to upload."
    else
        echo
        echo "Build v${version} (${BUNDLE_ID}) uploaded to App Store Connect."
        echo "Processing takes a while (a ~2.3 GB weights bundle is slow); then:"
        if [[ "${label}" == "TestFlight" ]]; then
            echo "  App Store Connect -> Apps -> SeeCal -> TestFlight: the build appears"
            echo "  automatically after processing; add internal/external testers there."
        else
            echo "  App Store Connect -> Apps -> SeeCal -> App Store tab -> your version:"
            echo "  click 'Add Build', select v${version}, then 'Submit for Review'."
            echo "  (Uploading never submits automatically — review submission is manual.)"
        fi
    fi
}
