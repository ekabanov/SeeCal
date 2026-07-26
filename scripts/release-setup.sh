#!/bin/bash
# One-time (re-runnable) release credential setup for SeeCal.
#
# Interactively collects the App Store Connect API credentials and signing
# identity, and stores them in repo-local .secrets/ so the release scripts
# never need manual environment setup:
#
#   .secrets/release.env   ASC key id, issuer id, key path, BUNDLE_ID,
#                          DEVELOPMENT_TEAM        (chmod 600)
#   .secrets/AuthKey.p8    your App Store Connect API private key (copied in,
#                          chmod 600)
#
# Where to find the values: App Store Connect -> Users and Access ->
# Integrations -> App Store Connect API. Create a key with "App Manager"
# access and download the .p8 file (Apple lets you download it exactly once).
#
# Safety: .secrets/ is verified to be gitignored BEFORE anything is written;
# the script refuses to proceed otherwise. Secret values are prompted with
# echo disabled and are never printed or logged. Re-running updates values
# (press Enter at any prompt to keep the current value).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_DIR="${REPO_ROOT}/.secrets"
ENV_FILE="${SECRETS_DIR}/release.env"
KEY_FILE="${SECRETS_DIR}/AuthKey.p8"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
    exit 0
fi

# --- 1. gitignore coverage FIRST — refuse to write anything otherwise -----
GITIGNORE="${REPO_ROOT}/.gitignore"
if ! grep -qxF '.secrets/' "${GITIGNORE}" 2> /dev/null; then
    echo "Adding '.secrets/' to .gitignore"
    printf '\n# Release credentials (scripts/release-setup.sh) — never commit\n.secrets/\n' >> "${GITIGNORE}"
fi
if ! git -C "${REPO_ROOT}" check-ignore -q .secrets/x; then
    echo "error: git does not ignore '.secrets/' even after updating .gitignore." >&2
    echo "Refusing to write credentials into a path git could track." >&2
    echo "Fix .gitignore (it must contain a '.secrets/' line) and re-run." >&2
    exit 1
fi

# --- 2. secrets directory --------------------------------------------------
mkdir -p "${SECRETS_DIR}"
chmod 700 "${SECRETS_DIR}"

# Existing values (re-run support). release.env is our own chmod-600 file.
ASC_KEY_ID="" ASC_ISSUER_ID="" BUNDLE_ID="" DEVELOPMENT_TEAM=""
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    echo "Existing configuration found — press Enter at any prompt to keep the current value."
fi

# --- 3. prompts ------------------------------------------------------------
# Secret-ish identifiers are read with echo disabled and never printed back.
prompt_hidden() { # <varname> <label> — empty input keeps the existing value
    local __var="$1" __label="$2" __cur __val
    __cur="${!__var:-}"
    if [[ -n "${__cur}" ]]; then
        __label="${__label} [set — Enter keeps it]"
    fi
    if [[ -t 0 ]]; then
        read -r -s -p "${__label}: " __val
        echo
    else
        read -r -p "${__label}: " __val || __val=""
        echo
    fi
    if [[ -n "${__val}" ]]; then
        printf -v "${__var}" '%s' "${__val}"
    fi
}

prompt_plain() { # <varname> <label> — empty input keeps the existing value
    local __var="$1" __label="$2" __cur __val
    __cur="${!__var:-}"
    if [[ -n "${__cur}" ]]; then
        __label="${__label} [current: ${__cur} — Enter keeps it]"
    fi
    read -r -p "${__label}: " __val || __val=""
    if [[ -n "${__val}" ]]; then
        printf -v "${__var}" '%s' "${__val}"
    fi
}

echo
echo "App Store Connect API credentials"
echo "---------------------------------"
prompt_hidden ASC_KEY_ID    "ASC API key ID (no echo)"
prompt_hidden ASC_ISSUER_ID "ASC issuer ID (no echo)"

if [[ -z "${ASC_KEY_ID}" || -z "${ASC_ISSUER_ID}" ]]; then
    echo "error: key ID and issuer ID are required." >&2
    exit 1
fi

P8_PROMPT="Path to your AuthKey_*.p8 file"
if [[ -f "${KEY_FILE}" ]]; then
    P8_PROMPT="${P8_PROMPT} [already stored — Enter keeps it]"
fi
read -r -p "${P8_PROMPT}: " P8_PATH || P8_PATH=""
if [[ -n "${P8_PATH}" ]]; then
    # Allow ~ expansion without eval.
    P8_PATH="${P8_PATH/#\~/$HOME}"
    if [[ ! -f "${P8_PATH}" ]]; then
        echo "error: no file at '${P8_PATH}'." >&2
        exit 1
    fi
    if ! grep -q "PRIVATE KEY" "${P8_PATH}"; then
        echo "error: '${P8_PATH}' does not look like a .p8 private key." >&2
        exit 1
    fi
    cp "${P8_PATH}" "${KEY_FILE}"
    chmod 600 "${KEY_FILE}"
elif [[ ! -f "${KEY_FILE}" ]]; then
    echo "error: a .p8 key file is required (none stored yet)." >&2
    exit 1
fi

echo
echo "Signing identity (not secret, but per-user)"
echo "-------------------------------------------"
prompt_plain BUNDLE_ID "App bundle identifier (e.g. com.yourname.seecal)"

# Offer a detected team id as the default when the keychain has exactly one
# code-signing identity (the parenthesised suffix of the certificate name).
if [[ -z "${DEVELOPMENT_TEAM}" ]] && command -v security > /dev/null 2>&1; then
    DETECTED_TEAM="$(security find-identity -v -p codesigning 2> /dev/null \
        | grep -oE '\([A-Z0-9]{10}\)' | tr -d '()' | sort -u)"
    if [[ -n "${DETECTED_TEAM}" && "$(wc -l <<< "${DETECTED_TEAM}")" -eq 1 ]]; then
        DEVELOPMENT_TEAM="${DETECTED_TEAM}"
        echo "Detected team ID ${DEVELOPMENT_TEAM} from your signing identity."
    fi
fi
prompt_plain DEVELOPMENT_TEAM "Apple Developer team ID (10 characters)"

if [[ -z "${BUNDLE_ID}" || -z "${DEVELOPMENT_TEAM}" ]]; then
    echo "error: bundle identifier and team ID are required." >&2
    exit 1
fi

# --- 4. write --------------------------------------------------------------
umask 177
{
    echo "# Generated by scripts/release-setup.sh — never commit (gitignored)."
    printf 'ASC_KEY_ID=%q\n' "${ASC_KEY_ID}"
    printf 'ASC_ISSUER_ID=%q\n' "${ASC_ISSUER_ID}"
    printf 'ASC_KEY_PATH=%q\n' "${KEY_FILE}"
    printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
    printf 'DEVELOPMENT_TEAM=%q\n' "${DEVELOPMENT_TEAM}"
} > "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

echo
echo "Saved:"
echo "  ${ENV_FILE} (chmod 600)"
echo "  ${KEY_FILE} (chmod 600)"
echo "  bundle id: ${BUNDLE_ID}   team: ${DEVELOPMENT_TEAM}"
echo "  ASC key ID / issuer ID: stored (not shown)"
echo
echo "You can now run scripts/release-testflight.sh or scripts/release-appstore.sh."
