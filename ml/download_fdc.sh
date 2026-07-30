#!/bin/bash
# Download the exact public USDA CSV releases used by the factored pipeline.
# Raw archives/extractions live under gitignored ml/datasets/fdc/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ROOT}/datasets/fdc"
mkdir -p "${DEST}/archives"

download_and_extract() {
    local name="$1"
    local url="$2"
    local archive="${DEST}/archives/${name}.zip"
    local extracted="${DEST}/${name}"
    if [[ ! -f "${archive}" ]]; then
        curl --fail --location --retry 3 --output "${archive}.part" "${url}"
        mv "${archive}.part" "${archive}"
    fi
    if [[ ! -d "${extracted}" ]]; then
        mkdir -p "${extracted}"
        unzip -q "${archive}" -d "${extracted}"
    fi
}

download_and_extract \
    foundation-2026-04 \
    https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2026-04-30.zip
download_and_extract \
    sr-legacy-2018-04 \
    https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip
download_and_extract \
    fndds-2021-2023 \
    https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip

"${ROOT}/.venv/bin/python" "${ROOT}/make_fdc_db.py" \
    --source "${DEST}/foundation-2026-04" \
    --source "${DEST}/sr-legacy-2018-04" \
    --source "${DEST}/fndds-2021-2023" \
    --output "${DEST}/seecal-nutrition.sqlite"

echo "Built ${DEST}/seecal-nutrition.sqlite"
