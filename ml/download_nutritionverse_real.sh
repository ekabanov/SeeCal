#!/bin/bash
# Download the public NutritionVerse-Real v2 OOD evaluation corpus.
# The Kaggle dataset is CC BY-NC-SA 4.0 and intentionally gitignored.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ROOT}/datasets/nutritionverse-real"
ARCHIVE="${DEST}/nutritionverse-real-v2.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/nutritionverse/nutritionverse-real?datasetVersionNumber=2"

mkdir -p "${DEST}"
if [[ ! -f "${ARCHIVE}" ]]; then
    curl --fail --location --retry 3 --output "${ARCHIVE}.part" "${URL}"
    mv "${ARCHIVE}.part" "${ARCHIVE}"
fi

unzip -tq "${ARCHIVE}"
if [[ ! -d "${DEST}/extracted" ]]; then
    mkdir -p "${DEST}/extracted"
    unzip -q "${ARCHIVE}" -d "${DEST}/extracted"
fi

(
    cd "${DEST}"
    shasum -a 256 "$(basename "${ARCHIVE}")" > archive.sha256
)

echo "Downloaded NutritionVerse-Real v2 to ${DEST}"
