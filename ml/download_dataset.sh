#!/usr/bin/env bash
# download_dataset.sh — fetch the Nutrition5k dataset subset this pipeline
# needs into ml/Nutrition5K/.
#
# Source (verified 2026-07-26 against the official repo README):
#   https://github.com/google-research-datasets/Nutrition5k
#   Public GCS bucket: gs://nutrition5k_dataset/nutrition5k_dataset/
#   (also browsable/downloadable anonymously over HTTPS via
#   https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/...)
#
# The FULL dataset (imagery/side_angles/ rotating video frames included) is
# ~181GB. This pipeline (select_images.py + prepare_finetune.py) only reads:
#   metadata/dish_metadata_cafe1.csv    per-dish nutrition + ingredient rows (cafe 1)
#   metadata/dish_metadata_cafe2.csv    per-dish nutrition + ingredient rows (cafe 2)
#   metadata/ingredients_metadata.csv   ingredient nutrition-per-gram lookup table
#   imagery/realsense_overhead/         rgb.png + depth_raw.png + depth_color.png per dish (~3-4GB)
# That is the DEFAULT subset this script downloads. Pass --full to mirror the
# entire bucket instead (imagery/side_angles/, dish_ids/ splits, scripts/, ~181GB).
#
# IMPORTANT — raw metadata format: the official dish_metadata_cafe{1,2}.csv
# files are NOT simple rectangular CSVs. Each row is
#   dish_id,total_cal,total_mass,total_fat,total_carb,total_protein,
#     ingr_id,ingr_name,grams,cal,fat,carb,protein,   <- repeated once per ingredient
#     ingr_id,ingr_name,grams,cal,fat,carb,protein, ...
# i.e. a variable number of trailing columns per dish. This script saves them
# as-is under ml/Nutrition5K/metadata_raw/, then automatically runs
# convert_metadata.py to derive the tidy, fixed-schema files this repo's own
# prepare_finetune.py / select_images.py actually read
# (dish_nutrition_values.csv, dish_ingredients.csv, ingredients_metadata.csv,
# at ml/Nutrition5K/ root) — see convert_metadata.py's module docstring for
# the exact conversion rules (including why cafe2's 238 dishes are excluded
# by default, to match the historically-derived tidy files). If
# ml/Nutrition5K/ already has those tidy files (e.g. from a colleague's copy,
# or a prior manual conversion), this script's fast path recognizes them
# directly and does nothing.
#
# License / citation (print with --help):
#   Nutrition5k is released by Google Research under CC BY 4.0. Cite:
#     Thames, Q., Karpur, A., Norris, W., Xia, F., Panait, L., Weyand, T.,
#     Sim, J. "Nutrition5k: Towards Automatic Nutritional Understanding of
#     Generic Food." CVPR 2021.
#   Contact: nutrition5k@google.com. Full terms: the dataset's GitHub repo.
#
# Usage:
#   ./download_dataset.sh [--dest DIR] [--full] [--include-cafe2] [--help]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUCKET_ROOT="gs://nutrition5k_dataset/nutrition5k_dataset"
DEST="$SCRIPT_DIR/Nutrition5K"
FULL=0
INCLUDE_CAFE2=0

usage() {
  cat <<'EOF'
Usage: ./download_dataset.sh [--dest DIR] [--full] [--include-cafe2] [--help]

Downloads the Nutrition5k dataset subset this pipeline needs into
ml/Nutrition5K/ (default), or a directory you choose with --dest. After
fetching, automatically converts the raw metadata into the tidy CSVs the
pipeline reads if they're not already present (see convert_metadata.py).

  --dest DIR        Destination directory (default: ml/Nutrition5K)
  --full            Mirror the ENTIRE public bucket (~181GB: adds
                     imagery/side_angles/ rotating-camera frames, dish_ids/
                     train-test splits, scripts/). Default downloads only
                     the ~3-4GB subset this pipeline reads: the three
                     metadata CSVs plus imagery/realsense_overhead/.
  --include-cafe2   Forwarded to convert_metadata.py as --cafes all: fold
                     dish_metadata_cafe2.csv's 238 dishes (~228 with
                     realsense_overhead imagery) into the tidy CSVs too
                     (does NOT match the historically-derived tidy files/
                     published metrics, which are cafe1-only — see
                     convert_metadata.py --help).
  -h, --help        Show this help and exit.

Idempotent: if the destination already has the metadata files and
imagery/realsense_overhead/ populated, this script prints status and exits
0 without touching the network. Re-running is always safe — download is
done via `gsutil -m rsync`, which only transfers missing/changed objects,
so an interrupted run resumes cleanly on the next invocation.

Requires gsutil (part of the Google Cloud SDK). Install:
  brew install --cask google-cloud-sdk && gcloud init
or see https://cloud.google.com/sdk/docs/install

Source (verified 2026-07-26):
  https://github.com/google-research-datasets/Nutrition5k
  gs://nutrition5k_dataset/nutrition5k_dataset/

After fetching, if the tidy CSVs (dish_nutrition_values.csv,
dish_ingredients.csv, ingredients_metadata.csv) are missing from the
destination, this script automatically runs convert_metadata.py against
metadata_raw/ to produce them. By default the conversion reproduces the
historically-derived tidy files exactly: cafe1 dishes only (cafe2's 238
dishes are excluded — see convert_metadata.py's docstring). Pass
--include-cafe2 to fold cafe2 in too (a different, larger dataset).

Dataset contents:
  metadata/dish_metadata_cafe1.csv, dish_metadata_cafe2.csv
      Per-dish nutrition totals + per-ingredient breakdown, RAW/wide format
      (variable trailing columns per dish — converted automatically into
      the tidy CSVs prepare_finetune.py reads; see convert_metadata.py).
  metadata/ingredients_metadata.csv
      Ingredient -> calories/fat/carb/protein per gram lookup table.
  imagery/realsense_overhead/<dish_id>/{rgb.png,depth_raw.png,depth_color.png}
      Top-down RGB-D capture per dish (~3,472 dishes, ~3-4GB total).
  imagery/side_angles/<dish_id>/camera_{A,B,C,D}frame*.jpeg  (--full only)
      4 rotating side-angle cameras per dish, ~178GB additional.

License & citation:
  Nutrition5k is released by Google Research under the Creative Commons
  Attribution 4.0 International license (CC BY 4.0) — sharing and
  adaptation permitted with attribution. Cite:

    Thames, Q., Karpur, A., Norris, W., Xia, F., Panait, L., Weyand, T.,
    Sim, J. "Nutrition5k: Towards Automatic Nutritional Understanding of
    Generic Food." Proceedings of the IEEE/CVF Conference on Computer
    Vision and Pattern Recognition (CVPR), 2021.

  Contact: nutrition5k@google.com
  This data is NEVER committed to git (see ml/.gitignore) and is not
  redistributed by this repository — each user downloads their own copy
  under the dataset's own license terms.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --dest=*)
      DEST="${1#*=}"
      shift
      ;;
    --full)
      FULL=1
      shift
      ;;
    --include-cafe2)
      INCLUDE_CAFE2=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "download_dataset.sh: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Fast-path idempotency check
# ---------------------------------------------------------------------------

overhead_dir="$DEST/imagery/realsense_overhead"
side_dir="$DEST/imagery/side_angles"

overhead_present() {
  [ -d "$overhead_dir" ] && [ -n "$(find "$overhead_dir" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]
}

side_angles_present() {
  [ -d "$side_dir" ] && [ -n "$(find "$side_dir" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]
}

# Pipeline-ready tidy CSVs (what prepare_finetune.py actually reads)
tidy_metadata_present() {
  [ -f "$DEST/dish_nutrition_values.csv" ] &&
  [ -f "$DEST/dish_ingredients.csv" ] &&
  [ -f "$DEST/ingredients_metadata.csv" ]
}

# Raw official CSVs, as fetched by this script (idempotency marker for its
# own downloads; NOT what prepare_finetune.py reads — see header comment).
raw_metadata_present() {
  [ -f "$DEST/metadata_raw/dish_metadata_cafe1.csv" ] &&
  [ -f "$DEST/metadata_raw/dish_metadata_cafe2.csv" ] &&
  [ -f "$DEST/metadata_raw/ingredients_metadata.csv" ]
}

metadata_present() {
  tidy_metadata_present || raw_metadata_present
}

# Runs convert_metadata.py against $DEST/metadata_raw/ to (re)produce the
# tidy CSVs. Prefers ml/.venv's python (no extra deps needed — the converter
# is stdlib-only) but falls back to system python3 so this works even before
# ./setup.sh has been run.
convert_tidy_metadata() {
  local py="$SCRIPT_DIR/.venv/bin/python3"
  if [ ! -x "$py" ]; then
    py="python3"
  fi
  echo "Converting raw metadata -> tidy CSVs (convert_metadata.py) ..."
  if [ "$INCLUDE_CAFE2" -eq 1 ]; then
    "$py" "$SCRIPT_DIR/convert_metadata.py" \
      --raw-dir "$DEST/metadata_raw" --out-dir "$DEST" --cafes all
  else
    "$py" "$SCRIPT_DIR/convert_metadata.py" \
      --raw-dir "$DEST/metadata_raw" --out-dir "$DEST"
  fi
}

print_status() {
  echo "Destination: $DEST"
  if tidy_metadata_present; then
    echo "  Metadata   : pipeline-ready tidy CSVs present at $DEST/"
    printf '    dish_nutrition_values.csv : %s rows\n' "$(($(wc -l < "$DEST/dish_nutrition_values.csv") - 1))"
    printf '    dish_ingredients.csv      : %s rows\n' "$(($(wc -l < "$DEST/dish_ingredients.csv") - 1))"
    printf '    ingredients_metadata.csv  : %s rows\n' "$(($(wc -l < "$DEST/ingredients_metadata.csv") - 1))"
  elif raw_metadata_present; then
    echo "  Metadata   : raw official CSVs present at $DEST/metadata_raw/"
    echo "               (NOT yet converted to the tidy format prepare_finetune.py reads)"
  else
    echo "  Metadata   : MISSING"
  fi
  if overhead_present; then
    local n
    n="$(find "$overhead_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    echo "  Overhead   : $n dish directories, $(du -sh "$overhead_dir" 2>/dev/null | cut -f1)"
  else
    echo "  Overhead   : MISSING"
  fi
  if side_angles_present; then
    local n
    n="$(find "$side_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    echo "  Side angles: $n dish directories, $(du -sh "$side_dir" 2>/dev/null | cut -f1) (--full)"
  else
    echo "  Side angles: not downloaded (only fetched with --full)"
  fi
}

if raw_metadata_present && ! tidy_metadata_present; then
  convert_tidy_metadata
fi

if metadata_present && overhead_present && { [ "$FULL" -eq 0 ] || side_angles_present; }; then
  echo "Nutrition5k subset already present — nothing to do."
  print_status
  exit 0
fi

# ---------------------------------------------------------------------------
# Need to fetch something — require gsutil.
# ---------------------------------------------------------------------------

if ! command -v gsutil >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: gsutil not found on PATH. gsutil (part of the Google Cloud SDK) is
required to download Nutrition5k from its public GCS bucket.

Install it, then re-run this script:
  brew install --cask google-cloud-sdk
  gcloud init          # first-time only; anonymous read access to this
                        # public bucket does not require a paid GCP project,
                        # but gcloud/gsutil still need to be initialized once.

Alternative install methods: https://cloud.google.com/sdk/docs/install
EOF
  exit 1
fi

mkdir -p "$DEST"

echo "Fetching metadata CSVs into $DEST/metadata_raw/ ..."
mkdir -p "$DEST/metadata_raw"
gsutil -m cp -n \
  "$BUCKET_ROOT/metadata/dish_metadata_cafe1.csv" \
  "$BUCKET_ROOT/metadata/dish_metadata_cafe2.csv" \
  "$BUCKET_ROOT/metadata/ingredients_metadata.csv" \
  "$DEST/metadata_raw/"

echo "Fetching imagery/realsense_overhead/ into $DEST/imagery/realsense_overhead/ ..."
mkdir -p "$overhead_dir"
gsutil -m rsync -r "$BUCKET_ROOT/imagery/realsense_overhead" "$overhead_dir"

if [ "$FULL" -eq 1 ]; then
  echo "Fetching imagery/side_angles/ (this is the ~178GB part; --full) ..."
  mkdir -p "$side_dir"
  gsutil -m rsync -r "$BUCKET_ROOT/imagery/side_angles" "$side_dir"

  echo "Fetching remaining bucket contents (dish_ids/, scripts/, ...) ..."
  gsutil -m rsync -r -x '^imagery/.*' "$BUCKET_ROOT" "$DEST"
fi

echo
echo "Fetch complete."

if raw_metadata_present && ! tidy_metadata_present; then
  convert_tidy_metadata
fi

print_status

if ! tidy_metadata_present; then
  cat <<'EOF'

NOTE: tidy CSVs (dish_nutrition_values.csv / dish_ingredients.csv /
ingredients_metadata.csv) are still missing after attempting automatic
conversion — check the convert_metadata.py output above for the error. If
you already have those tidy files from elsewhere, drop them directly into
ml/Nutrition5K/ and this script will recognize them as done next time.
EOF
fi
