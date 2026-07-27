#!/usr/bin/env bash
# download_negatives.sh — fetch the COCO val2017 *annotations* (not images) into
# ml/coco/, then run make_negatives.py `select` to filter out every food image,
# sample a balanced non-food set, and download just those ~300 images into
# ml/negatives/.
#
# These are the v7 "not-food" negative examples: real non-food photos paired
# with a `{"not_food": true}` refusal completion, so the v7 LoRA stops
# hallucinating calories for a computer mouse / empty plate. See
# docs/plans/2026-07-27-v7-notfood-plan.md.
#
# Source (COCO 2017, permissive terms; used for training, NOT redistributed —
# ml/coco/ and ml/negatives/ are gitignored, like Nutrition5K):
#   Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#   Images:      http://images.cocodataset.org/val2017/<file_name>  (fetched per-URL)
#
# Only the annotations zip (~241MB) is downloaded up front; then ~300 images
# (~50MB) are fetched individually by make_negatives.py. The full 1GB val2017
# image zip is never downloaded.
#
# Citation: Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
#
# Usage:
#   ./download_negatives.sh [--count N] [--hard-frac F] [--seed S] [--help]
# Flags are forwarded to `make_negatives.py select`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
  esac
done

COCO_DIR="$SCRIPT_DIR/coco"
ANN_JSON="$COCO_DIR/annotations/instances_val2017.json"
ANN_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

PY="$SCRIPT_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ ! -f "$ANN_JSON" ]; then
  echo "==> Fetching COCO val2017 annotations (~241MB) ..."
  mkdir -p "$COCO_DIR"
  tmp_zip="$COCO_DIR/annotations_trainval2017.zip"
  # -C: resume a partial download if interrupted.
  curl -L -C - -o "$tmp_zip" "$ANN_ZIP_URL"
  echo "==> Extracting instances_val2017.json ..."
  # Only need the instances file; extract just that one member.
  (cd "$COCO_DIR" && unzip -o "$tmp_zip" "annotations/instances_val2017.json")
  rm -f "$tmp_zip"
else
  echo "==> COCO annotations already present ($ANN_JSON)"
fi

echo "==> Selecting + downloading negatives (make_negatives.py select) ..."
"$PY" "$SCRIPT_DIR/make_negatives.py" select "$@"

echo
echo "Done. Spot-check ml/negatives/ (confirm no real food slipped through the"
echo "food-category filter), then build the refusal JSONL:"
echo "  $PY make_negatives.py jsonl"
