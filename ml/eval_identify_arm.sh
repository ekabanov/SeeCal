#!/bin/bash
# Evaluate one E3 IDENTIFY arm sequentially on test325 and NutritionVerse-Real.
# Run from ml/ or invoke by path. MLX evaluation is intentionally single-tenant.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: ./eval_identify_arm.sh <arm-name> <adapter-path> [portion-units|legacy-shares]" >&2
  echo "Example: ./eval_identify_arm.sh e3-n5k-only runs/factored/e3-n5k-only/adapter portion-units" >&2
  exit 2
fi

arm="$1"
adapter="$2"
contract="${3:-portion-units}"
output="runs/factored/${arm}"
database="datasets/fdc/seecal-nutrition.sqlite"
taxonomy="eval_taxonomies/eval_taxonomy_v3.json"
scale_run="runs/factored/scale-v2-probe-b-nv-1024"

if [[ ! -f "${adapter}/adapters.safetensors" ]]; then
  echo "Missing adapter: ${adapter}/adapters.safetensors" >&2
  exit 2
fi
if [[ ! -f "${taxonomy}" ]]; then
  echo "Missing frozen evaluation taxonomy: ${taxonomy}" >&2
  exit 2
fi
if [[ ! -f "${scale_run}/eval-nutrition5k-overhead-calibrated.json" ]]; then
  echo "Missing SCALE test325 predictions" >&2
  exit 2
fi
if [[ ! -f "${scale_run}/eval-nutritionverse-real-calibrated.json" ]]; then
  echo "Missing SCALE NutritionVerse-Real predictions" >&2
  exit 2
fi
if [[ ! -f "datasets/nutritionverse-real/eval-official-val-v2.jsonl" ]]; then
  echo "Missing frozen NutritionVerse official-Val IDENTIFY manifest" >&2
  exit 2
fi

mkdir -p "${output}"

.venv/bin/python identify_infer.py \
  --test-set finetune_data_id_v2/test.jsonl \
  --limit 354 \
  --adapter-path "${adapter}" \
  --contract "${contract}" \
  --output "${output}/identify-test354.json"

.venv/bin/python score_harness.py audit \
  --eval "${output}/identify-test354.json" \
  --database "${database}" \
  --taxonomy "${taxonomy}" \
  --output "${output}/audit-test325.json"

.venv/bin/python score_harness.py resolve \
  --eval "${output}/identify-test354.json" \
  --database "${database}" \
  --output "${output}/resolution-test325.json"

.venv/bin/python score_harness.py assemble \
  --identify-eval "${output}/identify-test354.json" \
  --scale-predictions "${scale_run}/eval-nutrition5k-overhead-calibrated.json" \
  --database "${database}" \
  --taxonomy "${taxonomy}" \
  --output "${output}/assembly-test325.json"

.venv/bin/python identify_infer.py \
  --test-set datasets/nutritionverse-real/eval-official-val-v2.jsonl \
  --limit 265 \
  --adapter-path "${adapter}" \
  --contract "${contract}" \
  --output "${output}/identify-nutritionverse-real.json"

.venv/bin/python score_harness.py audit \
  --eval "${output}/identify-nutritionverse-real.json" \
  --database "${database}" \
  --taxonomy "${taxonomy}" \
  --output "${output}/audit-nutritionverse-real.json"

.venv/bin/python score_harness.py resolve \
  --eval "${output}/identify-nutritionverse-real.json" \
  --database "${database}" \
  --output "${output}/resolution-nutritionverse-real.json"

.venv/bin/python score_harness.py assemble \
  --identify-eval "${output}/identify-nutritionverse-real.json" \
  --scale-predictions "${scale_run}/eval-nutritionverse-real-calibrated.json" \
  --database "${database}" \
  --taxonomy "${taxonomy}" \
  --output "${output}/assembly-nutritionverse-real.json"
