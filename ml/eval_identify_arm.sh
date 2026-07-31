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
scale_run="runs/factored/scale-v2-probe-c1-fpb-center"
oracle_n5k="runs/factored/oracle-v5-floor-attribution/nutrition5k-test325.json"
oracle_nv="runs/factored/oracle-v4-fpb-c1/nutritionverse-official-val.json"
oracle_nv_quality="runs/factored/oracle-v4-fpb-c1/nutritionverse-official-val-quality-v1.json"

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
for oracle in "${oracle_n5k}" "${oracle_nv}" "${oracle_nv_quality}"; do
  if [[ ! -f "${oracle}" ]]; then
    echo "Missing oracle audit: ${oracle}" >&2
    exit 2
  fi
done
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

.venv/bin/python filter_identify_eval.py \
  --input "${output}/identify-nutritionverse-real.json" \
  --manifest datasets/nutritionverse-real/eval-official-val-quality-v1.jsonl \
  --output "${output}/identify-nutritionverse-real-quality-v1.json"

.venv/bin/python score_harness.py audit \
  --eval "${output}/identify-nutritionverse-real-quality-v1.json" \
  --database "${database}" \
  --taxonomy "${taxonomy}" \
  --output "${output}/audit-nutritionverse-real-quality-v1.json"

.venv/bin/python score_harness.py assemble \
  --identify-eval "${output}/identify-nutritionverse-real-quality-v1.json" \
  --scale-predictions "${scale_run}/eval-nutritionverse-real-calibrated.json" \
  --database "${database}" \
  --taxonomy "${taxonomy}" \
  --output "${output}/assembly-nutritionverse-real-quality-v1.json"

.venv/bin/python model_oracle_gap.py \
  --model-assembly "${output}/assembly-test325.json" \
  --oracle-audit "${oracle_n5k}" \
  --scope all_complete \
  --output "${output}/model-oracle-gap-nutrition5k-all.json"

.venv/bin/python model_oracle_gap.py \
  --model-assembly "${output}/assembly-test325.json" \
  --oracle-audit "${oracle_n5k}" \
  --scope baseline_shared_clean \
  --output "${output}/model-oracle-gap-nutrition5k-clean72.json"

.venv/bin/python model_oracle_gap.py \
  --model-assembly "${output}/assembly-nutritionverse-real.json" \
  --oracle-audit "${oracle_nv}" \
  --scope all_complete \
  --output "${output}/model-oracle-gap-nutritionverse-raw.json"

.venv/bin/python model_oracle_gap.py \
  --model-assembly "${output}/assembly-nutritionverse-real-quality-v1.json" \
  --oracle-audit "${oracle_nv_quality}" \
  --scope all_complete \
  --output "${output}/model-oracle-gap-nutritionverse-quality-v1.json"
