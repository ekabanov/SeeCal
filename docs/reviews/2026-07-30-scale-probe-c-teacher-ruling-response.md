# SCALE Probe C — execution response to teacher ruling

**Date:** 2026-07-30  
**Responding to:** `2026-07-30-scale-probe-c-teacher-ruling.md`

## Decision

The blocking regret decomposition is complete. The pre-agreed half-total rule
trips on Nutrition5K all-complete groups and NutritionVerse raw validation, so
IDENTIFY-v2 LoRA work resumes. This does not claim SCALE is solved: SCALE still
dominates the cleaner N5K shared-72 and NutritionVerse quality slices. The two
branches now have distinct measured failure surfaces.

## Required regret decomposition

The decomposition uses the same eligible rows within each slice:

`C1 calorie MAE = true-mass floor + mass-attributable excess`

| Slice | C1 total MAE | True-mass floor | Mass-attributable excess | Floor / total | Half-total ruling |
|---|---:|---:|---:|---:|---|
| Nutrition5K all complete groups | 63.41 kcal | 33.92 kcal | 29.49 kcal | 53.49% | IDENTIFY / RESOLVE |
| Nutrition5K shared clean 72 | 38.50 kcal | 10.29 kcal | 28.22 kcal | 26.72% | SCALE |
| NutritionVerse quality slice | 191.15 kcal | 78.37 kcal | 112.79 kcal | 41.00% | SCALE |
| NutritionVerse raw official Val | 343.93 kcal | 205.19 kcal | 138.74 kcal | 59.66% | IDENTIFY / RESOLVE |

This explains the apparent anomaly. C1 helps substantially where names,
resolution, and shares are already clean, but broad-slice calorie error has a
large floor that perfect mass cannot remove.

The machine-readable values are recorded in:

- `ml/runs/factored/oracle-v4-fpb-c1/nutrition5k-test325.json`
- `ml/runs/factored/oracle-v4-fpb-c1/nutritionverse-official-val-quality-v1.json`
- `ml/runs/factored/oracle-v4-fpb-c1/nutritionverse-official-val.json`

## IDENTIFY restart sequence

The previously completed 32-record memorization result belongs to the retired
v1 percentage contract. It cannot validate the current v2 contract, which
emits positive integer `portion_units` and normalizes them deterministically.
The resumed sequence is therefore:

1. fresh 32-record v2 memorization gate;
2. FoodSeg-mixed v2 primary arm;
3. matched FoodSeg + BF2 fully-resolved ablation only after the primary arm;
4. frozen Nutrition5K and NutritionVerse evaluation with the v3 taxonomy.

The first tiny run stopped before any weight update because uncapped FoodSeg
images exceeded the trainer's 2,048-token ceiling. The primary and probe
manifests were rebuilt with the same 1,024-pixel maximum already used by BF2;
all 32 probe records then passed image/token/masking smoke coverage. Prompt
parity passes for both the tiny v2 probe and the 9,582-record BF2 corpus. The
BF2 vocabulary gate remains
leakage-safe: 4,122 / 4,122 retained FRB item occurrences resolve at rungs 1–2.

The corrected 20-epoch memorization run completed at loss 0.000125. Generation,
not loss, cleared the gate: 32 / 32 outputs were parse- and schema-valid without
repair, and all 32 exactly reproduced the item-name set, container, and
`portion_units` (name recall/precision 1.0, unit MAE 0). Distributed smoke
checks then passed at eight positions across the 7,677-record capped primary
corpus. The full two-epoch FoodSeg primary run is active at
`ml/runs/factored/e3-v2-foodseg-1024/adapter`.

## Other rulings applied

- C1 remains selected; C2 is rejected and Probe D remains parked.
- The v8-on-FPB evaluation is closed without running.
- The Swift disagreement comparator now uses `sum_item_portions`, retaining
  portion data as gate-only independent evidence.
- USDA portion ingestion selects one observed serving kind rather than taking
  a median across unlike kinds. The 112 g slice / 897 g whole-pizza case now
  selects the better-supported observed row instead of inventing 504.5 g.
- A separately rebuilt database retained all 13,589 profiles, 196 category
  defaults, and 100% reviewed-alias coverage. On C1, replacing the legacy
  comparator with `sum_item_portions` changes eligible-record gate fire rates
  from 36.68% to 29.15% on Nutrition5K, 16.60% to 4.15% on NutritionVerse,
  and 11.64% to 11.19% on FPB. The remaining high N5K rate confirms that this
  is a deliberately conservative confirmation signal, not a mass estimator.
- Probe E remains the next SCALE data-only arm: NutritionVerse-Synth at
  approximately 10–15% effective exposure, frozen real-domain tests, and the
  existing minimax-MAPE / Pareto rejection rules.

## Shipping state

Nothing in this work changes production selection. C1, IDENTIFY-v2, the
portion-prior gate, and Probe E remain shadow experiments; v8 remains the
shipping and rollback system until the factored pipeline clears its product
and device gates.
