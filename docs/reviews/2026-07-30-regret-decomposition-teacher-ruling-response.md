# Regret decomposition — execution response to teacher ruling

**Date:** 2026-07-30  
**Responding to:** `2026-07-30-regret-decomposition-teacher-ruling.md`

## Correction accepted

The true-mass floor uses oracle names and shares. It therefore does not measure
IDENTIFY error. The active IDENTIFY-v2 LoRA is reaching-the-floor work; only
the future model-versus-oracle gap is attributable to it. Floor-lowering work
belongs to RESOLVE and the visible-label/assembly policy.

NutritionVerse raw official Val is also removed from prioritization. Its
label-noise-inflated 205.2 kcal floor remains reported as a stress diagnostic,
but the NutritionVerse quality slice (41% non-mass share, hence SCALE-side)
is the decision signal. The IDENTIFY restart remains justified by clean
Nutrition5K all-complete evidence alone.

## Required Nutrition5K floor attribution

The 323 complete Nutrition5K groups were rescored with a telescoping
counterfactual waterfall. Absolute-error interactions make attribution order
dependent, so the order is frozen and recorded in the output:
visible-label/exclusion residual → rung-1/2 density → rung-3+ density →
share bucketing.

| Component | kcal MAE | Share of 33.92 floor | Decision |
|---|---:|---:|---|
| Visible-label and exclusion residual with true item densities | 20.63 | 60.80% | Structural label/assembly long tail |
| Rung-1/2 density mismatch | 11.82 | 34.84% | Prepared-variant resolver workstream |
| Rung-3+ density mismatch | 0.09 | 0.25% | Do not prioritize category expansion |
| Five-point share bucketing | 1.39 | 4.11% | Keep current buckets |
| **Total** | **33.92** | **100%** | Exact reconciliation |

Rung-3+ affected only 4 predictions. Category-default items contributed
0.262 predicted kcal per complete group on average, confirming that neither
alias coverage nor FNDDS-composite expansion is the current floor driver.

The dominant controllable resolver component is rung-1/2 density mismatch.
The diagnostic exposes exact aliases that resolve reliably but to the wrong
preparation state or food form:

| Emitted name | Current profile | Resolver density | Mean measured density | Absolute contribution |
|---|---|---:|---:|---:|
| wheat berry | Wheat, hard red spring | 351.0 kcal/100 g | 91.0 | 6.66 kcal/group |
| pork | Roasted pork tenderloin | 139.8 | 238.0 | 4.60 kcal/group |
| caesar salad | Low-calorie Caesar dressing | 115.2 | 44.0 | 3.80 kcal/group |
| fish | Raw spot fish | 118.1 | 90.0 | 2.92 kcal/group |

This defines the next resolver pass: reconsider existing exact aliases using
prepared/cooked FDC or FNDDS variants. Candidate selection and density targets
must come from Nutrition5K train/validation only; the frozen test diagnostics
may identify the failure class but must not choose the replacement profile.

The 20.63 kcal visible-label/exclusion residual is larger than the resolver
component. It is the remaining consequence of reallocating total measured mass
over visible items while total calorie truth still contains excluded
ingredients. Hidden-item recitation remains rejected by the prior OOD ruling,
so this residual must be represented as structural uncertainty or addressed
by a separately validated assembly policy—not relabelled as resolver error.

Machine-readable result:
`ml/runs/factored/oracle-v5-floor-attribution/nutrition5k-test325.json`.

## IDENTIFY accountability prepared

`ml/model_oracle_gap.py` now reports, on the exact same C1/resolver scope:

- model assembly completion coverage;
- model and oracle kcal MAE on mutually complete rows;
- model-minus-oracle kcal MAE.

The post-training evaluation path is switched from Probe B mass to selected C1
mass and will automatically emit the four required slices:

1. Nutrition5K all complete;
2. Nutrition5K shared clean 72;
3. NutritionVerse quality slice;
4. NutritionVerse raw Val, retained as diagnostic only.

The saved full NutritionVerse IDENTIFY output is filtered to the frozen quality
manifest on CPU, avoiding a duplicate 237-image VLM evaluation.

## Remaining ordered work

1. Finish the active FoodSeg IDENTIFY-v2 primary run.
2. Run frozen v3-taxonomy evaluation and the four model-versus-oracle gaps.
3. Derive prepared-variant alias candidates from train/validation only, then
   validate before touching the frozen test again.
4. Run Probe E when Metal is free.
5. Select a disagreement-gate operating point on validation data before any
   product wiring.

v8 remains unchanged and shipping; all work above remains shadow-only.
