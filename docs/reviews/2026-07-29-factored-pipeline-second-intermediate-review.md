# Factored nutrition pipeline — second intermediate teacher review

**Date:** 2026-07-29  
**Review state:** deterministic spine and fast baselines complete; SCALE-v2
phone-domain probe complete; FPB import pending; new IDENTIFY LoRA deliberately
not started  
**Binding design:** `docs/specs/2026-07-29-factored-pipeline-design.md`  
**Shipping state:** unchanged `v8-conditioned`; all work below remains shadow

## Review request

Please audit the revised experiment for leakage, invalid comparisons, hidden
label noise, and unjustified conclusions. In particular:

1. Is the group-aware, source-specific interval calibration statistically
   defensible for a deployment whose real inputs are consumer phone photos?
2. Is the NutritionVerse gain likely to survive broader phone-photo food
   distributions, or is it narrow adaptation to 158 official-Train scenes?
3. Should FRB equal-unit records enter IDENTIFY training, or does their weak
   portion supervision outweigh their visual-diversity value?
4. What is the smallest additional experiment required before spending hours
   on a new IDENTIFY LoRA?

Please separate blocking correctness findings from optional improvements.

## Executive result

The main SCALE failure was domain overfit, not corrupt NutritionVerse labels.
SCALE-v1 scored about 300 g MAE and 6% interval coverage on NutritionVerse
official Val despite 34.3 g MAE in-domain. Adding more views from the same
Nutrition5K cafeteria rig did not help. Training the same compact architecture
with a source-balanced mixture that included only NutritionVerse official
Train reduced official-Val equal-scene MAE to **96.2 g**, while retaining
**36.9 g** Nutrition5K overhead MAE.

The initial global calibration rule was invalid for heterogeneous cameras: it
made Nutrition5K intervals much too wide. Calibration now gives every dish
equal weight, uses the number of dishes rather than views for the finite-sample
rank, and preserves a margin for each capture domain. Held-out P10–P90 coverage
is **80.3%** on Nutrition5K overhead, **82.4%** on Nutrition5K side views, and
**81.3% equal-scene** on NutritionVerse.

IDENTIFY arithmetic closure is no longer assigned to the 4B model. The v2
model emits positive relative `portion_units`; deterministic code normalizes
them to 100%. The completed v1 percentage adapter remains a strong,
zero-training compatibility baseline after deterministic normalization, but
its NutritionVerse HMR remains high.

## Contract corrections since the first review

### IDENTIFY v2

The v1 contract required 5-point percentages summing to 100. On the completed
Nutrition5K arm it produced:

- 18.8% schema failures;
- 25.2% repaired outputs;
- 10.4% old-taxonomy HMR among structurally usable outputs.

The v2 completion is:

```json
{
  "not_food": false,
  "container": "plate",
  "items": [
    {"name": "rice", "portion_units": 3},
    {"name": "beans", "portion_units": 1}
  ]
}
```

Units are bounded positive integers and need not sum to anything. Python and
Swift merge duplicate normalized names, sort deterministically, and use
largest-remainder apportionment to produce 5-point shares totaling exactly
100. The legacy parser accepts arbitrary positive numeric percentages from the
finished v1 adapter and applies the same deterministic normalization.

Training truth now excludes hidden recipe ingredients and seasonings that
cannot be visually distinguished. A deterministic post-filter exists for old
outputs, but it is diagnostic rather than a substitute for retraining.

### Independent evaluation

Ingredient scoring performs exact matching before soft-category matching and
uses the frozen `eval_taxonomy_v2.json`:

```text
SHA-256 299ab41e9fc31d37a48d0ac576af6e335784c9f14394168812ec0bb85bebbdd7
```

The runtime USDA resolver cannot modify this report card. NutritionVerse exact
IDR/IDP currently expose an evaluation-vocabulary gap as well as model error;
HMR is the more informative OOD comparison until that independent vocabulary
is broadened without using model outputs.

## IDENTIFY fast baselines

Frozen slices use 64 Nutrition5K records (56 food + 8 negatives) and 63
NutritionVerse views / 16 scenes.

| Model | Set | Structural success | HMR | IDR | IDP | Share MAE |
|---|---|---:|---:|---:|---:|---:|
| untuned Qwen3.5-4B | Nutrition5K | 62/64 | 41.7% | 34.6% | 36.6% | 10.4 pt |
| tuned v1 percentages + normalization | Nutrition5K | 64/64 | 12.0% | 77.0% | 68.9% | 5.8 pt |
| untuned Qwen3.5-4B | NutritionVerse | 63/63 | 40.2% | 14.8% | 13.5% | — |
| tuned v1 percentages + normalization | NutritionVerse | 63/63 | 33.6% | 0.0% | 0.0% | — |

The tuned adapter refused all 8/8 negatives without food over-refusal.
Its hidden-condiment emission rate was 33.9% on Nutrition5K and 62.5% on
NutritionVerse. Filtering reduced the diagnostic to zero but barely changed
HMR: 12.04% → 12.21% in-domain and 33.64% → 31.30% OOD. This supports
retraining on visible truth rather than relying on filtering.

## SCALE results

All comparisons use measured total mass. NutritionVerse uses the authors'
official scene split; official Val is never used for training or model
selection.

| Model / probe | N5K overhead MAE | N5K side MAE | NV official-Val equal-scene MAE | NV coverage |
|---|---:|---:|---:|---:|
| SCALE-v1, N5K-only | 34.3 g | — | 300.0 g | 6.0% |
| Probe A, more N5K groups/views only | 44.2 g | 49.2 g | 331.8 g | 6.8% |
| Probe B, N5K + NV official Train | **36.9 g** | **41.3 g** | **96.2 g** | **81.3%** calibrated |

Probe B configuration:

- MobileNetV3-Large, ordered P10/P50/P90 heads;
- ImageNet initialization; full-backbone learning-rate multiplier 1.0;
- 8,192 samples/epoch, six epochs;
- source/group-balanced sampling: 60% Nutrition5K, 40% NutritionVerse;
- 39,849 physical train records / 4,419 groups:
  39,352 Nutrition5K views / 4,293 groups and
  497 NutritionVerse views / 126 scenes;
- calibration: 885 Nutrition5K views / 325 groups and
  127 NutritionVerse views / 32 scenes;
- frozen test: 325 Nutrition5K overheads plus
  265 NutritionVerse views / 67 scenes.

The best checkpoint was epoch 5. Point-estimate results:

| Held-out set | Records / groups | MAE | Equal-group MAE | MAPE | Median AE |
|---|---:|---:|---:|---:|---:|
| N5K overhead | 325 / 325 | 36.9 g | 36.9 g | 27.0% | 24.6 g |
| N5K side views | 1,179 / 311 | 42.2 g | 41.3 g | 30.6% | 26.9 g |
| NutritionVerse official Val | 265 / 67 | 96.4 g | 96.2 g | 27.7% | 66.7 g |

### Interval calibration correction

Raw interval coverage was 78.5% on N5K overhead but only 33.2% on
NutritionVerse. The first `max-source` calibration used one 63.4 g margin for
every source, producing 97.8% N5K coverage while still reaching only 74.3% on
NutritionVerse.

The corrected method:

1. gives each scene total weight one, independent of view count;
2. computes a separate conformity distribution for each capture source;
3. applies the finite-sample rank using independent group count
   (325 N5K dishes, 32 NutritionVerse calibration scenes);
4. retains the maximum margin as the safe fallback for an unknown source.

It selected 2.48 g for the controlled N5K rig and 79.26 g for phone photos.

| Held-out set | Calibrated coverage |
|---|---:|
| N5K overhead | 80.3% |
| N5K side views | 82.4% equal-group |
| NutritionVerse official Val | 81.3% equal-scene |

The production input is a consumer phone photo, so the phone margin is the
safe runtime default. Source-specific reporting prevents a consumer-phone
margin from making the controlled-rig benchmark look artificially honest.

## Diverse-data inventory

| Source | Role | Current state |
|---|---|---|
| Nutrition5K | measured SCALE + IDENTIFY truth | all usable cafe1/cafe2 groups and sampled A–D side frames represented in SCALE-v2 |
| NutritionVerse-Real v2 | measured phone-domain SCALE | official Train only in non-commercial training; official Val frozen |
| Food Portion Benchmark (FPB) | measured smartphone/RealSense SCALE | converter implemented and tested; user is completing the pinned download; not present in Probe B |
| FoodSeg103 | diverse IDENTIFY, mask-area weak portions | 4,983 train / 2,135 validation |
| Food Recognition Benchmark 2022 v2.0 (FRB) | diverse IDENTIFY names | existing user-provided archive reused; 3,987 train / 443 validation teacher-visible records prepared |
| COCO negatives | refusal | 100 train / 29 held-out |

FRB and FPB are different datasets. FRB has food annotations but no measured
mass; it cannot train SCALE. The new FRB IDENTIFY arm assigns one relative unit
to each teacher-accepted visible item, explicitly teaching recognition and
schema rather than portion ratios. Its combined corpus has 11,664 train /
2,930 validation / 354 test records, all schema-valid.

Adding FRB expands the training vocabulary to 795 normalized names. Only
277/795 currently resolve at USDA ladder rungs 1–2 with the existing reviewed
alias table. This is a blocking honesty issue for using the FRB arm in the
assembled system, even if identification improves.

## Licensing boundary

The project owner explicitly confirmed that SeeCal will remain free,
non-commercial, and open source. NutritionVerse (CC BY-NC-SA 4.0) and FPB
(CC BY-NC 4.0) may therefore enter a separately marked non-commercial training
track. A permissive-only SCALE manifest remains reproducible. Source datasets
are gitignored and never bundled. Public release of NutritionVerse-trained
weights still requires attribution, modification notice, compatible terms,
and a specific license review; the repository does not claim a general legal
conclusion about trained-weight adaptation.

## Verification and artifact integrity

- SCALE training writes arguments, manifest paths and SHA-256 values, source
  counts, model configuration, and calibration record counts before training.
- NutritionVerse 12 MP images are cached at a 1,024-pixel maximum edge for
  iteration speed; image content, scene grouping, target mass, and split are
  unchanged.
- Prompt parity passes for the IDENTIFY-v2 prompt.
- Python full suite: 137 passed. Tests cover the new cache, FRB split, source
  balancing, ordered quantiles, and group-aware calibration.
- Latest Swift result: 230 tests passed, 3 environment-gated skips.
- The production factory, shipping adapter, and bundled specialist are
  unchanged.

## Remaining work before any new LoRA

1. Finish the user-provided FPB download, convert it, and audit image/label
   counts, mass distribution, and group overlap.
2. Run one matched SCALE probe with FPB included. Keep Probe B unchanged as
   the comparison; reject FPB if it harms either N5K or NutritionVerse.
3. Expand or constrain RESOLVE coverage for the FRB vocabulary without using
   evaluation outputs.
4. Re-run full Python, Swift, prompt-parity, and data-integrity gates.
5. Only then run a tiny IDENTIFY-v2 overfit and short untuned/tuned comparison.
   A full LoRA remains the final expensive step.

## Requested teacher decisions

1. **Calibration:** accept the per-source, equal-group finite-rank method, or
   require a single image-conditional calibration model?
2. **FRB:** include equal-unit FRB records in the primary IDENTIFY arm, use
   them only for a matched ablation, or restrict FRB to single-item images?
3. **Resolution:** is training-vocabulary rung-1/2 coverage a pre-LoRA gate,
   or may it be repaired after observing candidate emissions, provided the
   independent evaluation taxonomy remains frozen?
4. **FPB:** should the first probe use a fixed 50/20/30
   N5K/NutritionVerse/FPB source mix, or a smaller FPB dose to preserve the
   now-demonstrated phone-domain gain?
