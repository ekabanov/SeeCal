# SCALE issues — report for teacher review

**Date:** 2026-07-30  
**Status:** SCALE is the main blocker in the factored pipeline. No new
IDENTIFY LoRA should be trained until the mass path improves.  
**Requested review:** diagnose whether the next experiment should primarily
change input geometry, model/objective, or source mixing, and propose the
smallest decisive ablation.

## Executive summary

SCALE is a MobileNetV3-Large quantile regressor that predicts total visible
food mass as P10/P50/P90. It works reasonably on the controlled Nutrition5K
camera rig, improved substantially on NutritionVerse after adding
NutritionVerse official-Train images, but fails badly on the untouched Food
Portion Benchmark (FPB) test set.

The current best checkpoint, Probe B, scores:

| Frozen evaluation | Groups / records | Mass MAE | MAPE | P10–P90 coverage |
|---|---:|---:|---:|---:|
| Nutrition5K overhead | 325 / 325 | **36.9 g** | 27.0% | 80.3% |
| Nutrition5K side views | 311 / 1,179 | **41.3 g** equal-group | 30.1% | 82.4% |
| NutritionVerse official Val | 67 / 265 | **96.2 g** equal-scene | 27.8% | 81.3% |
| FPB official test, zero-shot | 164 / 2,123 | **165.7 g** equal-group | 55.3% | 20.9% |

The FPB failure is not explained by one global scale offset:

- median prediction is 89 g while median truth is 206 g;
- 85% of FPB records are underestimated;
- average prediction rises only from 94 g for “small” groups to 104 g for
  “average” and 116 g for “big,” while corresponding mean truth rises from
  141 g to 256 g to 376 g;
- only 22 of 40 food families with complete small/average/big groups are
  ordered correctly by prediction;
- a deliberately test-leaking post-hoc multiplier still leaves approximately
  106 g group MAE.

Our current interpretation is representation/dynamic-range failure, not a
calibration problem.

## Why SCALE matters to the full system

The factored pipeline is:

`IDENTIFY → RESOLVE + SCALE → deterministic ASSEMBLE`

IDENTIFY emits food names and relative portion units. RESOLVE supplies
source-backed nutrient profiles. SCALE supplies absolute total mass. ASSEMBLE
normalizes the units to 100% and distributes SCALE mass across resolved items.

This removes arithmetic and nutrition hallucination from the language model,
but makes mass error directly multiplicative. On the 72-dish Nutrition5K
intersection that is also Tier-1-clean for the shipping v8 system:

| Assembly condition | Calorie MAE |
|---|---:|
| True mass + bucketed visible shares | **10.3 kcal** |
| Current SCALE P50 + bucketed visible shares | **56.9 kcal** |
| Shipping v8 pipeline | **29.5 kcal** |

A sensitivity sweep shows that retaining 50% of current SCALE error produces
30.3 kcal MAE, still slightly worse than v8; retaining 25% produces 18.8 kcal.
The Nutrition5K crossover is therefore roughly a 52% mass-error reduction,
approximately 36.9 g → 18 g. This 18 g target is specific to that
Nutrition5K shared-clean analysis, not a claim that the same absolute threshold
is realistic for every phone-photo domain.

## Model and training details

Current Probe B:

- MobileNetV3-Large, ImageNet initialization;
- three ordered log-mass quantiles;
- 224×224 input;
- resize short side to 232, then center crop to 224;
- whole-frame-preserving training policy: no random resized crop;
- mild rotation, horizontal flip, and color jitter;
- six epochs, selected epoch 5;
- 8,192 samples per epoch;
- source/group-balanced sampling;
- requested source weights: Nutrition5K 0.60,
  NutritionVerse official Train 0.40;
- 39,352 Nutrition5K records and 497 NutritionVerse records in the realized
  training manifest;
- checkpoint selected only on the mixed validation manifest;
- official Nutrition5K and NutritionVerse tests were excluded from selection.

The model predicts log1p mass. Quantiles are constrained to be ordered.
Intervals are calibrated after checkpoint selection using group-balanced,
source-specific conformal margins. The phone-photo margin is much larger than
the rig margin:

- Nutrition5K: +2.48 g;
- NutritionVerse: +79.26 g.

No FPB images were used in training or calibration before the zero-shot test.

## Experiments already run

### 1. In-domain mass-only retargeting

The original specialist’s implied mass achieved 35.9 g MAE on Nutrition5K and
beat Qwen-implied mass by 7.8 g. Retargeting it to mass-only quantiles and
calibrating intervals produced 34.3 g MAE with 77.2% coverage.

This established that a small specialist can work on the controlled rig.

### 2. More views from the same rig

A Nutrition5K-only multi-view probe scored:

- 44.2 g Nutrition5K overhead MAE;
- 49.2 g Nutrition5K side equal-group MAE;
- 331.8 g NutritionVerse equal-scene MAE.

Adding many correlated viewpoints from one capture system did not create
phone-photo robustness and slightly harmed the original overhead task.

### 3. NutritionVerse source-balanced training

Adding NutritionVerse official Train and source/group-balanced sampling
produced Probe B:

- Nutrition5K overhead: 36.9 g;
- Nutrition5K side: 41.3 g;
- NutritionVerse official Val: 96.2 g.

This was a large real OOD improvement over the roughly 300 g NutritionVerse
failure while mostly preserving Nutrition5K. It suggests that true source
diversity helps substantially, even with only 497 NutritionVerse training
views.

### 4. Point calibration

A five-fold, group-aware validation experiment compared identity, scale,
offset, affine, and log-affine point transforms. It selected a
NutritionVerse-specific log-affine transform in validation, but frozen-test
NutritionVerse MAE worsened from 96.2 g to 98.8 g and coverage fell from 81.3%
to 75.4%. The transform was rejected.

This is evidence against spending another iteration on point calibration
before improving the representation.

### 5. FPB zero-shot

The official FPB test partition was kept frozen. The importer found:

- 2,196 test images;
- 73 images containing an explicit `-1` unknown object weight, excluded;
- one label with no corresponding image, excluded;
- 2,123 clean-weight records across 164 food/portion groups retained.

Probe B produced 157.3 g record MAE, 165.7 g equal-group MAE, 55.3%
equal-group MAPE, and 20.9% equal-group coverage. The 95% Wilson interval for
group coverage is 15.4–27.8%.

The FPB training/validation download is currently incomplete, so an
FPB-trained probe has not yet been run.

## Suspected failure modes

These are hypotheses, not conclusions.

### A. Input normalization may erase monocular scale cues

Every image is resized by its short side and center-cropped to a fixed square.
This preserves more geometry than random resized crop, but it still normalizes
framing across sources. The same meal can occupy a similar fraction of the
224×224 tensor despite very different camera distance, plate size, or original
field of view. Center crop can also remove container boundaries that provide
the only useful size reference.

Possible ablations:

- aspect-preserving letterbox/padding instead of center crop;
- 320×320 input;
- explicitly retain original aspect ratio and image dimensions as auxiliary
  features;
- measure plate/container occupancy before choosing architecture changes.

### B. Total mass may be too weakly identifiable from one unconstrained RGB image

Nutrition5K has a fixed rig and recurring plates. A model may learn plate and
camera priors rather than food volume. FPB and NutritionVerse vary camera,
container, composition, and framing. Without a known reference object, depth,
or camera intrinsics, total grams can be intrinsically ambiguous.

The product nevertheless needs a useful estimate, so the question is whether
the model should:

- remain a direct regressor with broader training;
- predict ordinal portion/volume plus food-density-conditioned mass;
- use IDENTIFY output as a soft conditioning signal;
- use separate visual volume and semantic density components;
- predict a deliberately broad distribution and request confirmation when
  ambiguity is irreducible.

### C. The objective does not explicitly teach relative portion ordering

Probe B often fails even small < average < big within the same FPB food family.
Quantile loss penalizes absolute mass but provides no pairwise or ordinal
constraint between related examples.

A modest auxiliary objective could enforce pairwise ordering or predict a
coarse portion bin alongside log mass. The concern is whether this improves
general mass sensitivity or merely exploits FPB naming/collection structure.
Runtime inputs would remain images only.

### D. Source-balanced sampling may still optimize the wrong compromise

Source weights prevent Nutrition5K’s 39k views from completely drowning out
497 NutritionVerse views, but one shared head may still settle on a narrow
mass prior. Pooled validation MAE may also select a checkpoint that is not
Pareto-safe across domains.

Alternatives include:

- equal source sampling across Nutrition5K/NutritionVerse/FPB;
- group-balanced batches with one source per sub-batch;
- minimax or normalized per-source validation regret;
- shared backbone with small source-agnostic mixture heads;
- curriculum: diverse phone sources first, then balanced fine-tuning.

Source identity will not be available reliably at runtime, so a solution
requiring a known dataset/domain label is undesirable.

### E. Uncertainty calibration does not transfer to unseen domains

Source-specific conformal calibration is honest on known held-out sources, but
FPB coverage collapses to 20.9%. A largest-known-source fallback margin would
still not address severe point underestimation and may not cover the FPB tail.

This suggests the uncertainty model is conditional on training-domain support.
We need either a more diverse calibration mixture, an OOD-aware interval
inflation rule, or an explicit “mass unavailable / confirm” path.

## Proposed next experiment, pending teacher review

Once FPB train/validation are complete, the current conservative proposal is
one modest Probe C:

1. Keep all three official tests frozen:
   Nutrition5K test325, NutritionVerse official Val, and FPB official test.
2. Train on Nutrition5K training groups, NutritionVerse official Train, and
   FPB train/validation only.
3. Use approximately 50/25/25 effective source exposure rather than raw record
   counts.
4. Preserve group-aware sampling and ordered log quantiles.
5. Replace center crop with aspect-preserving letterbox as the only geometry
   change, unless the teacher recommends a more diagnostic first ablation.
6. Select the checkpoint using a source-stratified criterion, not pooled MAE.
7. Calibrate intervals only after point-model selection.
8. Reject the probe if it materially regresses Nutrition5K or NutritionVerse,
   even if FPB improves.

This proposal intentionally avoids a broad architecture search. However,
mixing both a new source and new preprocessing in one run weakens attribution.
We would value advice on whether the first FPB run should hold preprocessing
fixed for a clean dataset ablation, or whether the center-crop concern is
strong enough that repeating the known-bad geometry is wasteful.

## Questions for the teacher

1. Does the evidence support our conclusion that this is representation
   failure rather than calibration failure?
2. Is direct monocular total-mass regression a sound target across
   unconstrained phone images, or should SCALE be decomposed into visual
   portion/volume and semantic density?
3. For the smallest decisive next experiment, should we:
   - add FPB with preprocessing unchanged;
   - add FPB and switch to letterbox;
   - first run a preprocessing-only diagnostic on existing sources?
4. Would an auxiliary ordinal/pairwise portion loss be principled given the
   22/40 FPB size-ordering result?
5. Is MobileNetV3-Large at 224×224 likely capacity-limited here, or is the
   failure more plausibly caused by data/geometry?
6. What checkpoint-selection rule best protects three domains with very
   different absolute mass and difficulty distributions?
7. How should uncertainty behave on a novel unsupported source: wider
   intervals, OOD rejection, or both?
8. Is the Nutrition5K-specific ≈18 g crossover target a sensible optimization
   constraint, or should the pipeline gate use downstream calorie regret
   directly across all domains?

## Guardrails and reproducibility

- Shipping v8 weights and the production specialist are unchanged.
- No new IDENTIFY LoRA has been started.
- Official test sets are excluded from checkpoint selection.
- Related views are aggregated by group; coverage confidence intervals use
  independent group count.
- FPB unknown-weight rows are excluded rather than partially summed.
- FPB and NutritionVerse are confined to the non-commercial track.
- Current ML suite: 148 tests passing.

Primary artifacts:

- `ml/runs/factored/scale-v2-probe-b-nv-1024/run_config.json`
- `ml/runs/factored/scale-v2-probe-b-nv-1024/checkpoint_selection.json`
- `ml/runs/factored/scale-v2-probe-b-nv-1024/eval-nutrition5k-overhead-calibrated.json`
- `ml/runs/factored/scale-v2-probe-b-nv-1024/eval-nutritionverse-real-calibrated.json`
- `ml/runs/factored/scale-v2-probe-b-nv-1024/eval-fpb-test-zero-shot.json`
- `docs/specs/2026-07-29-factored-pipeline-design.md`
