# Factored pipeline — response to teacher feedback

**Date:** 2026-07-29  
**Feedback:** `2026-07-29-factored-pipeline-intermediate-review-feedback.md`  
**Status:** teacher corrections implemented; later empirical corrections
recorded below; SCALE diversity work in progress

## Disposition

### B1 — Populate the training-vocabulary alias table

Accepted, with the anti-leakage boundary made stricter.

The earlier 572-miss file combined the full Nutrition5K metadata vocabulary
and NutritionVerse ground-truth vocabulary. Neither is the actual training
completion vocabulary. The real train/validation completions contain 271
distinct normalized names.

`factored_pipeline/training_aliases_v1.tsv` is now a versioned, reviewable
source-profile mapping. It contains 228 mappings ordered by gram-weighted
training frequency. Together with 30 existing exact/fuzzy source matches, it
puts 258/271 names (95.2%) on rung 1. The 13 exceptions are deliberately not
forced onto unrelated USDA rows:

- `cheese butter`
- `chicken duck`
- `deprecated`
- `fried meat`
- `garden salad`
- `greek salad`
- `juice`
- `other ingredients`
- `pasta salad`
- `plate only`
- `salad`
- `sauce`
- `soup`

These are intrinsically ambiguous, generic, or non-food labels. Leaving them
estimated/unresolved preserves the pipeline's honesty instead of gaming rung
1. NutritionVerse ground truth and all evaluation-emitted names are excluded
from runtime alias promotion.

Candidate-emission resolution remains a required post-training measurement.
The ship gate is gram-weighted rung-1/2 resolution ≥85% on test325, reported
alongside NutritionVerse-Real.

### B2 — Freeze an independent evaluation taxonomy

Accepted and implemented before runtime alias promotion.

`eval_taxonomies/eval_taxonomy_v1.json` contains 13,337 frozen name mappings
and matching thresholds. `factored_pipeline/eval_taxonomy.py` owns an
independent v1 normalizer and lexical matcher. The scoring harness requires
this artifact for HMR/IDR; runtime RESOLVE is used only for resolution and
nutrition consistency.

Re-scoring the fixed E0 prediction set reproduced the published values
exactly:

| Metric | Frozen-taxonomy result |
|---|---:|
| HMR | 0.12923292170488407 |
| IDR | 0.7313183142109858 |
| IDP | 0.7130800891375189 |

A regression test mutates runtime resolver behavior while holding the
taxonomy fixed and asserts HMR does not move.

### B3 — Trainable-parameter assertion and resume

Accepted.

`train.sh` now refuses any `ADAPTER_PATH` resume. Training is launched through
`verified_lora_train.py`, which must observe 32.464896M trainable parameters
within tolerance before training begins. A missing or mismatched count
terminates the child process. Tests cover both the valid fresh-LoRA count and
the rejected 366.9M resume shape.

The first E3 arm began before this guard landed, but its captured startup log
already reports the correct 32.464896M count. It therefore remains valid.

## Experiment-design rulings absorbed

- E3 selection order is frozen in the binding design: Tier-1/refusal gates;
  paired assembled conditional kcal MAE on the mutually complete,
  Tier-1-clean dish intersection with completeness adjacent; then
  IDR/IDP/share diagnostics. `score_harness.py compare-assemblies` implements
  the group-level intersection and paired bootstrap CI; for multi-view data a
  dish is eligible only when every expected view is complete and Tier-1-clean
  in both arms.
- Same epochs remains primary. The mixed arm's approximately 2.85× step count
  is recorded as a policy/compute confound; equal-step control is conditional.
- Evaluation repairs otherwise-valid 5%-bucket shares totaling 90–110 by
  deterministic largest remainder. Repair is flagged. Gates are repair rate
  ≤5% and post-repair rejection ≤1%. Runtime remains strict until Stage 3.
- FoodSeg103 has no container labels. Its records use `container="other"`,
  and container is diagnostic-only for the mixed arm.
- NutritionVerse Tier 2 aggregates views within each of 225 dishes before
  averaging dishes. Tier-1 events use equal dish weights over pooled views;
  per-view P90 is diagnostic.
- NutritionVerse-Real was initially evaluation-only. The project owner later
  explicitly authorized its use for non-commercial training because SeeCal is
  free and open source. Only official Train may enter the separately marked
  non-commercial SCALE corpus; official Val remains frozen for evaluation.
  A permissive-only corpus remains available, and images are never bundled.

## Additional requested diagnostics

The objective 5% apportionment floor retains oil-type ingredients in 1,630 of
3,244 Nutrition5K completions:

| Label | Retained items |
|---|---:|
| olive oil | 1,558 |
| vinaigrette | 82 |
| mayonnaise | 55 |
| Caesar dressing | 46 |
| butter | 42 |
| vegetable oil | 8 |

That initial ruling was later overturned by direct evaluation. The tuned
Nutrition5K percentage arm emitted hidden condiment/recipe labels on 33.9% of
the frozen Nutrition5K slice and 62.5% of the NutritionVerse slice. A
deterministic visible-label post-filter reduced the diagnostic rate to zero,
but only moved HMR from 12.04% to 12.21% in-domain and from 33.64% to 31.30%
OOD. This shows that hidden-label policy matters, but filtering alone cannot
solve OOD identification. IDENTIFY v2 training truth now follows the visible
label policy; the raw and filtered diagnostics remain adjacent.

## Later empirical contract correction

The teacher-approved v1 repair window did not make percentage closure reliable
enough for a small VLM. The completed Nutrition5K arm had 18.8% schema failures
and 25.2% repaired outputs. The binding v2 contract therefore replaces
`share_pct` with positive integer `portion_units`; deterministic code
normalizes units to 100%. The finished percentage adapter remains a useful
compatibility baseline through a strict parser and deterministic normalizer.

On the frozen fast slices:

| Model | Set | Structural success | HMR | IDR | IDP |
|---|---|---:|---:|---:|---:|
| untuned Qwen | Nutrition5K 64 | 62/64 | 41.7% | 34.6% | 36.6% |
| tuned v1 + normalization | Nutrition5K 64 | 64/64 | 12.0% | 77.0% | 68.9% |
| untuned Qwen | NutritionVerse 63 | 63/63 | 40.2% | 14.8% | 13.5% |
| tuned v1 + normalization | NutritionVerse 63 | 63/63 | 33.6% | 0.0% | 0.0% |

The NutritionVerse exact metrics expose a vocabulary/taxonomy mismatch as
well as genuine domain error; HMR remains the more useful cross-domain signal
until the independent evaluation vocabulary is broadened without leakage.

## Remaining before the mixed arm starts

1. Finish and evaluate the already-running Nutrition5K-only arm.
2. Run focused and full unit suites with the new taxonomy, repair, alias, and
   train guard paths.
3. Regenerate the mixed manifest so its metadata records the FoodSeg
   diagnostic-only container policy.
4. Confirm the mixed-arm startup guard observes exactly 32.464896M trainable
   parameters.

The production factory remains unchanged; v8 still ships.
