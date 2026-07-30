# Factored nutrition pipeline — intermediate teacher review

**Date:** 2026-07-29  
**Review state:** Stages 0–1 implemented; Stage 2 E3 training in progress;
shipping v8 path unchanged  
**Binding design:** `docs/specs/2026-07-29-factored-pipeline-design.md`  
**Shipping architecture:** `docs/architecture.md` until every Stage-3 gate passes

## Review request

Please review the implementation and experimental design for correctness,
hidden leakage, metric validity, and likely failure modes. Prioritize issues
that could invalidate E3 or make a factored candidate appear safer than it is.
Separate:

1. blocking correctness problems;
2. experiment-design changes required before the second E3 arm;
3. improvements that can wait until after E3;
4. disagreements with the binding design.

For every blocking finding, name the violated invariant and propose the
smallest falsifiable check or code change.

## Executive summary

The migration replaces one model that emits names, grams, calories, and macros
with four components:

1. **IDENTIFY:** Qwen3.5-4B + LoRA emits only food names, coarse 5% shares, a
   closed container vocabulary, and `not_food`.
2. **SCALE:** a mass-only MobileNet quantile regressor emits P10/P50/P90 total
   mass.
3. **RESOLVE:** a read-only SQLite ladder maps each emitted name to a
   source-backed USDA nutrition profile, an explicitly estimated category
   median, or visible unresolved state.
4. **ASSEMBLE:** Swift performs deterministic interval arithmetic. Calories
   are always `4p + 9f + 4c`; unresolved items never become fabricated zeros.

Stages 0–1 are implemented in Python and Swift, with current v8 available as a
shadow input. Nothing selects the factored path in `ProductionFactory`.
SCALE-v1 passes its held-out interval gate. A 32-record IDENTIFY memorization
gate is perfect after uninterrupted training. The first full E3 arm
(Nutrition5K-only) is currently running; the FoodSeg103-mixed arm has not yet
started.

## Frozen IDENTIFY contract

The exact prompt, shared by data preparation, parity checking, inference, and
Swift, is:

> Identify the visible food without estimating grams, calories, or nutrients.
> Return exactly one JSON object with keys not_food, container, and items.
> container must be plate, bowl, cup, tray, packaging, or other. items must
> contain objects with name and share_pct only, sorted by share_pct descending.
> share_pct values must be multiples of 5 and sum to 100. For a non-food image
> return not_food true, container other, and an empty items list.

Food completion:

```json
{
  "not_food": false,
  "container": "plate|bowl|cup|tray|packaging|other",
  "items": [
    {"name": "food name", "share_pct": 65},
    {"name": "food name", "share_pct": 35}
  ]
}
```

Non-food completion:

```json
{"not_food": true, "container": "other", "items": []}
```

The parser rejects missing/extra keys, duplicate normalized names, invalid
containers, non-positive/non-5% shares, unsorted shares, and sums other than
100. At most 20 positive items can therefore exist.

## Implemented surface

### Python

- `ml/factored_pipeline/contract.py`: frozen prompt, strict schema, canonical
  names, largest-remainder 5% share apportionment.
- `ml/make_identify_data.py`: deterministic, interleaved Nutrition5K +
  FoodSeg103 + negative manifests. Training completions remain minimal while
  top-level evaluation ground truth preserves exact measured grams/nutrition.
- `ml/download_foodseg103.py`: pinned mirror and four enforced source hashes.
- `ml/download_nutritionverse_real.sh` and
  `ml/make_nutritionverse_eval.py`: pinned OOD corpus plus IDENTIFY, monolith,
  and SCALE evaluation manifests with explicit dish grouping.
- `ml/make_fdc_db.py` and `ml/make_alias_table.py`: pruned USDA database,
  aliases, category defaults, and a reviewable miss list.
- `ml/factored_pipeline/resolver.py`: exact → conservative fuzzy → estimated
  category cluster → hypothesis interface → unresolved.
- `ml/factored_pipeline/scoring.py` and `ml/score_harness.py`: HMR, DVR, AIR,
  IDR/IDP, share, mass, resolution, per-dish grouping, and deterministic
  IDENTIFY + SCALE + RESOLVE assembly evaluation.
- `ml/visual_specialist/scale.py`: mass-only P10/P50/P90 model, conformal
  calibration, evaluation, and Core ML export.
- `ml/identify_infer.py`: strict new-schema inference with parse, schema, and
  refusal accounting.

### Swift

- `FactoredNutritionPipeline.swift`: strict parser, immutable SQLite resolver,
  deterministic interval-aware assembly, visible unresolved state, confidence
  and scale-disagreement confirmation reasons, and current-v8 shadow adapter.
- `FactoredScaleModel.swift`: mass-only Core ML preprocessing/runtime and
  concurrent IDENTIFY/SCALE pipeline shape.
- `FactoredNutritionPipelineTests.swift`: prompt, parser, arithmetic,
  unresolved-state, resolver, shadow, real SQLite, and real Core ML tests.
- `Package.swift`: `SeeCalInference` links the system SQLite library.

## Data and provenance

| Source | Local use | Exact state |
|---|---|---|
| Nutrition5K | IDENTIFY measured gram shares; SCALE mass | Existing cafe1-derived split retained deliberately for exact v8/test325 pairing. Cafe2 is not silently added |
| FoodSeg103 | IDENTIFY OOD diversity; mask-area weak shares | 4,983 train + 2,135 validation; Apache-2.0 mirror pinned to `34e1208e14bc3595d544fc8c3f3c6673253fd9ef`; all four LFS SHA-256 values verified |
| COCO negatives | Refusal | Proven dose: 100 train / 29 held-out |
| NutritionVerse-Real v2 | OOD evaluation only | CC BY-NC-SA 4.0; 889 images, 225 represented dish IDs; archive SHA-256 `22910bc84cb5840cb4a66be20d71ce2b7ba51f1d1a7f62108debf78fe7180780` |
| USDA FDC | Runtime RESOLVE | Foundation 2026-04, SR Legacy 2018-04, FNDDS 2021–2023; 8,159 complete profiles, 25 category defaults |

The exact mixed IDENTIFY corpus is 7,677 train / 2,487 valid / 354 test.
The N5K-only E3 arm is 2,694 / 352 / 354. Both include the same refusal dose
and are deterministically shuffled. All 10,518 mixed records pass the frozen
schema validator. Prompt byte parity passes 8/8 on both arms.

The OOD archive contains metadata for 251 dishes but image files for 225 dish
IDs; metrics therefore report both 889 predictions and 225 groups, averaging
within dish before averaging across dishes.

## Measured results so far

### E0 — current v8 hard-mistake baseline on paired test325

| Metric | Result |
|---|---:|
| HMR | 12.9% |
| DVR | 40.5% |
| AIR | 31.6% |
| IDR | 73.1% |
| IDP | 71.3% |
| Tier-1-clean dishes | 39 / 325 |
| Conditional kcal MAE on those dishes | 30.4 kcal |

This baseline was recomputed after fixing three harness issues:

- zero-calorie/non-zero-macro items now count as AIR violations;
- identical normalized names match exactly before category logic;
- category selection uses a small candidate cluster rather than one odd USDA
  row.

### E1 — v8 name resolution

| Rung | Items | Rate |
|---|---:|---:|
| Exact alias | 343 | 16.5% |
| Fuzzy | 22 | 1.1% |
| Estimated category default | 1,707 | 82.1% |
| Unresolved | 7 | 0.3% |

Rungs 1–2 therefore resolve only 17.6% of emitted names. The alias builder
review file currently contains 572 unpromoted source-vocabulary names. No v8
evaluation-emitted name is promoted into rung 1.

### E2 and SCALE-v1

| Model | Mass MAE | MAPE | P10–P90 coverage |
|---|---:|---:|---:|
| Qwen-implied mass | 43.7 g | — | — |
| Existing S1 | 35.9 g | 23.6% | 74.2% |
| SCALE-v1 | 34.3 g | 22.4% | 77.2% |

SCALE-v1 is authoritative in the candidate architecture. Its 77.2% held-out
coverage passes the 75–85% gate. The exported Core ML model has been exercised
over all 325 images from Swift; final physical-iPhone latency/memory/thermal
checks remain outstanding.

### IDENTIFY memorization gate

The uninterrupted 20-epoch 32-record run ended at loss 0.000082:

| Metric | Result |
|---|---:|
| Parse failures | 0 / 32 |
| Schema failures | 0 / 32 |
| HMR | 0 |
| IDR / IDP | 1.0 / 1.0 |
| Share MAE | 0 |

A shorter 10-epoch run produced 25/32 schema-valid generations; all seven
failures were valid JSON with shares totaling 85–115. Continuous training,
not parser repair or post-hoc renormalization, closed those failures.

### Current live E3 status

Nutrition5K-only training uses the same base, rank/alpha, learning rate,
two-epoch schedule, completion-only loss, and frozen prompt as the mixed arm.
At the time this report was written it was at iteration 800 / 5,388, loss
0.171, with no failure and approximately 20.4 GB peak Metal memory. No other
GPU-heavy work is co-tenanted.

## Verification state

- Latest complete Python suite before the most recent grouped-assembly helper:
  109 passed.
- Focused tests for grouped manifests and factored assembly pass after that
  addition.
- Real FoodSeg prompt-parity and mlx-vlm image/mask smoke gates pass.
- Real SQLite Swift resolver test passes after the category-default fix.
- All 8,159 food rows and all 25 category defaults are exactly
  Atwater-consistent.
- Full Swift suite, iOS build check, and real Core ML test will be rerun after
  GPU training to avoid co-tenancy.

The current v8 production factory and bundled shipping artifacts are unchanged.

## Findings already corrected during execution

1. **Source arms were concatenated, not interleaved.** The builder now
   deterministically shuffles every split.
2. **Evaluation retained only coarse shares.** New manifests preserve exact
   measured grams/nutrition outside the training completion, restoring truly
   gram-weighted HMR.
3. **OOD images shared a non-unique inferred ID.** Manifests now carry explicit
   image and dish IDs.
4. **Category-default calories were independent medians.** Calories now derive
   from the median macros, making AIR zero by construction for all DB profiles.
5. **Identical names could be downgraded to a soft category match.** Exact
   normalized-name equality now wins before resolver-category logic.
6. **One lexical neighbor selected the category default.** The resolver now
   uses the strongest five-row category cluster and marks the result estimated.
7. **Nested training output paths could break logging.** `train.sh` now creates
   the output parent before starting the pipe.

## Known risks and requested feedback

### A. “Visible food” versus Nutrition5K ingredient labels

Nutrition5K completions include measured ingredients such as salt, oil,
vinegar, and spices that may not be visually separable. This follows the design
source-label rule but conflicts literally with “visible food.” The perfect
memorization result does not show that these labels are learnable out of
sample.

**Question:** Should v1 retain all measured ingredients for comparability, or
filter/merge visually latent condiments before the mixed arm? If filtering,
what objective rule avoids manual labeling and target leakage?

### B. Strict 100% closure

Runtime parsing rejects any share sum other than 100. The overfit gate proves
the model can learn closure, but not that full-run OOD generations will close.
There is intentionally no post-hoc share renormalizer in the accepted path.

**Question:** Is strict rejection the right safety behavior, or should a
separately reported deterministic repair be allowed for otherwise valid
5%-bucket outputs? What repair-rate threshold would block shipping?

### C. E3 compute fairness

Two epochs preserve two exposures to every N5K example in both arms, but the
mixed arm has 4,983 extra FoodSeg examples and therefore about 2.85× more
optimizer steps.

**Question:** Is that the correct test of “add diverse data,” or must E3 also
include an equal-step/control arm to separate diversity from compute?

### D. Weak area shares

FoodSeg shares are pixel-area fractions, not mass fractions. The 5% buckets
reduce false precision but cannot remove systematic thin-spread/dense-food
bias.

**Question:** Should E3 selection rank HMR/IDR first and treat FoodSeg share
quality as diagnostic only, or can share degradation on N5K block the mixed
arm even if OOD identification improves?

### E. Resolver estimation rate

Only 17.6% of v8 names hit exact/fuzzy profiles; 82.1% use category defaults.
These defaults are numerically coherent and visibly estimated, but high
coherence is not the same as correct food density.

**Question:** Does this estimation rate invalidate the claim that the
architecture removes hard nutrition mistakes, or is the explicit uncertainty
UI sufficient for Stage 1? Should a minimum rung-1/2 rate become a ship gate?

### F. Category matching and HMR

HMR uses FDC ID/category, then frozen lexical fallbacks. Category-default
selection is deliberately conservative in nutrition output but still affects
whether a name error is classified hard or soft.

**Question:** Should HMR category matching use a separate reviewed taxonomy
instead of the runtime resolver to prevent a resolver heuristic from improving
its own metric?

### G. OOD grouping

NutritionVerse has multiple viewpoints per dish. Results will average
predictions within each of 225 represented dishes before averaging dishes.

**Question:** Should the ship gate require success per dish (best/median/worst
view), or is mean-over-views then mean-over-dishes the correct primary metric?

### H. Resume semantics

An attempted adapter resume reported 366.9M trainable parameters instead of
the fresh run's 32.5M and restarted at high loss. The attempt was stopped; the
successful memorization run was restarted and kept continuous. The existing
`train.sh` resume documentation predates this track.

**Question:** Should resume be disabled until mlx-vlm adapter stacking is
characterized, or is there a correct invocation that preserves the original
LoRA parameter set?

### I. Metric eligibility for unresolved items

The factored assembly evaluator reports completeness separately and computes
conditional kcal only for fully resolved, Tier-1-clean predictions. HMR/IDR
still cover every schema-valid IDENTIFY prediction.

**Question:** Is excluding incomplete predictions from conditional kcal the
right decontamination, provided completeness is a co-primary gate, or does it
create survivorship bias that requires a penalized headline?

## Planned next steps unless review blocks them

1. Finish Nutrition5K-only training.
2. Evaluate all 354 test records: test325 food plus 29 held-out negatives.
3. Run current v8, SCALE-v1, and the N5K-only IDENTIFY adapter on all 889
   NutritionVerse images, aggregated by the 225 represented dish IDs.
4. Train the exact FoodSeg103-mixed arm under the same two-epoch schedule.
5. Repeat in-distribution and OOD evaluation; select by Tier 1 first, then Tier
   2, always paired by dish.
6. Only if a candidate clears E3, run E4 share granularity and E5 confidence
   ranking experiments.
7. Convert the winning adapter, bundle the SQLite/model resources, and run
   Stage-3 Swift/iOS/device gates. v8 remains shipping until all pass.

## Minimal reproduction commands

Run from `ml/`:

```bash
./download_fdc.sh
.venv/bin/python download_foodseg103.py
./download_nutritionverse_real.sh
.venv/bin/python make_nutritionverse_eval.py
.venv/bin/python make_identify_data.py \
  --foodseg-root datasets/foodseg103 \
  --output-dir finetune_data_id_v1
.venv/bin/python check_prompt_parity.py \
  --data finetune_data_id_v1/train.jsonl --records 8
.venv/bin/python smoke_test.py \
  --data finetune_data_id_v1/train.jsonl --records 8 --batch-size 4
DATASET=finetune_data_id_v1 \
OUTPUT_PATH=runs/factored/e3-foodseg/adapter \
EPOCHS=2 ./train.sh
```

Primary implementation and evidence files:

- `docs/specs/2026-07-29-factored-pipeline-design.md`
- `ml/factored_pipeline/`
- `ml/make_identify_data.py`
- `ml/make_fdc_db.py`
- `ml/score_harness.py`
- `ml/visual_specialist/scale.py`
- `ios/SeeCal/Sources/SeeCalInference/FactoredNutritionPipeline.swift`
- `ios/SeeCal/Sources/SeeCalInference/FactoredScaleModel.swift`
- `ml/runs/factored/e0-v8-hard-mistakes.json` (local, gitignored)
- `ml/runs/factored/e1-v8-resolution.json` (local, gitignored)
- `ml/runs/factored/e2-mass.json` (local, gitignored)
- `ml/runs/factored/scale-v1/test.json` (local, gitignored)
