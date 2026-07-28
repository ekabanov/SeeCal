# SeeCal accuracy improvement options

**Date:** 2026-07-28  
**Scope:** calorie, macro, ingredient, and portion accuracy for an offline,
on-device iPhone product.  
**Status:** research report, not an execution plan.

## Executive conclusion

The highest-value next move is **not another base-model bake-off** and it is
not a direct retry of the v6 scalar-depth prompt. The evidence points to a
three-part program:

1. Establish an honest phone-photo benchmark before optimizing further.
2. Separate food understanding from numeric portion/nutrition regression
   instead of asking one generative loss to learn everything.
3. Add real phone data, then revisit depth as geometry or RGB-D features
   inside the portion estimator—not as one line of text sent to the VLM.

The current model is a good product baseline: v7b has a 30.5 kcal median
absolute error, valid output on 324/325 held-out food dishes, and perfect
not-food refusal on the 29-image negative test. Its 63.4 kcal MAE, however,
is measured entirely on the same fixed cafeteria rig distribution used for
training. It does not establish accuracy on the handheld photos the product
will receive.

There are also two architectural reasons to expect that the current approach
has headroom:

- The shipping adapter has 248 LoRA targets, **all in the language model**.
  It adapts neither the vision tower nor the multimodal projector.
- Training optimizes next-token cross-entropy over a long JSON response. It
  does not directly optimize calorie MAE, gram error, ingredient recall, or
  consistency between item nutrition and totals.

Published Nutrition5k work supports a decomposed approach. In its own
protocol, direct RGB nutrition prediction scored 70.6 kcal MAE, whereas
separating portion-independent composition from a volume-assisted mass
estimate reached 41.3 kcal MAE. Those values are not directly comparable to
SeeCal because the split and implementation differ, but the direction is
strong evidence for an architecture experiment, not a promised SeeCal gain.

### Recommended order

| Order | Work | Why now |
|---:|---|---|
| 1 | OOD benchmark: NutritionVerse-Real plus a small private iPhone set | Prevents optimizing the cafeteria rig instead of the product |
| 2 | Reproduce a small specialist RGB multi-task regressor | Tests whether the generative VLM is the numeric bottleneck; cheap enough for Core ML |
| 3 | Hybrid prototype: VLM ingredients/refusal + specialist mass/totals | Preserves SeeCal's ingredient UX while using the right loss for numbers |
| 4 | Cafe2 + downscaled, sampled side-angle training | Uses data already on disk and improves viewpoint robustness |
| 5 | Tail-aware sampling and numeric-aware training | Directly targets the observed high-calorie regression-to-the-mean |
| 6 | Collect weighed phone RGB/RGB-D meals | Supplies the missing target-domain supervision |
| 7 | Per-item depth/geometry fusion | Promising only after it can be trained and tested on phone depth |
| 8 | Visual LoRA/projector adaptation | Untried and plausible, but requires training and iOS adapter work |
| 9 | Other VLMs / larger Qwen | Useful as controlled probes or teachers; low priority for shipping |

## 1. What the evidence says now

### 1.1 Current result

The canonical numbers are in the top-level [README](../README.md) and raw
summaries under `ml/runs/`.

| Model | Evaluation | Calories MAE | Median | Failures | Important interpretation |
|---|---|---:|---:|---:|---|
| Qwen3.5-4B base | 50 Nutrition5k dishes | 83.4 | 63.9 | 0 | Untuned baseline only |
| v5 | 325 Nutrition5k dishes | 59.0, n=322 | 31.4 | 3 parse | Food-only predecessor |
| v6 scalar depth | 325 Nutrition5k dishes | 62.5, n=325 | 35.3 | 0 | Paired tie with v5: +3.48 kcal, t=0.89 |
| **v7b shipping** | 325 Nutrition5k dishes | **63.4, n=324** | **30.5** | 1 parse | Paired tie with v5: +0.87 kcal, t=0.23 |
| v7b refusal | 29 non-food images | — | — | 0 | 29/29 correct refusals; 0 false refusals on food |
| Gemma 4 E4B base | 50 Nutrition5k dishes | 159.4 | 122.7 | 0 | Too inaccurate and too large in the tested form |

The v5/v7b headline MAEs use different answered subsets and must not be
compared directly. The paired result is the valid comparison.

### 1.2 The mean hides a serious high-calorie tail

The v5 full-evaluation log preserves dish IDs, predictions, and targets, so it
can be joined to the committed test JSONL. This diagnostic was recomputed for
this report; it is not a new model evaluation.

| Ground-truth calories | Dishes | MAE | Mean bias, prediction − truth | Median absolute error |
|---|---:|---:|---:|---:|
| <100 | 96 | 14.8 | +5.1 | 9.4 |
| 100–299 | 111 | 45.5 | +19.1 | 32.4 |
| 300–499 | 72 | 71.0 | −23.3 | 62.2 |
| ≥500 | 46 | **165.6** | **−115.7** | **139.9** |

The largest 50 errors account for 52.5% of total absolute error. The model is
good on many small/simple dishes and regresses large meals toward the training
mean. This is exactly the kind of failure that a 30.5 kcal median obscures.

The training set reinforces the explanation: its calorie median is 208.1,
mean is 255.7, and only 322/2,594 records are at least 500 kcal. The next
training experiments should therefore report calorie-bin bias and p90 error,
not only one aggregate MAE.

The tail analysis is from v5 because the v7b run did not preserve a per-dish
log in the repository. Since v7b and v5 are a paired statistical tie, the
finding is a strong hypothesis for v7b, not a substitute for regenerating a
v7b per-dish artifact.

### 1.3 The current evaluation is in-domain

The current 3,244 food records across train/validation/test are all cafe1
overhead views from the Nutrition5k RealSense rig. The held-out split changes
the dish, but not the camera, capture distance, plate style, environment, or
lighting family. The test therefore answers:

> How well does this model generalize to another dish photographed by the same
> cafeteria rig?

It does not answer:

> How well does this model work on a handheld iPhone photo in a home,
> restaurant, takeaway box, bowl, or low-light setting?

This is the largest unmeasured risk.

### 1.4 v6 did not close the depth question

v6 tested one representation: a plate-inclusive geometric volume and maximum
height rendered as a text line in the prompt. It was useful early in training
but converged to a tie with RGB-only v5.

That result closes the case for **shipping that scalar-prompt implementation**.
It does not test:

- aligned RGB and depth feature fusion;
- per-item segmentation and per-item volume;
- a learned mass regressor conditioned on food class/density;
- phone LiDAR captured in the product distribution;
- multi-frame depth accumulation;
- a user-provided scale or known total mass.

The original Nutrition5k paper found that both depth as a fourth channel and a
volume-assisted mass model improved its RGB baseline, with the decomposed
volume-scalar pipeline performing best in its protocol. More recent
Nutrition5k systems also use separate RGB/depth encoders, cross-modal
attention, ingredient features, and task-specific regressors. This suggests
that *where* depth enters the system matters more than the fact that it is
available.

### 1.5 The visual representation is frozen

The shipping adapter's config contains 248 LoRA keys:

- 248 under `language_model`;
- 0 under the vision tower;
- 0 under a multimodal projector/aligner/connector.

The model can learn how to map its existing visual representation to SeeCal
JSON, but it cannot specialize the representation itself for fine-grained
food, portion, container, or ingredient cues.

`mlx-vlm` exposes `--train-vision`, but in the pinned version this unfreezes
the full vision stack; it does not create a compact vision LoRA. That would
produce a much larger trainable artifact and does not fit the current
language-LoRA conversion/loading contract. A practical visual-adaptation
experiment needs targeted LoRA in the last vision blocks/projector, plus an
iOS loading design.

### 1.6 A single photo has an accuracy ceiling

Some calorie information is simply not visible: cooking oil absorbed into
food, sugar in a sauce, the composition of a puree, filled items, and the
density of visually similar portions. Depth can improve volume, but it cannot
infer invisible ingredients or convert volume to mass without a density
assumption.

A 2026 comparative VLM study on Nutrition5k reports large nutrient errors even
when models are given correct ingredient input. A depth-smartphone study
reached 41.2 kcal mean absolute error on 48 meals, but food type was manually
selected and segmentation could be corrected by the user. That is evidence
that geometry plus structured food identity can be strong—not evidence that
fully automatic single-photo recognition has solved the problem.

## 2. Options ranked

The upside ratings are qualitative judgments for SeeCal, not projected
percentage improvements.

| Option | Expected upside | Effort | Shipping risk | Priority |
|---|---|---|---|---|
| Honest OOD evaluation | Enables every later decision | Low–medium | None | **P0** |
| Specialist RGB multi-task regressor | High | Medium | Low; compact Core ML/MLX model | **P0** |
| Hybrid VLM + numeric estimator | High | Medium–high | Moderate integration complexity | **P0** |
| Weighed phone-data collection | Highest long-term | High operational effort | Consent/licensing/data governance | **P0** |
| Cafe2 + sampled side-angle views | Medium | Low–medium | Low | **P1** |
| Tail-aware sampling / loss | Medium | Low–medium | Low | **P1** |
| Simplified structured target + nutrition DB | Medium–high | Medium | Ingredient mapping/versioning | **P1** |
| Visual LoRA / projector tuning | Medium–high | Medium–high | Training and Swift adapter changes | **P1** |
| Per-item RGB-D geometry/fusion | High if calibrated | High | LiDAR coverage, UX, sensor domain shift | **P1 after phone data** |
| Optional known weight / scale input | Very high when used | Low–medium | Extra user action | **P1 product option** |
| NutritionVerse-Synth pretraining | Medium, uncertain | Medium–high | Synthetic gap; verify license | P2 |
| Test-time multi-view/ensemble | Medium | Medium | 2× latency and energy | P2 |
| Qwen 9B or another larger VLM | Uncertain | Medium | Does not fit current bundle/RAM budget | P2/research |
| Smaller VLM/FastVLM | Low as a direct accuracy move | Medium | New serving stack | P2/efficiency |

## 3. Data options

### 3.1 Dataset comparison

| Dataset/source | Useful signal | Best role for SeeCal | Caveats |
|---|---|---|---|
| Nutrition5k cafe1, current | Weighed ingredients, macros, overhead RGB-D | Keep as core supervised set and regression benchmark | Fixed rig and one cafe; current split is in-domain |
| Nutrition5k cafe2 | About 228 usable overhead dishes | Cheap compatible expansion | Same collection style; modest size |
| Nutrition5k side angles | 271,616 frames already local; multiple cameras and views | Viewpoint/domain augmentation | Near-duplicates; 19 GB; must downscale and split by dish |
| NutritionVerse-Real | 889 iPhone images, 251 dishes, 45 food types, weighed ingredients and segmentation masks | **First untouched OOD benchmark**; later target-domain fine-tuning | Small; preserve a dish-level test split; verify dataset terms before training |
| NutritionVerse-Synth | 84,984 rendered images with nutrition, depth, instance and semantic masks | Segmentation/depth pretraining before real fine-tuning | Synthetic-to-real gap; audit license and constituent assets |
| FoodSeg103 | 7,118 images, 103 ingredient classes, pixel masks | Ingredient segmentation/visual-encoder pretraining | No mass or nutrition; curated recipe images differ from phone capture |
| FastFood | 84,446 images, 908 categories, ingredients and nutrition | Food/ingredient representation pretraining | Portion variation and product applicability need validation; audit image/data rights |
| MetaFood3D | 637+ scanned food objects, RGB-D/3D/nutrition/weight | Geometry, rendering, and portion-estimation research | Current site is CC BY-NC 4.0; do not train a commercial shipping model without permission |
| SimpleFood45 | 513 phone images of 45 items with volume, mass, energy and a checkerboard | Geometry sanity benchmark | Only 12 food types; reference marker; too small/simple as main training data |
| Food2K / Recipe1M+ | Very large category/recipe corpora | General food representation or ingredient retrieval | No weighed portion ground truth; web-image rights and product licensing need careful review |

### 3.2 Immediate expansion from data already present

#### Cafe2

Regenerate metadata with `--include-cafe2`, preserve dish-level splitting, and
add the roughly 228 overhead-capable dishes. This is low risk and should be
included in any new data generation, but it is unlikely to fix phone-domain
shift alone.

Do not overwrite the historical v2 split or reuse its headline numbers under
the same name. Produce a new versioned dataset and evaluate both the historical
test set and the new split.

#### Side-angle frames

The side-angle archive is the most underused asset in the repository. The
right experiment is not to train on all ~115 frames per dish:

1. Select 2–4 maximally separated, sharp views per dish.
2. Downscale each view to a measured image-token budget below the 2,048-token
   sequence cap.
3. Treat views as independent training records initially; keep every view of a
   dish in one split.
4. Add phone-like crops, mild rotation/perspective, exposure, white-balance,
   blur, and compression augmentation.
5. Keep the overhead-only historical test untouched and add a separate
   view-robustness test.

This can take training from 2,594 food records to roughly 8,000–12,000 without
new annotation. The recent VLM comparison study also found that progressive
multi-view input improves ingredient recognition, although a two-photo product
flow must be evaluated separately from multi-view training.

### 3.3 Build the missing product dataset

The long-term answer is a SeeCal phone dataset. A useful staged collection is:

#### Stage A: 150–250 meal benchmark

- Untouched evaluation only.
- Multiple people, homes/restaurants/takeaway, cuisines, bowls/plates/boxes,
  lighting conditions, and iPhone models.
- Ingredient-by-ingredient weighed ground truth, including oils, dressings,
  and sauces.
- One normal product photo; optionally a second angle.
- ARKit/AVFoundation depth, intrinsics, confidence, camera pose, and distance
  when available.
- Split by meal preparation/session and contributor, never by image.

#### Stage B: 500–1,500 training meals

- Oversample the current weak regions: ≥500 kcal meals, mixed dishes, bowls,
  sauces, fried foods, layered foods, and culturally diverse meals.
- Retain at least 20% as a permanently untouched test.
- Use assisted segmentation tools, then human review.
- Store explicit food/database IDs and preparation state when possible.

User edits from the app can become an active-learning signal, but ordinary
edits are not measured ground truth. They are most trustworthy for ingredient
identity and failure discovery. Gram and calorie supervision should be marked
as measured, package-derived, user-estimated, or model-derived rather than
mixed together.

## 4. Model and training options

### 4.1 Reproduce a specialist numeric baseline

This should be the first model experiment because it tests the core
architecture assumption cheaply.

Train a compact visual backbone with independent heads for:

- total mass;
- calories, protein, fat, and carbohydrates;
- optional calorie density / macros per gram;
- ingredient multi-label prediction;
- optional food/not-food.

Use Huber or L1-style numeric losses, log/relative variants for the long tail,
and explicit task weights. Start RGB-only so the comparison is clean. Suitable
backbones include a small ViT/MobileViT/EfficientNet-class encoder that can be
exported to Core ML. It need not generate text.

Why this is promising:

- It directly optimizes numeric error.
- It is likely much smaller and faster than a 4B VLM.
- Published Nutrition5k systems built around multi-task regressors report
  materially better in-protocol numbers than generic VLM prompting.
- It creates a natural place to fuse depth later.

Why it cannot simply replace Qwen:

- SeeCal needs ingredient names and editable item breakdowns, not only totals.
- Closed-set ingredient heads will not cover every real food.

That makes the specialist model a candidate for a hybrid, not necessarily a
full replacement.

### 4.2 Hybrid: VLM semantics, specialist numbers

A practical hybrid can retain Qwen for:

- food/not-food;
- open-vocabulary meal and ingredient names;
- item count and coarse segmentation prompts;
- optional clarification questions.

The specialist path handles:

- total mass/volume;
- item gram allocation;
- calories/macros or a correction factor over database-derived totals;
- uncertainty.

Candidate fusion order:

1. Qwen emits normalized ingredient names and rough grams.
2. A local nutrition table maps ingredients to per-100 g values.
3. A small visual/depth regressor estimates total mass and/or item mass shares.
4. The app computes totals deterministically.
5. A calibration head corrects residual bias using visual features and
   container/meal type.

USDA FoodData Central provides downloadable Foundation Foods, FNDDS, and
branded data in CSV/JSON, so a curated local subset is feasible. The mapping
must be versioned and reviewed; blindly fuzzy-matching generated ingredient
names would introduce a new silent failure mode.

Before implementation, run three oracle analyses:

1. Ground-truth ingredient names + ground-truth grams + database nutrition.
2. Ground-truth names + model grams.
3. Model names + model grams.

These isolate composition, portion, and recognition error and reveal whether
the database path actually improves the current targets.

### 4.3 Simplify the generative target

Today the model generates totals and six numeric fields per item. That repeats
the same nutrition information and spends loss/output tokens on JSON syntax
and redundant arithmetic.

Test a new completion schema containing only:

- `not_food`;
- meal title;
- item names;
- estimated grams;
- optional confidence/ambiguity flags.

Calculate nutrition from a curated local table, or have a separate numeric
head produce totals. Benefits include shorter inference, fewer parse
opportunities, deterministic item/total consistency, and a clearer division
between recognition and nutrition density.

This is a new adapter/app contract. Prompt parity remains mandatory, but the
v7 lesson still applies: introduce behavior through the completion unless a
new versioned prompt is deliberately trained and shipped together.

### 4.4 Tail-aware and numeric-aware training

Low-risk experiments on the existing approach:

- Sample or weight dishes by calorie/mass bins so ≥500 kcal meals are not a
  small minority.
- Weight total-calorie, total-mass, and gram-value tokens more than repeated
  JSON keys.
- Compare L1/Huber auxiliary losses on parsed numeric fields against token CE
  alone.
- Predict `total_mass_g` and per-100 g nutrient density as explicit
  intermediate targets.
- Stratify by ingredient count, bowl/container type, and high-fat foods.
- Try curriculum: food recognition/ingredients first, quantities second.
- Evaluate checkpoints, not only the final epoch, because the v6 probe showed
  that early and final behavior can diverge.

Stock `mlx-vlm` does not provide field-weighted parsed losses, so the auxiliary
numeric objective requires a custom trainer. Simple calorie-bin sampling can
be tested first without changing the trainer.

### 4.5 Adapt the vision side

Three increasingly expensive probes:

1. Train only the multimodal projector/connector.
2. Add LoRA to the final vision blocks plus projector.
3. Unfreeze the full vision stack.

The second option is the best research target: compact enough to preserve an
adapter-style update, but able to learn food-specific textures, boundaries,
portion cues, and containers. It requires custom `mlx-vlm` targeting and a
Swift loading/conversion extension. The third option should only follow if the
compact visual adapter proves insufficient.

Gate this work with ingredient recall and portion metrics. A lower calorie MAE
that damages open-vocabulary ingredient quality is not a product win.

### 4.6 Hyperparameter search

The current rank 16, alpha 32, LR 1e-4, two-epoch recipe is validated but not
optimized. A small matched sweep can test:

- rank 8/16/32;
- learning rate around 3e-5, 1e-4, and 3e-4;
- two vs three epochs with checkpoint selection;
- small LoRA dropout;
- language-only vs projector/vision adaptation.

This is likely a modest-gain track. Run it after the evaluator is improved so
the search does not overfit a noisy 50-dish headline.

### 4.7 Other base models

#### Qwen3.5-9B

The official model card shows only incremental gains over 4B on many general
vision benchmarks, while the MLX 4-bit conversion is about 5.6 GB. It exceeds
the 4 GB maximum uncompressed iOS app size if bundled, and the current 4B
model already peaks near 4.1 GB of device memory after MLX cache hardening.
Apple-hosted Background Assets can move a large model out of the main build,
but they do not solve runtime RAM/jetsam risk or first-run availability.

Use 9B as an offline experiment/teacher only unless device measurements
demonstrate otherwise.

#### Qwen3.5-2B / 0.8B and FastVLM 1.5B

These are efficiency candidates, not obvious direct accuracy upgrades.
Qwen3.5-2B keeps the same model family; Apple FastVLM supplies an iPhone demo
and emphasizes a much cheaper vision encoder. A smaller model could make
two-view inference, ensembles, or a specialist side model affordable.

Benchmark only after the hybrid path exists. A smaller, well-supervised
specialist system may beat a larger generic generator, but a bare model swap
is unlikely to.

#### Gemma

The local Gemma 4 E4B base evaluation is already decisive for the tested
artifact: 159.4 kcal MAE on the same 50 dishes and excessive size. Do not spend
more on Gemma unless a materially different, shippable release changes both
facts.

#### Distillation

A larger model can provide ingredient pseudo-labels for unlabelled phone
images or supervise a smaller student, but measured nutrition labels remain
more valuable than teacher-generated numbers. Never treat a teacher's calorie
estimate as ground truth.

## 5. Depth and portion-size options

### 5.1 Recommended depth architecture

Treat depth as a measurement feeding a portion model:

```text
RGB ───────► food/item segmentation ─┐
                                     ├─► per-item geometry ─► volume
depth + intrinsics + confidence ─────┘

ingredient identity + volume + learned density ─► mass ─► nutrition
```

This is materially different from:

```text
depth ─► one total-volume number ─► text prompt ─► VLM generates everything
```

The stronger design permits:

- per-item rather than whole-plate volume;
- container/bowl surface modeling;
- class-conditioned density;
- confidence filtering;
- a numeric loss on volume/mass;
- clear sensor calibration tests independent of the VLM.

### 5.2 iPhone capture requirements

Apple exposes:

- `smoothedSceneDepth`, which averages depth across frames;
- a per-pixel confidence map;
- camera calibration/intrinsics with captured depth;
- synchronized video and depth through AVFoundation.

The capture gate should reject or fall back when:

- depth confidence is poor over the meal;
- reflective/dark surfaces degrade readings;
- the support plane is not visible;
- camera angle/distance is outside the calibrated region;
- segmentation cannot separate food from bowl/plate.

Never hardcode the Nutrition5k focal length for iPhone. Use each frame's
calibration data and transform depth into the RGB coordinate system.

### 5.3 Required calibration dataset

Before Swift implementation, collect at least 100–200 weighed phone meals with
RGB, raw/smoothed depth, confidence, intrinsics, device model, pose, and
per-item masks. Evaluate the depth module independently:

- volume error where volume can be measured;
- mass error after density conversion;
- plane bias in millimeters;
- missing/low-confidence pixel rate;
- error by bowls, plates, reflective food, and capture angle.

Only then compare RGB vs RGB-D nutrition models on the same dish split.

### 5.4 Multi-view alternatives

For non-LiDAR devices or difficult bowls:

- two RGB views with guided top/side capture;
- short video with pose-aware frame selection;
- monocular depth/shape reconstruction;
- a known plate diameter or reference object.

Two-view reconstruction has reported sub-10% average volume error on a
77-dish research set, but it adds capture friction and reconstruction work.
Monocular predicted depth adds no new sensor information, yet it can supply a
useful inductive bias; DPF-Nutrition and shape-reconstruction systems report
gains on Nutrition5k. Both need validation on the phone benchmark.

### 5.5 The highest-accuracy optional input: known mass

Add an optional “I know the total weight” path, supporting manual entry and
possibly a scale photo or Bluetooth scale later. The Nutrition5k
portion-independent result—24.1 kcal MAE when portion is supplied in its
protocol—shows how much of the problem is portion size.

This will not be used for every meal, but it can provide:

- a high-accuracy mode for users who weigh food;
- measured labels for the data flywheel;
- a calibration reference for depth;
- a way to allocate total mass across model-predicted items.

This option has a better accuracy/engineering ratio than trying to infer
invisible mass from a single photo in every case.

## 6. Evaluation protocol for all experiments

### 6.1 Test sets

Maintain three separate, versioned suites:

1. **N5k historical:** the current 325 dishes, untouched, for continuity.
2. **Public OOD:** NutritionVerse-Real, split by dish rather than image.
3. **Product OOD:** permanently held-out SeeCal iPhone meals.

Add targeted suites for not-food, ambiguous/non-visible ingredients, bowls,
high-calorie meals, and sensor failure.

### 6.2 Metrics

Report at minimum:

- calorie/macro MAE and median absolute error;
- p75/p90/p95 absolute error;
- signed bias;
- calorie-bin metrics: <100, 100–299, 300–499, ≥500;
- relative error with a documented near-zero policy;
- total and per-item gram MAE;
- ingredient precision, recall, and F1 after normalized matching;
- item-count error;
- food/refusal precision and recall;
- JSON/schema failure rate;
- latency, peak memory, thermal behavior, and app/model size.

Use one failure-aware headline metric over all dishes. Do not silently drop
parse/schema/refusal failures from one arm and compare it with another arm's
different subset. Preserve per-dish prediction artifacts for every run.

### 6.3 Statistical decision rule

- Compare identical dish IDs.
- Use paired bootstrap confidence intervals for MAE/median/tail changes.
- Report wins/losses and signed per-dish differences.
- Require no meaningful regression in refusal, ingredient recall, p90 error,
  latency, or memory.
- A probe may graduate on a directional result, but shipping requires the full
  historical and OOD suites.

Suggested graduation gates:

| Stage | Gate |
|---|---|
| 50-dish diagnostic | Catches wiring failures only; never a shipping decision |
| 325 N5k | ≥5% paired calorie-MAE improvement or a clearly better p90/tail with no median regression |
| Public OOD | Directionally consistent improvement and no new large failure class |
| Product OOD | Statistically credible improvement on the actual capture flow |
| Device | Peak memory/thermal stability and acceptable end-to-end latency |

## 7. Concrete experiment program

### Phase 0 — evaluator and baselines

1. Make `infer.py` persist every dish ID, target, raw output, parsed output,
   failure reason, and latency.
2. Add failure-aware paired comparison and calorie-bin/tail reports.
3. Evaluate v7b on NutritionVerse-Real without training on it.
4. Capture a 30–50 meal pilot iPhone set to expose gross domain failures.
5. Recompute the v7b high-calorie diagnostic with preserved IDs.

**Decision:** if OOD degradation is large, prioritize phone data and
augmentation over hyperparameter tuning.

### Phase 1 — cheapest accuracy probes

Run matched, GPU-serialized probes:

1. Cafe2 expansion.
2. Cafe2 + 2–4 downscaled side views per dish.
3. Tail-balanced sampling on the same data.
4. Simplified `ingredients + grams` completion with deterministic nutrition.
5. Compact RGB multi-task regressor.

Do not combine all changes initially; the goal is to learn which mechanism
helps.

### Phase 2 — hybrid and visual adaptation

1. Run the three oracle database analyses.
2. Fuse Qwen semantic outputs with specialist total mass/nutrition.
3. Prototype projector-only adaptation.
4. Implement compact visual LoRA only if projector tuning is insufficient.
5. Compare the hybrid with current end-to-end JSON generation on all suites.

### Phase 3 — target-domain data

1. Freeze the first private OOD test set.
2. Collect 500+ weighed phone meals.
3. Fine-tune with a controlled mix of Nutrition5k and phone data.
4. Use source-balanced sampling so the larger rig dataset does not drown out
   the target domain.
5. Re-evaluate by contributor, location, cuisine, container, and calorie bin.

### Phase 4 — depth

1. Build an offline Python reference pipeline on captured iPhone RGB-D.
2. Validate geometry/volume before any VLM integration.
3. Compare total scalar, per-item geometry, and learned RGB-D fusion.
4. Port only the winning, calibrated representation to Swift.
5. Keep RGB-only fallback in-distribution through sensor dropout during
   training.

## 8. What not to do

- Do not ship v6 behind a toggle as an “accuracy improvement”; it is a measured
  tie and its training sensor does not match the phone.
- Do not evaluate adapters on different answered subsets.
- Do not train on every side-angle frame; near-duplicate leakage will inflate
  sample count without comparable information.
- Do not mix NutritionVerse-Real into training before recording the untouched
  OOD baseline.
- Do not infer that a lower language-model benchmark score predicts worse food
  regression, or that a larger VLM predicts better nutrition.
- Do not use non-commercial 3D datasets in a shipping model without explicit
  rights.
- Do not call user-estimated corrections measured ground truth.
- Do not optimize only mean MAE; the current product risk is concentrated in
  high-calorie/tail failures.

## 9. Source notes

Repository evidence:

- [Top-level results and current architecture](../README.md)
- [Training pipeline and validation ladder](../ml/README.md)
- [ML/iOS contract](architecture.md)
- [Training and failure history](training-history.md)
- [Depth experiment design and sensor findings](design/2026-07-26-depth-design-brief.md)

Primary/public sources:

- [Nutrition5k paper and RGB/depth/volume results](https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf)
- [NutritionVerse-Real: 889 phone images, 251 dishes, measured ingredients](https://arxiv.org/abs/2401.08598)
- [NutritionVerse-Synth and real/synthetic transfer study](https://arxiv.org/abs/2309.07704)
- [FoodSeg103 ingredient segmentation benchmark](https://arxiv.org/abs/2105.05409)
- [FastFood and visual-ingredient fusion](https://arxiv.org/abs/2505.08747)
- [MetaFood3D official dataset page and license](https://lorenz.ecn.purdue.edu/~food3d/)
- [SimpleFood45 official dataset page](https://lorenz.ecn.purdue.edu/~gvinod/simplefood45/)
- [DPF-Nutrition RGB/predicted-depth fusion](https://arxiv.org/abs/2310.11702)
- [Monocular shape-reconstruction portion estimation](https://arxiv.org/abs/2308.01810)
- [2026 VLM comparison on ingredient and nutrient estimation](https://www.sciencedirect.com/science/article/pii/S266592712600105X)
- [Depth-smartphone volumetric study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7142738/)
- [Apple ARKit smoothed scene depth](https://developer.apple.com/documentation/arkit/arframe/smoothedscenedepth)
- [Apple ARKit depth confidence](https://developer.apple.com/documentation/arkit/ardepthdata/confidencemap)
- [Apple AVFoundation depth capture](https://developer.apple.com/documentation/avfoundation/avcapturedepthdataoutput)
- [Apple depth camera calibration data](https://developer.apple.com/documentation/avfoundation/avdepthdata/cameracalibrationdata)
- [Qwen3.5-4B official model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5-9B MLX 4-bit size](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)
- [Apple FastVLM official repository and iPhone path](https://github.com/apple/ml-fastvlm)
- [USDA FoodData Central downloadable data](https://fdc.nal.usda.gov/download-datasets/)
- [Apple maximum build sizes and Background Assets](https://developer.apple.com/help/app-store-connect/reference/app-uploads/maximum-build-file-sizes)

Dataset licenses and terms can change or be narrower than paper access. Audit
the exact downloaded artifact and intended commercial use before any external
dataset enters a shipping model.
