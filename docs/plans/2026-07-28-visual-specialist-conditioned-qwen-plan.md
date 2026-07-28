# Visual specialist → conditioned Qwen plan

*Added 2026-07-28. Branch: `codex/visual-specialist`. Status: complete and
selected for the app. The user chose conditioned Qwen3.5-4B + S1 specialist
after the full 325-dish and 29-negative evaluation. No weights or source images
are published.*

## Final decision and implementation

The selected system is:

- S1 MobileNetV3-Large specialist (`s1-sideviews-v2/best.pt`);
- split-conformal widening fitted at 80% target coverage;
- Qwen3.5-4B 4-bit with `adapters_v8_numeric_4b`;
- exact five-field auxiliary prompt block; no class, container, or classifier
  outputs are sent to Qwen;
- Qwen remains responsible for food refusal and ingredient-level JSON.

Full untouched Nutrition5K test results: calorie MAE 56.8, median 29.7,
protein/fat/carbohydrate MAE 5.1/4.4/6.0, 325/325 valid schemas, and exact
normalized ingredient-name micro-F1 0.594. The conditioned model refused 29/29
held-out negatives. Against v5 on 322 shared dishes it improved calorie MAE
from 59.0 to 57.2 (mean paired delta −1.8 kcal, bootstrap 95% CI −8.3 to
+4.7); this is a statistical tie, not evidence of a large historical-domain
gain. The user selected it for its overall quality, zero failures, best median,
and extensible calibrated specialist path.

The app runs the specialist sequentially before Qwen, reproduces the training
preprocessing and fixed JSON renderer in Swift, and sends the trained
`{"available":false}` block if an individual specialist prediction fails.
Device builds require the `v8-conditioned` adapter and compiled specialist.
The specialist compiles to 9.5 MB; its macOS Core ML integration check completed
in 0.67 seconds on one cold held-out image. A 50-image Swift-versus-Python
parity gate measured 10.0% mean relative drift across all numeric estimates
and 10.2% on calories, within the 20% perturbation range used during conditioned
Qwen training. iPhone memory, thermal, and latency remain a post-merge
physical-device gate rather than an inferred claim.

## Objective

Improve SeeCal's real-photo nutrition accuracy without requiring a new
user-collected dataset by combining:

1. a compact visual specialist trained to estimate whole-meal mass, calories,
   and macros from measured data, while learning better food representations
   from licensed crowdsourced photos; and
2. a fresh Qwen3.5 LoRA that receives the specialist's fallible structured
   prediction as auxiliary evidence and uses Qwen's pretrained knowledge to
   produce the open-vocabulary ingredient breakdown.

Both Qwen3.5-4B and Qwen3.5-2B are candidates. The 4B model is the accuracy
control; the 2B model is a memory/latency candidate trained with conservative
photo augmentations. The final choice is empirical rather than assumed.

The production hypothesis is:

```text
photo
  ├─► compact specialist
  │      └─► mass/calories/macros + calibrated uncertainty
  │
  └─► Qwen image input
          +
      fixed auxiliary-measurement text block
          │
          ▼
      new conditioned LoRA
          │
          ▼
      existing editable ingredient JSON
          │
          ▼
      deterministic validation and item-derived totals
```

Late fusion—rescaling Qwen's items after generation—is retained as a control,
not assumed to be the final architecture. It tells us whether the specialist
contains useful independent signal before a new 4B LoRA is trained.

## Constraints and non-goals

- The shipping `adapters_v7b` weights and its prompt contract remain untouched.
- The conditioned adapter is a new version, provisionally `v8`; it is trained
  from a clean 2B or 4B base model, not continued from v7b.
- The v8 training and iOS prompts must again be byte-identical. The specialist
  block is real per-photo sensor data, not a wording-only attempt to teach new
  behavior.
- Public or teacher-labeled photos never receive invented calorie, macro, or
  gram targets. Their numeric loss masks are always off.
- Nutrition5k's 325 historical test dishes and NutritionVerse-Real remain
  untouched by training and hyperparameter selection.
- No arbitrary web scrape is in scope. Every image requires stored provenance
  and a license/terms record.
- No existing dataset, adapter, or model artifact is deleted.
- No model is published and no paid API spend exceeds the approved cap.
- This pilot can support a research decision without new user photos. A
  production claim about ordinary SeeCal captures still requires a later
  device-domain gate.

## System responsibilities

### Compact specialist

Inputs:

- one RGB meal image;
- no Qwen output and no generated text.

Initial outputs:

```json
{
  "food_probability": 0.99,
  "mass_g": {"p10": 390.0, "p50": 470.0, "p90": 570.0},
  "calories": {"p10": 530.0, "p50": 640.0, "p90": 770.0},
  "protein_g": {"p10": 25.0, "p50": 32.0, "p90": 42.0},
  "fat_g": {"p10": 17.0, "p50": 24.0, "p90": 34.0},
  "carbs_g": {"p10": 58.0, "p50": 71.0, "p90": 88.0}
}
```

The interval widths provide an uncertainty signal. Only fields that calibrate
on held-out validation data are passed to Qwen. Food classes, container type,
and segmentation are auxiliary training tasks; they enter the Qwen prompt only
if an ablation demonstrates additional value.

### Conditioned Qwen

Qwen retains responsibility for:

- food/not-food refusal;
- meal and ingredient names;
- open-vocabulary recognition;
- per-item gram and nutrition allocation;
- valid JSON in the existing app schema.

The exact auxiliary prompt block will live in one shared specification and be
rendered independently by Python and Swift. A representative form is:

```text
Auxiliary visual measurement (fallible; use as evidence, not ground truth):
{"available":true,"mass_g":470.0,"calories":640.0,"protein_g":32.0,"fat_g":24.0,"carbs_g":71.0,"confidence":0.81}
```

Formatting is part of the model contract: key order, whitespace, decimal
rounding, boolean spelling, unavailable representation, and placement relative
to the image tokens and existing prompt are fixed and tested.

#### 2B candidate

`Qwen/Qwen3.5-2B` is an official, Apache-2.0, natively multimodal checkpoint.
The current MLX-community 4-bit conversion is approximately 1.7 GB on disk,
compared with 2.9 GB for the local 4B conversion. This makes room for the
specialist and may reduce runtime memory and latency, but the reduction must be
measured: the two sizes may share substantial vision-tower cost.

The smaller model is not presumed equivalent. Qwen's official generic
vision-language results show a meaningful capacity gap—for example, MMMU-Pro
50.3 for Qwen3.5-2B versus 66.3 for Qwen3.5-4B. The hypothesis is narrower:
specialist conditioning reduces the amount of numeric visual reasoning Qwen
must perform, allowing 2B to retain the ingredient/open-vocabulary role.

Sources:

- [Official Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Official Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [MLX-community Qwen3.5-2B 4-bit conversion](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit)

### Deterministic app layer

The app does not invent a second inference policy. It:

1. runs the specialist and retains its small structured result;
2. renders the exact v8 auxiliary block;
3. runs Qwen with the original image and v8 prompt;
4. handles `{"not_food": true}` exactly as today;
5. validates non-negative finite item values; and
6. derives displayed and logged meal totals from the returned items, preserving
   the current editing/reset/persistence contract.

If the specialist fails, v8 receives the trained `{"available":false,...}`
variant rather than an ad-hoc prompt. Development builds retain a v7b fallback
until v8 clears every gate.

## Data design

| Source | Role | Numeric losses | Semantic losses | Split rule |
|---|---|---:|---:|---|
| Nutrition5k overhead | Measured numeric supervision | On | On | Existing dish split; test untouched |
| Nutrition5k/Cafe2 side views | Viewpoint robustness tied to measurements | On | On | All views grouped by dish |
| v7b COCO negatives | Food/not-food preservation | Off | On | Existing split |
| Food Recognition Benchmark 2022 v2.1 public training release | Crowdsourced phone-domain representation | Off | On | 5,000 stratified images from 54,392; group/deduplicate |
| NutritionVerse-Real | Public phone OOD test | Evaluation only | Evaluation only | Group by dish; never train |

Every manifest row records source, source ID, image checksum, license, split,
dish/group ID, label provenance, teacher model/version, prompt hash, raw
teacher response, normalized labels, and per-field loss masks.

Before splitting, perceptual hashes detect exact and near duplicates across
sources. All views, crops, and duplicates from the same dish stay in one split.

## Model design

Start with two Core ML-compatible pretrained backbones:

1. a `torchvision` MobileNetV3-Large baseline; and
2. EfficientNetV2-S, or a smaller Core ML-friendly FastViT if its export and
   weight-license audit are cleaner.

An untrained export smoke test happens before any long training. Unsupported
operators, excessive memory, or an unexpectedly large package eliminate a
candidate early.

The selected backbone receives independent heads for:

- log-scaled total mass;
- log-scaled calories, protein, fat, and carbohydrates;
- p10/p50/p90 quantiles for numeric uncertainty;
- food/not-food;
- Food Recognition Benchmark 2022/source food classes;
- optional container/cooking-state labels;
- optional segmentation only after the non-segmentation pilot.

Measured records optimize numeric and semantic losses. Public-photo records
optimize only applicable semantic losses. Source-balanced batches prevent
5,000 public records from drowning the smaller measured set. The first target
mix is:

- 50% measured Nutrition5k/Cafe2;
- 20% measured side views;
- 20% public semantic records;
- 10% food/not-food and hard negatives.

Exact ratios may change using validation data, but the historical test and
NutritionVerse-Real never participate in that choice.

## Execution phases

### Phase 0 — guardrails, environment, and frozen baselines

1. Record toolchain versions, current commit, model hashes, disk space, and
   current branch.
2. Add specialist/teacher-specific environment lock files rather than
   destabilizing the known-good `mlx-vlm` environment. The teacher lock is
   `ml/teacher_labeling/requirements.lock`; the specialist lock follows after
   the backbone probe chooses its framework.
3. Add gitignore rules for downloaded images, teacher caches, specialist
   checkpoints, Core ML build products, and run artifacts.
4. Preserve the existing v7b 325-dish and 29-negative results as the historical
   baseline.
5. Extend evaluation output as needed so every system preserves dish ID,
   target, raw/parsed prediction, failure type, confidence, and latency.
6. Add a failure-aware paired comparison with calorie bins, p75/p90/p95,
   signed bias, and bootstrap confidence intervals.

Deliverable: reproducible environment and frozen evaluation specification.

### Phase 1 — licensed data acquisition and manifests

1. Acquire NutritionVerse-Real and verify its expected image/dish counts and
   checksums.
2. Acquire the Food Recognition Benchmark 2022 v2.1 public training release
   and retain its original 323-class labels and MS-COCO polygons.
3. Build a 5,000-image class-stratified pilot with extra weight on mixed dishes,
   bowls, sauces, fried food, and rare classes.
4. Select two to four usable side frames per measured training dish, resize
   without distorting aspect ratio, and group all frames by dish.
5. Run exact/perceptual deduplication before producing manifests.
6. Emit a machine-readable provenance/license inventory.

Gate: no image without traceable source and permitted pilot use enters a
training manifest.

Execution note (2026-07-28): the user-provided 4.9 GB archive is an FRB v2.0
Deep Lake mirror, verified at 39,962 train images, 76,491 annotations, and 498
classes. Its SHA-256 is
`00afdf97a392e6baaf6e1bd13c5487917227b8e4030847f804c79ab165aa7879`.
It is acceptable as the representation-learning fallback under the original
AIcrowd 2022 CC BY 4.0 grant, but every derived record is marked `v2.0`; it is
never reported as v2.1. The selected pilot covers all 498 classes, has at least
eight images per class, and passed JPEG integrity plus exact/dHash≤2 duplicate
gates after three exact duplicate pairs were replaced.

#### Food Recognition Benchmark 2022 license posture

The v2.1 public release replaces the older MyFoodRepo-273/v0.4 source in this
plan. The 2022 challenge rules explicitly identify the dataset as CC BY 4.0 and
do not contain the old competition-only or cross-referencing clauses. AIcrowd's
general Terms of Use also state that content made available under a Creative
Commons license is governed by that license.

Before downloading, preserve timestamped local snapshots and hashes of:

- the 2022 challenge rules;
- the dataset page/file manifest;
- AIcrowd Terms of Use and Participation Terms;
- the CC BY 4.0 legal code;
- any click-through text shown during account-gated download.

The public training archive is used only for local R&D in this phase. We do not
submit a competition entry, redistribute images, attempt identity linkage, or
publish derived weights. Entry-specific licenses to the organizers apply if a
model is submitted to the challenge; no submission is in scope. Attribution
and the final provenance inventory remain mandatory.

Sources:

- [Food Recognition Benchmark 2022 rules](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022/challenge_rules)
- [Food Recognition Benchmark 2022 dataset description](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022)
- [AIcrowd Terms of Use](https://www.aicrowd.com/terms)
- [AIcrowd Participation Terms](https://www.aicrowd.com/participation_terms)
- [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)

### Phase 2 — teacher bake-off and semantic enrichment

Teacher labels are limited to visible semantic facts:

- normalized food/ingredient presence;
- cooking state;
- container type;
- visibility/occlusion and mixed-dish flags;
- ambiguity/abstention;
- optional prompts that can be reconciled with supplied polygons.

Teachers are explicitly prohibited from supplying training targets for
calories, macros, grams, or hidden recipe ingredients.

1. Build a 325-image bake-off from training/validation sources—not the
   historical test—including source-labeled and difficult examples.
2. Run the same schema against Gemini Flash, Flash-Lite, and a small
   local-Qwen sample.
3. Score source-label agreement, normalized ingredient precision/recall,
   hallucination rate, abstention, JSON validity, latency, and actual cost.
4. Choose the cheapest teacher within 1–2 ingredient-F1 points of the best,
   provided hallucination is also within the acceptance bound.
5. Label the 5,000-image pilot with Gemini Flash and Flash-Lite. These are
   useful agreement checks but are not independent model families, so source
   metadata remains authoritative.
6. Send only disagreements, rare classes, and a random audit sample to local
   Qwen3.6-27B. Do not run local Qwen over the full pilot.
7. Accept enrichment only when the Gemini outputs agree after taxonomy
   normalization, do not conflict with source metadata, and survive the audit.
   Otherwise retain the source label and omit the disputed field.
8. Visually audit contact sheets containing at least 100 random records and
   100 disagreements/hard cases.

Execution result (2026-07-28): the 325-image bake-off covered all 498 v2.0
classes. Both models returned 325/325 schema-valid records with no abstentions
or false not-food outputs. The source-label lexical screen favored Flash-Lite:
F1 0.565 and unmatched-prediction rate 44.1%, versus Flash F1 0.486 and 52.3%.
Cross-model label agreement was 0.786. Flash-Lite cost $0.166581; Flash cost
$0.578818. Because Lite was both materially better on this extraction prompt
and about 3.5× cheaper, the remaining 4,675 records use Lite only. Flash is
retained for targeted disagreement audits rather than duplicated full-corpus
labeling. Source annotations remain authoritative, and unmatched teacher names
are recorded but excluded from accepted enrichment.

Full-pilot result: Flash-Lite returned 5,000/5,000 valid records across eleven
resumable jobs with zero API failures. It used 6,248,543 input and 928,782
output tokens and settled at $2.098259. Including the $0.578818 Flash bake-off,
total Google spend is $2.677077; no reservation remains open and OpenAI spend
is zero. The merged manifest accepted 10,303 source-compatible visible names,
quarantined 7,179 unmatched names, recorded 16 abstentions, and added zero
calorie, gram, macro, or hidden-ingredient targets.

Pilot planning cost: approximately $15–20 for Gemini 3.5 Flash plus Flash-Lite
on all 5,000 images, including a small contingency for the bake-off/retries.
Set a $25 Google-only cap for this phase. OpenAI is not needed unless the
Gemini audit fails or a later full-corpus scale decision explicitly adds an
independent API provider.

#### Budget enforcement

API billing is finalized after a request or asynchronous batch completes, so
the labeling runner uses both settled spend and conservative reservations:

```text
committed spend = settled API-reported usage
                + reservations for submitted unfinished batches
                + reservation for the next proposed batch
```

Before submitting anything, the runner rejects the batch if committed spend
would exceed its provider cap or the automated total ceiling. Reservations use
the bake-off's p95 observed image/input/output usage, current pinned list
prices, and a 20% usage buffer. Until the bake-off exists, use the more
conservative planning estimate. Retries are new spend and require a new
reservation; they are never invisible.

The overall project envelope may remain $100, but the first Gemini-only pilot
uses a separate $25 active cap and a $5 safety reserve. Thus ordinary
automation stops at $20 of committed API usage. Raising the active cap for
full-corpus labeling or enabling another provider requires an explicit
decision after reviewing the pilot ledger. The margin covers tokenization
drift, failed records, and price/billing differences; taxes and currency
conversion remain provider/account-level charges rather than token usage.

Every request/batch appends a record to the gitignored:

```text
ml/runs/teacher-labeling/<run-id>/budget-ledger.jsonl
```

Each record includes timestamp, provider, exact model/version, pricing snapshot
hash, batch/request ID, image count, status, input/image/output/reasoning token
usage when supplied, estimated/reserved USD, settled USD, retry relationship,
and cumulative provider/total spend. The run also maintains:

- `budget-summary.json` for a human-readable current balance;
- `pricing.json` with price date and official source URLs;
- raw provider usage metadata for reconciliation;
- a list of submitted-but-unsettled batch IDs.

On resume, the runner reconstructs committed spend from the append-only ledger
and queries/reconciles unfinished batches before allowing new submissions. A
missing/corrupt ledger, unknown model price, unavailable usage metadata, or
provider mismatch is fail-closed: no new paid call is submitted.

Deliverable: versioned raw and normalized teacher caches plus an audit report.

### Phase 3 — specialist controls and pilot training

Run matched experiments with the same measured split:

| Run | Training data | Purpose |
|---|---|---|
| S0 | Measured overhead only | Numeric specialist baseline |
| S1 | Measured overhead + side views | Value of measured viewpoint diversity |
| S2 | S1 + raw FRB 2022 labels | Value of public images without teachers |
| S3 | S1 + teacher-enriched public labels | Incremental value of distillation |

For each run:

1. overfit 32 measured dishes to validate losses and masks;
2. benchmark 200–500 images to measure throughput and memory;
3. train the full pilot with early stopping on validation only;
4. calibrate quantile coverage/confidence on validation;
5. export to Core ML and compare Mac predictions before and after conversion;
6. persist per-record outputs and the exact training manifest/hash.

Expected M3 Ultra time:

- 10–30 minutes for frozen-backbone probes;
- 45–120 minutes for a serious partial-fine-tune run;
- about 4–6 hours for the initial matched pilot matrix;
- longer only if full-backbone or segmentation training earns a follow-up.

### Gate A — does the specialist add useful information?

Proceed to conditioned-Qwen training only if validation shows at least one of:

- specialist calorie MAE is at least 5% better than Qwen's matched validation
  result; or
- a pre-registered linear/late-fusion control improves calorie MAE by at least
  5%, demonstrating complementary residuals.

Also require:

- no new catastrophic p90/p95 tail;
- calibrated 80% intervals with observed coverage reasonably close to target;
- S2 or S3 improves phone-domain semantic metrics over S1;
- teacher enrichment S3 beats raw-label S2 before teacher labels are credited;
- successful Core ML conversion with numerically close outputs.

If Gate A fails, stop. Do not spend 3.5 hours training a conditioned LoRA around
an uninformative auxiliary signal.

### Phase 4 — leakage-safe specialist predictions

Qwen must train on realistic specialist mistakes, never on specialist
predictions from a model that trained on the same dish.

1. Create five folds grouped by dish.
2. For each fold, train the winning specialist on the other four folds. Public
   semantic records may be shared, but every image/view of the held-out
   measured dish is excluded from numeric training.
3. Infer the held-out fold and concatenate predictions until every measured
   Qwen training dish has one out-of-fold auxiliary record.
4. Train a final deployment specialist on all permitted training dishes.
5. Infer validation/test/OOD only with a specialist that has never trained on
   those dishes.
6. Store model hash, fold, prediction intervals, and failure state with every
   auxiliary record.

Expected compute: roughly 6–10 hours for five folds plus the final specialist,
serialized on the GPU.

### Phase 5 — Qwen 2B augmentation screen

Before conditioned training, measure whether the smaller model can learn the
existing task and whether augmentation helps it. Use the same measured
train/validation split and example budget:

| Run | Base | Visual adaptation | Training images | Purpose |
|---|---|---|---|---|
| Q4-control | 4B | Existing language-only LoRA | Clean | Shipping v7b reference |
| Q2-clean | 2B | Language-only LoRA | Clean | Isolate model-size effect |
| Q2-aug | 2B | Language-only LoRA | Conservative augmentation | Isolate augmentation value |
| Q2-visual | 2B | Projector/final-vision-block LoRA if feasible | Same augmentation | Test whether saved memory enables useful visual adaptation |

First run all new arms through the 32-dish overfit and matched validation probe.
Only a promising 2B arm receives a full run. A small matched 4B augmentation
probe may be added if Q2-aug wins, so augmentation is not incorrectly credited
to model size.

Augmentations are training-only, deterministic from record ID/epoch/seed, and
applied before the standard Qwen processor:

- mild exposure, contrast, saturation, white-balance, and hue changes;
- mild sensor noise, blur, sharpening, and JPEG compression;
- small rotations and translations;
- horizontal reflection;
- scale/crop only when at least 90% of the original meal remains visible;
- resized measured side views as true viewpoint augmentation.

Do not use vertical flips, MixUp, CutMix, mosaics, large crops, random erasing,
or strong perspective warps. Those can remove ingredients or change apparent
portion geometry while retaining an invalid nutrition target.

The augmentation implementation must verify that:

- the original image is never overwritten;
- validation/test images remain byte-identical;
- all transformed images still reach the vision encoder;
- prompt bytes and completions do not change;
- a saved contact sheet makes the augmentation strength auditable.

The 2B arm graduates to conditioned training if it has valid JSON/refusal
behavior and is either within 10% of the 4B validation calorie MAE or shows a
clear augmentation/visual-adaptation gain that plausibly closes the gap once
conditioned. This is a screening rule, not a shipping allowance.

Expected M3 Ultra time: benchmark first; planning estimate is 1.5–2.5 hours for
a full 2B LoRA run, plus approximately 1.7 GB for the downloaded 4-bit base and
less than 1 GB per retained adapter/checkpoint set.

### Phase 6 — v8 prompt contract and conditioned LoRAs

1. Define one versioned auxiliary schema and canonical formatter.
2. Make training, Mac inference, and Swift fixtures render identical bytes.
3. Generate `finetune_data_v8` from the existing measured completions plus
   out-of-fold auxiliary predictions.
4. Keep v7b's 100 negative-training dose. Run the specialist over negatives or
   use the canonical unavailable form; the completion remains
   `{"not_food": true}`.
5. Add robustness examples based on the empirical out-of-fold error
   distribution:
   - approximately 15% unavailable auxiliary records;
   - approximately 20% plausibly perturbed estimates;
   - confidence reduced when perturbation grows;
   - no perfect ground-truth auxiliary values.
6. Extend prompt-parity and smoke tests to v8. They must pass before weights
   are loaded.
7. Run the established 32-dish overfit/probe ladder.
8. Train a fresh 4B rank-16/alpha-32 two-epoch conditioned LoRA using the
   established completion-only recipe unless the probe gives evidence to
   change it.
9. If the Phase 5 screen passes, train a matched 2B conditioned LoRA using the
   winning conservative augmentation/visual-adaptation configuration.

Expected 4B LoRA time: approximately 3.5 hours, with 47–58 GB peak Metal
memory. The 2B time and memory are measured rather than inferred from parameter
count. No Qwen run shares the GPU with another workload.

### Phase 7 — matched evaluation

Evaluate up to six systems on identical dish IDs:

1. shipping v7b;
2. specialist alone;
3. v7b plus deterministic late-fusion control;
4. best 2B augmentation-only model;
5. 4B v8 conditioned on specialist predictions;
6. 2B v8 conditioned on specialist predictions, if its screen passed.

Suites:

- 325 historical Nutrition5k dishes;
- NutritionVerse-Real grouped by dish;
- 29 held-out non-food images;
- auxiliary robustness variants: unavailable, low confidence, and
  validation-realistic perturbations.

Metrics:

- failure-aware calories/macros MAE and median;
- p75/p90/p95, bias, relative error, and calorie bins;
- total/per-item gram error where labels exist;
- ingredient precision/recall/F1 and item-count error;
- refusal recall, food over-refusal, and schema failures;
- uncertainty calibration and error by disagreement bin;
- paired bootstrap confidence intervals;
- latency, peak memory, model size, and Core ML conversion drift.

Research graduation requires:

- at least 5% paired calorie-MAE improvement on the historical suite, or a
  clearly better high-calorie/p90 tail without median regression;
- directionally consistent improvement on NutritionVerse-Real;
- no meaningful ingredient-recall regression (provisional bound: 2 points);
- 29/29 held-out refusal recall and zero food over-refusals;
- no increase in parse/schema failures;
- graceful degradation when auxiliary data is unavailable or corrupted.

A conditioned 2B model is preferred over conditioned 4B only if it clears the
same accuracy/refusal/OOD gates and is statistically tied or better on the
primary metrics. If 2B is slightly worse, report the exact accuracy-memory-
latency tradeoff for a product decision rather than silently weakening the
accuracy bar.

A shipping recommendation additionally requires the paired evidence to exclude
a meaningful regression, acceptable device memory/thermal behavior, and a
target-domain device check.

### Phase 8 — iOS prototype

Only after the Mac evaluation gates pass:

1. convert and quantize/palettize the specialist for Core ML;
2. add a small specialist runner behind a protocol and feature flag;
3. run it sequentially before Qwen to limit peak memory;
4. render the canonical auxiliary block in Swift;
5. retain the existing Qwen image input and completion schema;
6. add cross-language byte-parity fixtures;
7. keep specialist failure non-fatal via the trained unavailable form;
8. measure added latency, peak memory, package size, and thermal behavior;
9. run all Swift, ML, and iOS build checks.

Provisional device budgets:

- specialist package: target ≤100 MB after compression;
- added specialist latency: target ≤1 second on iPhone 15 Pro or newer;
- added peak memory: target ≤250 MB and no jetsam/thermal regression;
- total user-visible inference remains within the existing 15–30 second
  product envelope.
- record total package size, steady/peak memory, time-to-first-token, decode
  rate, and end-to-end latency for both 2B and 4B finalists.

These are engineering defaults, not product commitments, and can be changed by
the user before the device gate.

### Phase 9 — scale decision

Do not label all 54,392 FRB 2022 v2.1 public training images merely because
budget remains.
Scale only if S2/S3 and v8 demonstrate measurable value.

If the pilot wins:

1. expand to all licensed FRB 2022 v2.1 public training images;
2. retain source-balanced batches;
3. under the normalized planning assumptions, run Flash-Lite on all images
   (about $33) and Flash only on the 10–20% disagreement/hard subset (about
   $13–27), remaining inside a roughly $65 Google cap;
4. use local Qwen only for disagreements, rare classes, and failed API records;
5. add a paid second provider only if the pilot audit demonstrates a material
   need and the user explicitly approves it;
6. retrain the winning specialist configuration;
7. regenerate leakage-safe out-of-fold predictions;
8. train a versioned successor rather than overwriting v8;
9. repeat every evaluation and device gate.

## Resource and recovery plan

The Mac is an M3 Ultra with a 60-core GPU and 96 GB unified memory.
GPU-heavy jobs are serialized. Every long run writes:

- configuration and manifest hash before launch;
- append-only progress/checkpoints;
- resumable record-level teacher caches;
- final per-record predictions and metrics;
- failure reason and last completed unit.

Estimated pilot wall-clock after data and credentials are available:

| Work | Estimate |
|---|---:|
| Data ingestion, manifests, deduplication | 0.5–1 day |
| Teacher bake-off and 5k batch labeling | hours to 1 day, provider-dependent |
| Specialist implementation and control matrix | 1–2 days |
| Five-fold OOF + final specialist | 6–10 GPU hours |
| 2B augmentation screen | 2–5 GPU hours depending on probe result |
| v8 data/parity/probe/full training | 5–10 GPU hours for one or two sizes |
| Full v8 evaluation | ~1.5 hours per Qwen arm plus OOD |
| Core ML/iOS integration and tests | 1–2 days after accuracy gates |

As verified on 2026-07-28, the current disk has about 326 GB free. Both the 5k
pilot and a later 54k expansion have comfortable working headroom. Recheck
before every scale-up and stop if free space would fall below 60 GB. Existing
adapters and datasets are never cleanup candidates.

## Inputs and current execution state

### Needed before paid labeling or long training

1. **Paid API authorization**
   - Confirm that the first Gemini-only pilot may spend up to **$25**.
   - The broader experiment may retain the user's **$100 total** envelope, but
     increasing the active cap beyond $25 requires a pilot review.
   - The pilot runner holds back a **$5 safety reserve**, so it stops at $20
     committed usage unless the user explicitly releases part of the reserve.
   - A budget statement is not treated as permission to exceed any cap.

2. **API credentials**
   - Gemini API key for the primary bake-off/teacher.
   - No OpenAI key is needed for the first pilot. OpenAI remains an optional
     later escalation only if Gemini quality or correlated errors justify it.
   - Put secrets and the explicit caps in the following absolute path:

     ```text
     /Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/.secrets/teacher-labeling.env
     ```

   - The file is dotenv syntax: UTF-8, one `NAME=value` per line, no `export`,
     and no spaces around `=`. Do not paste keys into chat or commit the file.
     Current broader-envelope shape (keys and values are never logged):

     ```text
     GEMINI_API_KEY=...
     TEACHER_BUDGET_TOTAL_USD=100.00
     TEACHER_BUDGET_GOOGLE_USD=65.00
     TEACHER_BUDGET_OPENAI_USD=35.00
     TEACHER_BUDGET_RESERVE_USD=10.00
     ```

   - An `OPENAI_API_KEY` may already be present for unrelated work. Presence is
     not authorization: the committed pilot config sets OpenAI's cap to zero
     and enables only Google, so this run cannot submit OpenAI work.
   - `ml/teacher_labeling/configs/gemini_5k_pilot.json` narrows the values
     above to total/Google/OpenAI caps of $25/$25/$0 and holds back $5. The
     effective automatic ceiling is therefore $20.
   - Set file permissions to owner-only (`chmod 600`). The loader may report
     missing variable names, but must never log variable values.

3. **Dataset access**
   - Confirm local research use of the Food Recognition Benchmark 2022 v2.1
     public training release under its recorded terms is acceptable.
   - If the download is account-gated, either provide the already-downloaded
     archive path or complete the provider's login/terms step when requested.
     No account password should be placed in the repository or chat.

   - Resolved with the v2.0 fallback archive described above. A future v2.1
     archive remains useful but is no longer required to run the pilot.

4. **Disk space — resolved**
   - About 326 GB is free as of 2026-07-28.
   - No build-cache deletion is required. Existing build products, datasets,
     adapters, run results, and source files remain untouched.

5. **GPU-use window**
   - Confirm roughly 24–36 hours of exclusive GPU time, which may be split
     across multiple unattended windows, in which Codex may serialize
     specialist folds, the 2B screen, Qwen training, and evaluation.
   - LM Studio and other GPU-heavy applications must remain closed. Codex will
     detect conflicts and wait/fail safely; it will not kill the user's
     processes without separate permission.

6. **Pilot use posture**
   - Confirm the first pass is local R&D: do not redistribute dataset images,
     publish derived weights, or ship a new adapter without a separate user
     decision after license and quality gates.

### Needed only at the final device gate

7. **Target hardware**
   - Default: iPhone 15 Pro or newer, matching the repository's current
     recommendation.
   - Provide a different minimum device if the product must support it.
   - No up-front preference between 2B and 4B is required; measured accuracy,
     memory, and latency will be reported before that product decision.

8. **Device-domain check**
   - Access to the target iPhone and a small set of ordinary meal photos for
     latency/memory and qualitative checks.
   - Because the user currently has no capacity to collect data, this is
     explicitly deferred. NutritionVerse-Real can graduate the research
     prototype, but cannot by itself authorize shipping.

### Decisions Codex can make without interrupting the user

Unless a gate fails or scope changes, Codex can independently choose:

- backbone winner using validation accuracy, Core ML compatibility, and size;
- the winning conservative augmentation strength from clean validation;
- image resolution, batch size, learning rate, and stopping point from probes;
- teacher winner using the pre-registered food-label metrics and spend cap;
- taxonomy normalization and loss weights;
- public-data sampling and rejection thresholds;
- fold assignment, deterministic seeds, and checkpoint cadence;
- whether teacher enrichment is discarded in favor of raw source labels;
- whether to stop before v8 because Gate A fails.

## Stop and escalation conditions

Pause and report instead of improvising if:

- a dataset's license or access terms are ambiguous;
- projected paid usage would cross a provider or total cap;
- free disk would fall below 15 GB;
- an operation would require deleting an existing dataset/model/adapter;
- a GPU conflict persists or training repeatedly OOMs;
- teacher hallucination or disagreement exceeds the audit bound;
- Core ML conversion materially changes predictions;
- Gate A fails;
- v8 regresses refusal, ingredient recall, tail error, or OOD accuracy;
- a shipping/publishing decision is reached.

## Completion definition

The pilot is complete when the repository contains reproducible scripts,
manifests, prompt/parity tests, specialist and v8 run metadata, paired
evaluations for every graduated 2B/4B arm, a Core ML feasibility result, and a
written go/stop recommendation. Shipping weights, public release, and new
user-collected data are separate decisions.
