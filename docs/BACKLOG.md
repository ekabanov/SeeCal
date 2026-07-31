# SeeCal backlog

Deferred work items. Not scheduled; picked up when prioritized. Dated when added.

## Correction-first meal review (implemented 2026-07-30; usability pilot pending)

The first product pivot from exact reconstruction toward fast acceptance is now
implemented. Food names open a five-choice, database-backed replacement tray;
replacements keep the estimated amount while changing nutrition density; each
row has inline ±15 g controls; and the complete editor remains under **Edit
details**. Privacy-safe local instrumentation measures active review time,
actions, corrections, keyboard use, and save/discard/dismiss without recording
photos, food names, or nutrition.

**Next step:** run 10–15 representative meals with at least five people,
counterbalanced against the previous form-first flow. Measure median/P90 active
review time, keyboard rate, corrections-to-save, and abandonment. Do not add
retrieval-first food additions, hidden-ingredient shortcuts, personalization,
or another expensive LoRA solely from intuition; let the observed costly paths
choose the next slice. Plan and provisional gates:
`docs/plans/2026-07-30-correction-first-ux-plan.md`.

## Hint-assisted photo re-analysis (added 2026-07-28)

**Recovery and proactive slices implemented 2026-07-31:** the factored pipeline now raises a
structured human-input request when deterministic nutrition resolution misses.
The app keeps the same photo and recognized labels, asks for broad context, and
re-runs IDENTIFY with a whitespace-normalized, bounded hint. A repeated miss
stays in the neutral help loop; hints and food names remain out of diagnostics
and are not persisted. A fresh successful result also exposes **Fix estimate
with a hint** in its summary and after the five local replacement choices. The
original draft remains intact until the same-photo retry succeeds; failure
returns to the hint panel. The byte-identical no-hint prompt is unchanged. The
paired hint-quality evaluation below remains open.

**Goal:** when the photo is usable but the model identifies a food incorrectly,
let the user add a short fact such as “the white cubes are tofu” or “this is
turkey, not chicken” and re-run nutrition estimation on the same photo.

**Why this is more useful than renaming an item:** editing already lets the user
rename “chicken” to “tofu”, but a rename deliberately preserves the current
nutrition density. That is correct for ordinary corrections, but not when the
identity error also makes calories/macros wrong. A hinted re-run gives the model
the opportunity to recompute the whole estimate using the corrected identity.

**Prompt compatibility remains load-bearing:** the app does not append an
unstructured `User hint` line. Both hint entry points use the same bounded,
schema-reinforcing envelope, while ordinary scans retain the byte-identical
trained prompt. The current adapter was not trained on appended hints, so the
paired evaluation below is still required to measure JSON reliability, food
accuracy, and not-food refusal. Every hint envelope must remain represented
exactly in parity tests and any future hint-conditioned training.

**Implemented product shape:** one short field with examples re-analyzes the
same stored photo. The current draft—including user edits—remains unchanged
until success, when the complete model estimate replaces it. The panel tells
the user this explicitly. Failure preserves the current draft. Whitespace and
control characters are normalized, input is capped at 240 characters, and the
hint is not retained after logging.

**Training/evaluation approach:**
1. First run a paired zero-shot pilot on `adapters_v7b`: the same 50 diverse food
   images with no hint and with a correct, ground-truth-derived hint. Include
   ambiguous identity hints, a deliberately wrong hint set, and held-out
   not-food images with food-sounding hints. This tells us whether the existing
   adapter can use hints at all; it is not a ship gate by itself.
2. If zero-shot is unstable, make a teacher-generated hint-conditioned dataset
   variant. Keep most train records byte-identical to v7b and add a short hint
   to a deterministic minority (feature dropout). An initial mix worth testing
   is 75% unchanged/no-hint and 25% hinted: roughly 15% positive identity
   (“the cubes are tofu”), 7.5% contrastive correction (“tofu, not chicken”),
   and 2.5% preparation or otherwise visually useful clarification. Treat those
   percentages as an ablation parameter, not a product constant.
3. Extend `check_prompt_parity.py` and Swift prompt tests to cover no-hint and
   every supported hint envelope. The no-hint string must remain byte-identical
   to v7b.
4. Gate on paired results: no-hint food MAE/parse rate remain a statistical tie
   with v7b; correct hints improve ingredient identification and do not worsen
   calorie MAE; incorrect hints fail gracefully; food-sounding hints do not
   materially bypass v7b’s not-food refusal.

### Teacher-generated synthetic hints

The teacher should not freely relabel the image. Give it the photo **and** the
authoritative Nutrition5K ingredient names, and ask it to express one sparse fact
that a user could realistically type. The measured/source list remains the truth;
the teacher supplies wording and, for contrastive hints, a plausible visual
confusion. This is safer than asking a teacher to infer the hint from the image
alone, which can reproduce exactly the recognition error the hint is meant to
correct.

The strongest version also gives the teacher `adapters_v7b`’s baseline item names
for that **train-split** image. When the prediction conflicts with the measured
ingredient list, ask for the smallest hint that corrects the actual error:
baseline “chicken” + source “tofu” becomes “the cubes are tofu, not chicken”.
Prefer these real student-error corrections over teacher-invented confusions.
Use ordinary positive hints on correctly recognized records so every hinted
example does not imply that a contrastive correction is required.

Use a structured teacher response, for example:

```json
{
  "abstain": false,
  "kind": "contrastive_identity",
  "text": "the cubes are tofu, not chicken",
  "supported_items": ["tofu"],
  "excluded_items": ["chicken"]
}
```

Fail closed unless every supported item maps to the dish’s source ingredients,
every excluded item is absent, the hint contains no calories/macros/grams, and
the text is short enough for the app’s limit. Ask for at most one or two facts;
do not let the teacher paste the complete ingredient list into every prompt or
Qwen will learn a synthetic shortcut unlike real user behavior. `abstain` is the
right answer when no useful, visually grounded hint can be made.

Reuse the existing `ml/teacher_labeling/` batch, budget, raw-response, prompt-hash,
and provenance machinery. The current 5,000-image semantic bake-off found
Gemini Flash-Lite cheaper and better on the registered extraction screen, so it
is the natural first generator to test; hint quality still needs its own
200-record audited pilot before scaling. Store the teacher model/version, exact
prompt and schema hashes, raw result, normalized hint, source item mapping, and
accept/reject reason with each record.

Apply hint selection deterministically by record/dish ID so datasets regenerate
exactly. Replace the prompt on the selected 25% rather than duplicating them,
which would otherwise overweight those dishes. When a dish has multiple views,
mix hinted and unchanged views where possible so the model sees both conditions
without changing dish-level split isolation. Within the hinted slice, prioritize
records where v7b’s normalized item names disagree with the source list, then
fill the remaining quota with correctly recognized but visually useful examples.
Store the baseline output and matching decision as provenance; teacher wording
must never override the measured label.

Teacher-generate training hints from the **train split only**. After the
generation prompt and acceptance rules are frozen, create validation hints
separately. Keep the final paired hint gate human-written or at least
human-audited, with wording not produced by the training teacher, so success is
not merely adaptation to one model’s synthetic prose.

**First step:** add an offline paired-hint evaluation mode to `infer.py` and a
200-record teacher-hint preparation/audit mode alongside the existing semantic
teacher pipeline; do not start with UI work or a full paid labeling run. This is
an ML behavior experiment with a small UI tail, not just a text-field feature.

## Completed: Add a meal manually, with no photo
(added 2026-07-28; completed 2026-07-28)

**Status:** shipped. Manual opens a photo-less editable draft through the
shared result/edit flow. Persistence, totals, editing, deletion, and photo
cleanup all handle the missing image path. The material below is retained as
the implementation rationale, not remaining work.

**Goal:** allow logging when the user has no photo, already knows the nutrition,
or wants to enter a snack quickly. No inference and no network are involved.

**Existing leverage:** manual **ingredients** already work inside the shared
result/edit sheet and persist with `modelEstimate == nil`. What is missing is a
way to create the containing meal without first taking a photo.

**Recommended product shape:**
- Add a visible “Add meal” action to Today’s MEALS section. Keep the center Scan
  FAB as the one-tap camera path; do not add an action sheet to every scan.
- Start a new draft at the current time and inferred meal slot, immediately open
  one blank “Added manually” ingredient editor, and reuse all current validation,
  density scaling, add/delete/Undo, meal-name editing, and Log meal behavior.
- A cancelled first ingredient may return to an empty draft, but Log stays
  disabled until there is at least one valid item. Blank nutrition values remain
  zero, matching the shipped manual-ingredient contract.
- Use a neutral utensil/manual glyph wherever a photo thumbnail would appear.
  Do not manufacture a placeholder image file.
- A later extension can make date/time editable for back-filling older meals;
  keep v1 at “now” unless that need is prioritized separately.

**Required model/persistence work:**
- Make `MealLogEntry.imagePath` optional and decode all existing string-valued
  entries unchanged. Update thumbnail, result-header, deletion, and photo-cleanup
  paths to handle `nil`.
- Add a manual-new-meal initializer to `MealEditDraft` rather than fabricating a
  `FoodScanResult` or fake scan context.
- Consider an entry-level origin (`photo`, `manual`, later `barcode`) only if the
  UI or analytics needs it; absence of a photo is sufficient for the v1 behavior.

**Acceptance gates:** manual meals round-trip through persistence, contribute to
Today/History totals, can be edited and deleted, never touch the inference
runtime, and do not attempt photo deletion. Old meal JSON remains compatible.

**Effort/value:** small-to-medium and high-confidence. This should land before
barcode entry because barcode results need the same photo-less meal path.

## Completed: Add packaged food by barcode
(added 2026-07-28; completed 2026-07-28)

**Status:** shipped. EAN/UPC/Code 128 detection shares the camera capture
session, normalizes GTINs, resolves through a cached Open Food Facts v3 lookup,
and keeps label nutrition source-backed and resettable. Missing fields fall
back to manual entry without invented zero values. A local launch-market
coverage study and any richer liquid-unit model remain possible follow-ups;
the core barcode feature is no longer backlog.

**Goal:** scan a UPC/EAN/GTIN on packaged food, retrieve its label nutrition,
enter the amount actually consumed, and add it as a source-backed meal item.
This complements photo estimation; it does not help restaurant or unpackaged
food.

**Important distinction:** a normal retail barcode is primarily a product
identifier, not a nutrition payload. SeeCal must query a product database, and
the package quantity is not the consumed quantity. The user still has to confirm
grams, millilitres, or servings eaten.

### Data-source survey

**Recommended v1 source — Open Food Facts.** Its current v3 API can retrieve a
product directly by barcode, read access needs no API key, and records can include
product name, ingredients, serving size, quantity, and normalized nutrition per
100 g / 100 ml or serving. It is global and free, but crowdsourced, incomplete,
and explicitly provides no accuracy guarantee. Product reads are currently
limited to 15 requests/minute/IP, which is ample for direct user scans.

- API and limits:
  [Open Food Facts API introduction](https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/)
- Barcode endpoint:
  [GET product by code](https://openfoodfacts.github.io/documentation/docs/Product-Opener/v3/products/get-api-v3-product-code/)
- Nutrition field semantics:
  [product nutrition schema](https://openfoodfacts.github.io/documentation/docs/Product-Opener/schemas/schemas/product_nutrition/)
- Licensing: the database is ODbL, individual contents are DbCL, and product
  images are separately CC BY-SA. Follow their attribution/reuse guidance and
  review compliance before shipping:
  [Open Food Facts licensing guide](https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/tutorials/license-be-on-the-legal-side/)

**Possible later fallback — USDA FoodData Central.** Its data is public domain
and its Branded Foods data includes GTIN/UPC, but it is US-oriented and every API
request requires a data.gov key that the USDA says must not be made public. The
default limit is 1,000 requests/hour. A production integration therefore needs a
SeeCal backend/proxy or a periodically prepared local subset; embedding the key
in the iOS app is not acceptable. That conflicts with the present no-backend
architecture, so it is not recommended for v1.

- [FoodData Central API, key, limits, and licensing](https://fdc.nal.usda.gov/api-guide/)
- [FoodData Central downloadable datasets](https://fdc.nal.usda.gov/download-datasets/)

**Not a nutrition source — Verified by GS1.** It is useful for verifying that a
GTIN is valid and who assigned it, with some basic product data when available.
Its public service is limited to 30 single queries/day and enterprise/API access
is arranged through a local GS1 office. It is not a general, open nutrition
database and is not the right primary lookup for SeeCal.

- [Verified by GS1 overview](https://www.gs1.org/services/verified-by-gs1)

### Recommended v1 architecture and behavior

1. Land photo-less manual meals first, then add a “Scan barcode” route alongside
   “Add manually” from Today’s Add meal flow. Use Apple VisionKit’s
   `DataScannerViewController` for live barcode recognition where supported,
   with a clear unavailable/manual-code fallback.
2. Normalize and checksum-validate the GTIN, then call only Open Food Facts’
   product-by-code endpoint through a `BarcodeProductLookup` protocol. Send an
   identifying User-Agent as their API requests recommend.
3. Request only the needed fields and consume normalized
   `energy-kcal_100g`, `proteins_100g`, `fat_100g`, and
   `carbohydrates_100g` values. Treat a missing product or any missing required
   macro as incomplete: show what was found and let the user finish it manually;
   never silently turn missing fields into trusted zeroes.
4. Ask for consumed amount before adding. Prefer the package serving as a
   convenience default only when it has a normalized quantity; never assume the
   whole package was eaten.
5. Address liquids explicitly. Open Food Facts’ normalized `_100g` fields mean
   per 100 g **or per 100 ml for liquids**, while SeeCal currently models every
   amount as grams. Either extend `MealItem` to carry `g`/`ml`/`serving` units,
   or constrain the first release to mass-based products. Do not silently equate
   1 ml with 1 g.
6. Store a nutrition snapshot in the meal so old logs do not change when the
   community record changes. Also store barcode, provider, lookup timestamp, and
   the immutable imported reference so Reset can return to the scanned label.
   The current `modelEstimate` concept should evolve into a source-neutral
   reference (`model`, `barcode`, or none/manual) instead of pretending barcode
   data came from Qwen.
7. Cache successful lookups by barcode on-device for speed and offline reuse.
   On a cache miss without connectivity, offer code entry/manual nutrition; do
   not block meal logging.
8. Omit remote product images in v1. This avoids unnecessary traffic/storage and
   a separate CC BY-SA image-attribution surface; the meal row can use a barcode
   or package glyph.

**Privacy/product copy:** barcode lookup is not fully on-device. The app sends
the numeric barcode—not the meal photo—to Open Food Facts. Replace absolute copy
such as “nothing is uploaded” with precise language: photo analysis stays on
this iPhone; barcode lookup contacts Open Food Facts. The feature should be
clearly unavailable/offline when the user has no network.

**Data-quality spike before implementation:** scan 50–100 packaged products from
the actual launch market (including Estonian/EU store brands, imported products,
solids, drinks, and serving-only labels). Record barcode hit rate, presence of all
four required nutrition fields, usable serving quantity, and obvious label
mismatches. The global database size is not a substitute for local coverage.
Set a ship/no-ship threshold from that evidence before building the full UI.

**Acceptance gates:** a recognized complete product produces the same totals as
its label for the entered amount; incomplete/not-found/offline cases fall back
cleanly to manual entry; repeat scans use the cache; provider attribution and
privacy copy are visible; no API secret is shipped; existing photo and manual
meals remain unchanged.

## Suggested order for these three

1. **Manual meal without a photo** — lowest risk and creates the shared
   photo-less entry path.
2. **Barcode coverage spike, then Open Food Facts v1 if coverage is good** —
   product-value feature with network, unit, provenance, and licensing work.
3. **Hint-assisted re-analysis experiment** — potentially high value, but its
   real cost/risk is adapter training and regression evaluation, not UI.

## Out-of-distribution test set (added 2026-07-28)

**Status (2026-07-30): dataset, SCALE evaluation, oracle assembly audit, FPB
zero-shot evaluation, and FPB-trained SCALE C1/C2 ablation complete; trained
IDENTIFY/full candidate evaluation pending.**
NutritionVerse-Real v2 is pinned, downloaded, checksummed, and converted into
IDENTIFY, monolith, and SCALE manifests: 889 images across 225 represented dish
groups. The owner authorized non-commercial training: official Train enters
only a separately marked non-commercial SCALE corpus, while official Val
remains frozen. SCALE-v2 reduced official-Val equal-scene mass MAE from about
300 g to 96.2 g with 81.3% calibrated coverage. A corrected FNDDS importer and
official-Train-only reviewed aliases now make all official-Val scenes
assemblable; the first database had silently dropped all 5,430 FNDDS rows
because that release uses legacy nutrient IDs. The full raw oracle also exposed
a corrupt `near-whole-chicken` nutrition template, retained in the primary
score and isolated in a separately labelled Train-derived quality slice. A
validation-selected point correction failed the frozen test and was rejected.
Probe B also failed the frozen FPB test zero-shot: 165.7 g equal-group MAE,
55.3% MAPE, and 20.9% interval coverage on 2,123 clean-weight images / 164
groups. The model predicts small < average < big correctly for only 23/40
complete food triads, and a post-hoc global multiplier cannot rescue it. The
selected FPB-trained C1 checkpoint cuts FPB equal-group MAE to 73.3 g and
repairs ordering to 37/40 triads while keeping frozen N5K/NV regressions below
10%. The subsequent calorie-regret decomposition shows a split constraint:
true-mass non-mass floors are 53.5% of total error on all-complete N5K and
26.7% on clean-72; the NutritionVerse quality slice is 41.0%. Raw
NutritionVerse's 59.7% is retained only as a label-noise stress diagnostic.
The pre-agreed rule therefore resumes IDENTIFY-v2 as reaching-the-floor work,
without claiming SCALE is solved. The 33.92 kcal N5K floor itself attributes
to 20.63 visible-label/exclusion residual, 11.82 rung-1/2 density mismatch,
0.09 rung-3+ mismatch, and 1.39 share bucketing. Prepared-variant alias
re-curation from train/validation is therefore the next resolver lever. Its
signed train/validation audit is now complete: mean residual is
−14.93/−12.60 kcal, but the median is zero and variance contributes 81–85% of
MSE. A uniform train-derived +14.93 kcal correction worsens validation MAE
14.02→19.55 and is rejected. Item-count-keyed train-P90 residual intervals
validate at 92.0% overall coverage and remain shadow-only. A semantics-first
alias pass rejects generic Fish NFS, blocks Caesar salad for lack of a dressed
whole-dish profile, and leaves cooked wheat berry plus Pork NFS provisional:
they improve 31 affected N5K validation groups, but have zero direct coverage
on the NutritionVerse quality slice, so the canonical database is unchanged.

Teacher follow-up diagnostics are complete. FPB is inside the numerical
training target support, but Probe B compresses its visual mass mapping.
NutritionVerse predictions are more coupled to annotated occupancy than truth
is, supporting a framing shortcut. EXIF/intrinsics were stripped from every
NutritionVerse and FPB image. Raw USDA serving priors and all simple fusions
regress Nutrition5K/NutritionVerse and are not an accepted fallback. Probe C
completed as two matched arms with minimax per-source MAPE selection and a 10%
Pareto guard. Center-crop C1 wins: N5K overhead/side 40.1/42.0 g, NV 99.0 g,
FPB 73.3 g. Letterbox C2 scores 37.5/41.0 g, 100.3 g, and 80.0 g respectively,
so its small N5K gain does not offset worse worst-domain MAPE. Naive pooled
phone-union width calibration undercovers NV (70.1%); a max-source
width-normalized union restores NV to 84.0% but is conservative on FPB (97.3%),
confirming the need for the wide-interval confirm path.

**Goal:** measure accuracy on images that look like what users actually shoot.

**Why:** all 325 held-out dishes are Google-cafeteria trays captured by one fixed
RealSense rig — same lighting, same plates, same overhead distance — and the tidy
CSVs are cafe1-only, so even the second cafeteria is excluded. Every number we
quote (v5 59.0, v7b 63.4) is therefore an *in-distribution* number, and nothing
offline tells us how far off real handheld phone photos are. This is the single
biggest gap between our metrics and the product.

**Note the common misconception:** Nutrition5K is NOT one-food-per-photo. 63.3%
of its 4,768 dishes are multi-ingredient (mean 5.71, max 34). The weakness is
scene diversity, not mixed food — so a replacement dataset only helps if it adds
*visual* variety.

**Best candidate: NutritionVerse-Real** (889 real-life images, 251 dishes, 45 food
types; every ingredient weighed and nutrition computed) — the only real-image
option found with weighed ground truth in our exact shape. Small, so it is a test
set first and a fine-tuning supplement second. Others surveyed:
NutritionVerse-Synth (84,984 rendered images, 12 viewpoints, perfect labels — but
synthetic domain gap), MetaFood3D (637 3D objects → render unlimited angles, but
per-object so scenes must be composed), FoodSeg103/154 (9,490 real images, avg 6
ingredients, pixel masks — but **no nutrition or weights**, so ingredient ID only).

**Next step:** finish the active 1,024-pixel-capped FoodSeg IDENTIFY-v2 primary
arm, then run the frozen v3 taxonomy evaluation and four model-versus-oracle
gap slices. Run Probe E after Metal frees. Resolve direct OOD support for the
provisional wheat/pork aliases before any database promotion; do not substitute
raw NutritionVerse or the frozen Nutrition5K test as a selector.

## Further multi-view Qwen training (v9?) (added 2026-07-28)

**Goal:** train on the side-angle cameras as well as overhead, for viewpoint
robustness — real users will not shoot perfectly top-down.

**What's available but unused:** `Nutrition5K/imagery/side_angles/` holds 4
rotating cameras (A–D) × ~29 frames = ~115 frames per dish at 1920×1080 (19GB
already downloaded). `select_images.py` extracts only frame005 from cameras A and
C; B and D are never touched, and 114 of ~115 frames per dish are discarded.
`prepare_finetune.py` already emits one record per *image*, and its docstring
notes this trains viewpoint robustness — the machinery exists.

**The blocker:** a 1920×1080 side frame costs ~2,040 image tokens, against
`train.sh`'s `--max-seq-length 2048`. Such a record would consume the entire
sequence before the prompt or the target JSON, so it would train on nothing.
`--only-overhead` exists precisely for this.

**Approach:** downscale side frames to ~640×480 (≈300 tokens, matching overhead)
in `select_images.py`, then drop `--only-overhead`. Would take training data from
2,594 to ~9,400 records. Raising `--max-seq-length` instead is the expensive path
— runs already peak near 58GB Metal.

**Why it's interesting now:** it is an accuracy/robustness lever pointing the
opposite way to the v7 negatives, which *cost* a little food precision. More food
signal could offset that.

## Completed: Fully editable food data — ingredients, amounts, nutrition
(added 2026-07-27; completed 2026-07-28)

**Goal:** every aspect of a scan result should be user-editable — ingredient
names, the ingredient list itself (add / remove), amounts, and the nutrition
numbers. The model is an estimator, not an authority; when it misidentifies or
misjudges something the user must be able to correct it rather than log a number
they know is wrong (or discard the scan).

**Status (2026-07-28): implemented and verified.** The functional interaction is
binding in
`docs/design/prototype/seecal-prototype.html`, the data and behavior contract is
in `docs/specs/2026-07-26-app-spec.md` §2/§5, and the implementation sequence is
in `docs/plans/2026-07-28-editable-nutrition-plan.md`.

**What shipped:**
- A correction redefines that field's nutrition density at the entered grams;
  later gram changes scale the corrected value.
- Calories, protein, fat and carbs stay independent. No 4/9/4 consistency rule.
- Meal totals always equal the sum of ingredients; there is no total override.
- Added ingredients are manual in v1: blank name, grams, calories and macro fields.
- Compact rows open a focused editor inside the shared result/edit sheet.
- Model-backed items have per-field reset plus Reset item to estimate.
- Ingredient deletion is immediate with a five-second Undo.
- Meal title is editable; logged meals can be deleted from their edit sheet after
  confirmation, including their stored photo.
- Fresh results and previously logged meals use the same editing capabilities.
- Old meal JSON lazily migrates to the new model-backed shape; corrected and
  manual ingredients round-trip without losing reset data.

**Verification:** `scripts/test.sh` — 41 ML tests, 190 Swift tests (one
environment-gated parity test skipped), and the iOS simulator build all green.

## On-device depth (v6) — real-life depth test (added 2026-07-27)

**Goal:** ship both the v5 (plain) and v6 (depth-augmented) adapters and let the
user switch between them in Settings, so depth can be tested on real handheld
LiDAR captures — the one thing the held-out eval could not answer (v6 ≈ v5 on
rig-captured depth, but real-world LiDAR is a different distribution).

**Why it's a feature, not a toggle:** v6's prompt is v5's plus one measured line —
`Estimated food volume from depth sensor: ~<V> ml (max height <H> mm).` Feeding v6
a v5-style prompt (no line) breaks adapter↔prompt parity → garbage. So v6 needs
that line computed on-device, and computed *on the same scale it was trained on*.

**The hard part — calibration.** `ml/depth_features.py` is hard-calibrated to the
Nutrition5K RealSense rig: `FOCAL_PX = 465.1` (D435 color sensor @ 640×480), fixed
0.359 m overhead distance, and plane-fit thresholds tuned to that rig's glass
platform ("glass-platform trap"). iPhone LiDAR is a different sensor (256×192,
different intrinsics), handheld at variable distance, no glass platform. A naive
port produces volumes on a different scale → v6 gets out-of-distribution input and
the "test" is meaningless (cf. the documented 1.75× bias from a wrong focal length).

**Scope (in effort order):**
1. LiDAR depth capture on device (ARKit / AVFoundation `AVDepthData`) + camera intrinsics.
2. Swift port of `plane_fit` + `food_stats` (volume_ml, max_height_mm).
3. **Calibration** so phone volume estimates land on the training-time scale — the load-bearing part.
4. Settings toggle {v5 (no depth line) | v6 (with depth line)}; both adapters fused/bundled (`ml/fuse.sh adapters_v6 --out-path fused_v6` is trivial).

**Suggested approach:** Fable-planned track (capture → Swift features → calibration
strategy → toggle), then implement. A quick uncalibrated version (rough phone volume,
eyeballed) is possible first but is not a rigorous test — label it as such if shipped.
