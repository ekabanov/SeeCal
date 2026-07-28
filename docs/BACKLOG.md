# SeeCal backlog

Deferred work items. Not scheduled; picked up when prioritized. Dated when added.

## Out-of-distribution test set (added 2026-07-28)

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

**First step:** convert NutritionVerse-Real to our JSONL schema and run
`eval.sh adapters_v7b` against it. Expect the MAE to be worse than 63.4; the
point is to learn *how much*, not to pass a gate.

## Multi-view training (v8?) (added 2026-07-28)

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
