# SeeCal backlog

Deferred work items. Not scheduled; picked up when prioritized. Dated when added.

## Fully editable food data — ingredients, amounts, nutrition (added 2026-07-27)

**Goal:** every aspect of a scan result should be user-editable — ingredient
names, the ingredient list itself (add / remove), amounts, and the nutrition
numbers. The model is an estimator, not an authority; when it misidentifies or
misjudges something the user must be able to correct it rather than log a number
they know is wrong (or discard the scan).

**What works today** (`MealEditDraft`, `MealResultSheet`):
- Grams only, via `stepGrams` (±5 g, floored at 5 g) and `setGrams`.
- Nutrition is **derived**: per-item calories/macros are scaled proportionally
  from grams, and dish totals are the sum over items (`totals`).

**Missing:** editing an ingredient's name, adding an ingredient, removing an
ingredient, and setting calories/macros directly.

**The load-bearing design decision — derived vs. overridden.** Because nutrition
is currently a pure function of grams, letting the user type a calorie value
creates a conflict: the next gram step would silently recompute and destroy their
edit. Options:
1. **Per-field override flags** — a manually-set field stops tracking grams
   (needs a visible "edited" affordance so the user knows it detached, and a way
   to revert to the estimate).
2. **Edit grams-per-100g instead** — the user corrects the *density*, nutrition
   stays derived. Keeps one source of truth; less direct.
3. **Free-form totals** — dish totals become independently editable and stop
   being the item sum. Simplest UI, but then items and totals can disagree.

Recommend (1) with an explicit revert, and totals staying the item sum.

**Also needs deciding:** whether an added ingredient requires a nutrition lookup
(a per-100g food table shipped in-app — Nutrition5K's `ingredients_metadata.csv`
is a candidate source) or the user types its macros directly.

**Touches:** `MealEditDraft` (mutation API + override state), `MealResultSheet`
(name field, add/remove rows, numeric entry), `MealLogEntry` persistence (must
round-trip overrides), and `docs/specs/2026-07-26-app-spec.md` — the spec only
describes gram steppers today, so it needs updating first (spec wins over code).

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
