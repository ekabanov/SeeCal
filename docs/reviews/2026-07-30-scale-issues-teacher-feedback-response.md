# Response to SCALE teacher feedback

**Date:** 2026-07-30  
**Feedback reviewed:** `2026-07-30-scale-issues-teacher-feedback.md`  
**Status:** zero-training diagnostics and Probe C1/C2 complete; C1 selected.
IDENTIFY LoRA remains paused.

## Accepted framing

The central correction is accepted: the factored architecture has already
demonstrated a strong in-domain oracle floor, and SCALE is the isolated
bottleneck. IDENTIFY LoRA remains paused. Calibration remains parked until the
point representation improves.

The proposed ordering is also adopted:

1. data/domain support;
2. input geometry;
3. ordinal auxiliary objective if ordering remains broken;
4. architecture capacity last.

Probe C will remain a two-arm factorial:

- C1: add FPB training data, geometry unchanged;
- C2: same data/exposure, switch center crop to letterbox.

Official Nutrition5K, NutritionVerse, and FPB tests stay frozen.

## Diagnostic 1 — occupancy response

A reproducible audit now measures source-specific occupancy proxies:

- Nutrition5K: calibrated depth-derived food footprint;
- NutritionVerse: COCO food segmentation area;
- FPB: union of YOLO food bounding boxes.

The proxies are not compared numerically across sources. Correlations are
computed within each source, both per-record and equal-group. The audit also
simulates the current 224 center crop and a 224 letterbox.

### Equal-group Pearson correlation after current center crop

| Source | Truth mass vs occupancy | Probe B P50 vs occupancy |
|---|---:|---:|
| Nutrition5K | 0.20 | 0.16 |
| NutritionVerse | 0.24 | **0.51** |
| FPB | **0.83** | **0.83** |

Interpretation:

- On NutritionVerse, Probe B is more than twice as coupled to visible
  occupancy as truth is. This supports a framing/occupancy shortcut.
- On FPB, the model does respond strongly to occupancy, and occupancy itself is
  strongly related to truth. The failure is therefore not “no portion signal
  at all”; it is a severely compressed mapping from that signal to grams.
- Nutrition5K truth and predictions remain tightly related to one another
  (P50-vs-truth Pearson 0.94 / Spearman 0.95) despite weak relationship to the
  2-D footprint alone. Height, composition, and fixed-rig priors carry much of
  the information there.

Letterbox simulation does not materially change these correlations before
training:

| Source | Current crop truth/pred r | Letterbox truth/pred r |
|---|---:|---:|
| Nutrition5K | 0.20 / 0.16 | 0.20 / 0.16 |
| NutritionVerse | 0.24 / 0.51 | 0.24 / 0.51 |
| FPB | 0.83 / 0.83 | 0.85 / 0.86 |

This does not make C2 unnecessary: preserving container/context may change
what the network learns. It does mean the pre-training diagnostic cannot
predict a large letterbox win.

Artifact:
`ml/runs/factored/scale-v2-probe-b-nv-1024/occupancy-audit.json`

## Diagnostic 2 — mass support

The measured histograms contradict the proposed numerical mass-support gap.
Unique-group mass distributions are:

| Source | Median | P90 | Maximum |
|---|---:|---:|---:|
| Nutrition5K training | 142 g | 374 g | 7,975 g |
| NutritionVerse training | 315 g | 724 g | 1,669 g |
| FPB frozen test | 208 g | 473 g | 1,494 g |

Only 3.7% of FPB groups exceed the NutritionVerse training P90, and none exceed
its maximum. FPB targets are well represented numerically in the mixed
training corpus.

The corrected diagnosis is therefore:

- prior-dominated/compressed output in the FPB visual domain;
- domain mapping and framing shortcut;
- not a simple lack of large gram labels.

FPB training remains the right first intervention because it supplies the
missing mapping between FPB visual evidence and already-supported target
values.

## Diagnostic 3 — EXIF and intrinsics

The exported datasets have stripped all EXIF:

| Source | Images inspected | Any EXIF | Focal metadata | Camera model |
|---|---:|---:|---:|---:|
| NutritionVerse | 889 | 0 | 0 | 0 |
| FPB test | 2,196 | 0 | 0 | 0 |

iOS can provide runtime intrinsics, but there is no corresponding training or
frozen-evaluation metadata in these two sources. Intrinsics-aware SCALE is
therefore not an immediately testable Probe C arm.

## Diagnostic 4 — USDA portion prior and fusion

The database and Swift runtime already contain a partial version of the
proposed mechanism:

- 13,035/13,589 food profiles have `typical_portion_g`;
- Swift computes `median(typical_portion/share)` as a plausible total;
- it uses that only to add `scaleDisagreement`, never to replace SCALE.

A new oracle-name/share audit evaluated five prior aggregations plus arithmetic
fusion, geometric fusion, and the current disagreement-triggered fallback.

### Best literal prior (`sum_item_portions`) and fusions

| Frozen source | SCALE P50 | Prior only | Arithmetic fusion | Geometric fusion | Current-rule fallback |
|---|---:|---:|---:|---:|---:|
| Nutrition5K, shared 319 groups | **37.4 g** | 231.5 g | 121.1 g | 94.2 g | 169.1 g |
| NutritionVerse, 67 groups | **96.2 g** | 151.5 g | 107.4 g | 106.1 g | 109.0 g |
| FPB, portion-resolved 90 groups | 170.0 g | **120.5 g** | 133.7 g | 143.8 g | 142.9 g |

USDA item-portion coverage is 90.2% on Nutrition5K, 97.2% on
NutritionVerse, and only 53.4% on FPB. Container conditioning cannot be
validated with these sets:

- all Nutrition5K food records are `tray`;
- all NutritionVerse records are `other`;
- FPB has no container labels.

The current Swift formula is particularly unsafe as a fallback:
`median(portion/share)` scores 412.2 g MAE on Nutrition5K and 482.8 g on
NutritionVerse. Typical USDA portions describe reference servings, not the
quantity visible in a cafeteria component. For example, `cheese pizza`
resolves to a profile whose two portions are a 112 g slice and an 897 g whole
pizza; the stored median is 504.5 g, while many Nutrition5K pizza components
weigh around 30–40 g.

The current disagreement threshold also has the wrong operating point:

- fires on 32.3% of Nutrition5K;
- fires on 6.0% of NutritionVerse;
- fires on only 16.4% of FPB, where mass is failing most severely.

Conclusion: retain the portion prior only as experimental independent evidence.
Do not make it an automatic fallback or fuse it into production yet. FPB
suggests a portion prior may help an unsupported domain, but its coverage and
cross-domain false-alert rate are inadequate. A reviewed serving-size table or
validation-derived gating policy would be required before product use.

Artifact:
`ml/runs/factored/scale-v2-probe-b-nv-1024/portion-prior-audit.json`

## Probe C results

FPB completed with 8,929 clean train records / 8,404 source-capture groups and
2,170 clean validation records / 2,170 groups. The importer excludes 779
records with explicit unknown/non-positive weights and two transformed train
images derived from frozen-test captures. A builder regression test ensures
official FPB validation is used only for validation.

Both arms used 50/25/25 effective N5K/NV/FPB exposure, source/group-balanced
sampling, worst-source equal-group-MAPE selection, and a 10% per-source Pareto
guard.

| Frozen test | Probe B | C1 center crop | C2 letterbox |
|---|---:|---:|---:|
| N5K overhead MAE | 36.9 g | 40.1 g | **37.5 g** |
| N5K side equal-group MAE | 41.3 g | 42.0 g | **41.0 g** |
| NutritionVerse equal-scene MAE | **96.2 g** | 99.0 g | 100.3 g |
| FPB equal-group MAE | 165.7 g | **73.3 g** | 80.0 g |
| FPB equal-group MAPE | 55.3% | **35.7%** | 38.6% |
| FPB ordered size triads | 23/40 | **37/40** | 33/40 |

C1 wins worst-source MAPE. It cuts FPB MAE 55.8% while the N5K overhead
regression remains inside the declared 10% ceiling and the other frozen-domain
changes are small. C2's geometry change is rejected: its modest N5K benefit
does not offset worse FPB and NutritionVerse results.

Ordering is no longer broadly broken: C1 gets 117/120 pairwise size
comparisons correct. Probe D is therefore parked rather than adding a third
training variable for three residual family failures.

## Downstream calorie regret

Swapping C1 mass into the existing oracle-visible-name/share assembly gives a
mixed result:

| Oracle assembly slice | Probe B mass | C1 mass |
|---|---:|---:|
| N5K, all complete groups | 64.2 kcal MAE | **63.4** |
| N5K, shared v8 Tier-1-clean 72 | 56.9 kcal MAE | **38.5** |
| NutritionVerse quality slice | **187.5 kcal MAE** | 191.2 |
| NutritionVerse raw official Val | **337.9 kcal MAE** | 343.9 |

On the shared clean N5K set, C1 closes most of the gap but still trails v8's
29.5 kcal MAE. Across all groups its gain is negligible, and NutritionVerse
slightly regresses. FPB has measured object weights but no independent nutrient
ground truth, so an FPB calorie-regret score would be circular and is not
reported. SCALE remains the binding constraint; this table does not justify
resuming IDENTIFY LoRA.

## Calibration result

Width-normalized phone-union calibration was implemented after selection. A
literal pooled union is dominated by FPB's 2,170 validation groups versus
NutritionVerse's 32 and undercovers frozen NutritionVerse at 70.1%. A
domain-robust union takes the maximum source-specific width multiplier while
applying one runtime rule. It reaches 84.0% NutritionVerse coverage but remains
conservative on FPB at 97.3%. This is safer than undercoverage but confirms
that wide-interval confirmation/fallback UX is required; conformal calibration
cannot make the two shifted domains exchangeable.

## Remaining teacher prescription

1. **v8 implied-mass baselines:** a deterministic one-view-per-scene
   NutritionVerse screen is complete. On 66 shared valid scenes (one v8 parse
   failure), v8 implied mass is 156.8 g MAE versus C1's 88.4 g; C1 wins 66.7%
   of scenes and improves paired MAE by 68.4 g. The full four-view pass was
   stopped after its measured ETA exceeded one hour. FPB v8 output remains
   outstanding and should use a group-screen first rather than blindly
   launching 2,123 long-form generations.
2. **Downstream calorie regret:** rerun oracle assembly with selected C1 mass
   on every domain for the actual promotion gate.
3. **Probe D:** parked; C1 repaired ordering to 37/40 triads.
4. **Calibration:** implemented; robust coverage remains conservative on FPB,
   so product fallback thresholds still need validation.
5. **Product fallback:** the explicit wide-interval/disagreement confirm path
   is now specified. Runtime threshold validation and UI integration remain;
   the unvalidated USDA prior is not promoted to the displayed estimate.

## Verification

- Occupancy audit tests: passed.
- Portion-prior audit tests: passed.
- Full ML suite: **161 passed**.
- Shipping adapter and production specialist: unchanged.
- No new LoRA training started.
