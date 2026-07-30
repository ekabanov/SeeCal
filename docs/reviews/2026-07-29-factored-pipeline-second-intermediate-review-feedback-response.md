# Factored pipeline — response to second intermediate teacher feedback

**Date:** 2026-07-29  
**Feedback:** `2026-07-29-factored-pipeline-second-intermediate-review-feedback.md`  
**Shipping state:** unchanged `v8-conditioned`; no new LoRA has started

## Outcome

All four blocking findings were accepted and implemented. The oracle audit
did reorder the work: IDENTIFY is not currently the limiting component.
Perfect visible identification plus current SCALE mass has **62.6 kcal MAE**
on the 72-dish intersection that is also Tier-1-clean for v8, versus
**29.5 kcal** for v8. With true mass, the same oracle assembly falls to
**12.4 kcal**. SCALE accuracy is therefore the immediate blocker; spending
hours on an IDENTIFY LoRA now would not make the assembled candidate win.

## BF1 — oracle assembly and hidden-label bias

The new `ml/oracle_assembly_audit.py` uses no VLM inference. It runs visible
ground truth through the real resolver and production assembly with true mass,
SCALE P10/P50/P90, exact shares, and 5-point bucketed shares.

### Nutrition5K test325

| Variant | Complete dishes | kcal MAE | Median signed kcal bias |
|---|---:|---:|---:|
| True mass + exact shares | 323/325 | 33.1 | +1.29% |
| True mass + bucketed shares | 323/325 | 34.4 | +1.20% |
| SCALE P50 + exact shares | 323/325 | 65.5 | −1.10% |
| SCALE P50 + bucketed shares | 323/325 | 65.8 | −0.93% |

Visible-label exclusion removes a median **0.58% of mass** and **1.64% of
kcal**. It therefore passes the requested ≲5% median-bias criterion. No
prepared-variant compensation is introduced, and hidden ingredients remain
excluded. The mean excluded-kcal fraction is 10.9%, so the long tail remains
a diagnostic even though the median decision passes.

The 5-point share buckets cost only **+1.28 kcal MAE** with true mass.
This confirms that label discretization is not the main error source.
Current SCALE P10–P90 maps to only **66.6% kcal interval coverage**, lower
than its mass coverage because density/resolution uncertainty is additional.

### NutritionVerse official Val

Only 22/67 scenes assemble completely with the current runtime resolver.
On that incomplete subset, true-mass/exact-share oracle MAE is 98.2 kcal and
SCALE-P50/exact-share MAE is 126.7 kcal. Median hidden exclusion is 10.45% of
mass and 11.38% of kcal, but the resulting true-mass median signed error is
−1.73%; redistribution partly offsets the omitted energy. These figures are
not promoted as full-domain accuracy because resolver completeness is only
32.8%.

## BF2 — FRB resolution before training

The original FRB arm failed the gate: 63.4% of item occurrences and only
36.6% of records were fully rung-1/2 resolvable. The 457 automatically
suggested aliases were reviewed as unsafe as a bulk promotion; examples
included red wine→vinegar, olive→olive loaf, and salad greens→dressing.

The replacement is a matched `+FRB` ablation that keeps a record only when
**every** teacher-accepted visible item resolves at rung 1 or 2. It contains:

| Split | FRB records | Combined records |
|---|---:|---:|
| Train | 1,461 | 9,138 |
| Validation | 162 | 2,649 |
| Test | 0 | 354 |

Across the 1,623 retained FRB records, all **3,088/3,088 item occurrences**
resolve at rung 1–2; all currently resolve as exact aliases. Prompt parity
passes. This exceeds the ≥90% occurrence gate without teaching partial truth.
FoodSeg remains the primary diverse-data arm and FRB remains a matched
ablation. If equal-unit FRB harms share behavior, the next fallback is the
teacher-approved single-item FRB arm.

The training smoke test also found a separate hard failure: oversized FoodSeg
images could consume the entire 2,048-token sequence and leave zero completion
tokens. Dataset generation now materializes 1,024-edge cached copies only for
oversized images (876 train, 197 validation) and reuses all smaller originals.
The post-resize smoke test passes with exact image-token counts and non-empty
completion masks.

## BF3 — evaluation taxonomy v3

`eval_taxonomy_v3.json` is frozen at SHA-256
`5c3cad9f733ad74ab99c2df2ae7cc54bef0526cf566ed9b1796ccbd6b3d71a27`.
Its 44-row reviewed override file is derived from NutritionVerse ground-truth
labels, never model outputs. Relative to v2 it adds 2 entries and changes 33;
there are no removals. The diff mainly replaces unsafe lexical mappings,
assigns reviewed coarse families to composites, and attaches exact FDC IDs
only where defensible.

One correction to the feedback diagnosis: v2 already contained 40 of the 42
NutritionVerse labels. The zero exact IDR/IDP was not solely missing
vocabulary; many mappings were null/wrong, and the legacy tuned model also
genuinely emits different identities.

Fast-slice rescoring under v3:

| Model | NutritionVerse HMR | IDR | IDP |
|---|---:|---:|---:|
| Untuned Qwen3.5-4B | 42.7% | 24.9% | 27.7% |
| Tuned legacy adapter + normalization | 28.9% | 9.8% | 11.3% |

The metrics are now non-zero and usable, while also showing that lower HMR
does not imply better exact identity recall. Nutrition5K results are unchanged
under v3. Future arm evaluation is pinned to taxonomy v3 and the 265-view /
67-scene official NutritionVerse Val manifest.

Runtime resolution remains deliberately separate from evaluation taxonomy.
NutritionVerse official-Train truth currently resolves at rung 1–2 for only
9.7% of occurrences and 7.6% of visible mass. Reviewed train-only runtime
canonicalization is therefore a new deterministic blocker; official-Val
labels will not be promoted into the runtime database.

## BF4 — checkpoint selection and calibration uncertainty

SCALE training now writes `checkpoint_selection.json`. For Probe B:

- selected epoch: 5;
- selection set: `datasets/scale_v2_nc_1024/valid.jsonl`;
- selection metric: minimum validation mass MAE, **48.7729 g**;
- selection-manifest SHA-256:
  `111b02ff65e862ff47429b423934edc72560548844ba235d65a0b8e759f096d8`;
- both official test manifests and their hashes are recorded as excluded.

Coverage reports now include equal-group Wilson 95% intervals:

| Frozen set | Coverage | 95% interval |
|---|---:|---:|
| Nutrition5K overhead, 325 dishes | 80.3% | 75.6–84.3% |
| Nutrition5K side, 311 dishes | 82.4% | 77.7–86.2% |
| NutritionVerse official Val, 67 scenes | 81.3% | 70.4–88.9% |

The NutritionVerse result is therefore reported as consistent with the 80%
target, not equal to it. Width-normalized conformity remains optional after
the point-estimate blocker is improved.

## Remaining sequence

1. Improve and remeasure SCALE; current mass errors erase the oracle
   IDENTIFY advantage.
2. Canonicalize NutritionVerse official-Train runtime vocabulary without
   using official-Val labels.
3. When the user-provided FPB download is available, run Probe B zero-shot on
   FPB before using any FPB training data.
4. If zero-shot warrants it, run one modest ≈50/25/25
   Nutrition5K/NutritionVerse/FPB probe while preserving NutritionVerse
   exposure. Reject if either frozen N5K or NV regresses.
5. Only after the deterministic spine and SCALE gates improve, run a tiny
   IDENTIFY-v2 overfit/preflight, FoodSeg primary arm, and matched `+FRB`
   ablation. A full LoRA remains last.

Any release containing weights trained on NutritionVerse or FPB remains
blocked on a release-specific non-commercial license review. The
permissive-only SCALE manifest remains reproducible.

## Continuation after this response

Further deterministic work found that the database had silently dropped all
FNDDS rows: FNDDS 2021–2023 uses legacy nutrient numbers 203/204/205 in
`food_nutrient.nutrient_id`, while the importer accepted only modern IDs.
The fixed importer is covered by a regression test and now produces 13,589
profiles, including 5,430 FNDDS composites, and 196 category medians.

With 38 reviewed mappings derived only from NutritionVerse official Train:

- official-Train rung-1/2 occurrence resolution is 92.6%;
- all 624 official-Train and 265 official-Val views assemble;
- the regenerated FRB ablation grows to 1,905 train / 212 validation records,
  with 4,122/4,122 occurrences resolving at rungs 1–2;
- shared-clean Nutrition5K true-mass oracle improves 12.4 → 10.3 kcal MAE;
- shared-clean SCALE-P50 oracle improves 62.6 → 56.9 kcal MAE, still behind
  v8's 29.5.

The mass sensitivity sweep quantifies the remaining SCALE target. On the same
shared clean set, retaining 50% of current mass error gives 30.3 kcal MAE
(still behind v8), while retaining 25% gives 18.8. The crossover requires
roughly a 52% reduction in current mass error, approximately 36.9 → 18 g MAE.

The now-complete NutritionVerse oracle exposed a source-label defect:
`near-whole-chicken` is encoded as 480 kcal/100 g with 64.39 g
carbohydrate/100 g. The primary raw benchmark retains it. A separate quality
slice is frozen using official Train only and excludes its seven affected Val
scenes; on the remaining 60 scenes, true-mass/bucketed-share oracle MAE is
78.4 kcal and current SCALE-P50/bucketed-share MAE is 187.5 kcal.

A validation-only point calibration selected a NutritionVerse log-affine
correction, but it failed the frozen-test acceptance gate: MAE worsened
96.2 → 98.8 g and coverage fell 81.3% → 75.4%. It is rejected; Probe B stays
authoritative.

The user-managed FPB download exposed a complete official test partition before
train/validation finished, allowing the required pre-training zero-shot gate.
The importer now supports a test-only manifest and explicitly excludes 73
images containing `-1` unknown object weights; 2,123 clean-weight images across
164 groups remain. Probe B scored:

| FPB frozen test metric | Result |
|---|---:|
| Record mass MAE | 157.3 g |
| Equal-group mass MAE | 165.7 g |
| Equal-group MAPE | 55.3% |
| P10–P90 equal-group coverage | 20.9% (95% CI 15.4–27.8%) |

This is representation failure, not a small calibration miss. Median prediction
is 89 g against 206 g truth, 85% of records are underestimated, and only 22/40
complete food triads are ordered small < average < big. Even a deliberately
test-leaking global-multiplier diagnostic leaves about 106 g equal-group MAE.
The frozen FPB result therefore justifies one modest FPB-trained SCALE probe
once train/validation are available; it does not justify any IDENTIFY LoRA yet.

## Verification

- Python: **148 passed**
- IDENTIFY-v2 safe FRB arm prompt parity: **passed**
- Post-resize image/batch/completion-mask smoke test: **passed**
- Shipping adapter, production factory, and bundled specialist: **unchanged**
