# v7 — not-food refusal adapter (plan)

*Added 2026-07-27. Author: Opus. Status: awaiting sign-off (schema decision) before any data download or training.*

## Problem

The shipping v5 adapter was trained on 100% food (4768 Nutrition5K dishes). It has
**never seen a non-food image**, so it always emits a nutrition JSON — it identified a
computer mouse as an object but still returned 19 kcal for it. The base Qwen3.5-VL knows
perfectly well that a mouse isn't food; fine-tuning *clobbered* that knowledge by only
ever rewarding a nutrition answer. v7 un-clobbers it: same food accuracy, but a clean
refusal on non-food.

## Core design decision — keep the prompt byte-identical to v5

We do **not** add "if this isn't food, respond with X" to the prompt. The prompt
(`SYSTEM_PROMPT + "\n\n" + USER_PROMPT` in `prepare_finetune.py`) stays **byte-for-byte
identical to v5**. Refusal is taught entirely by the *target completion* on negative
examples, not by prompt instructions. Consequences:

- **Parity is free.** The app already sends this exact prompt; no iOS prompt change, and
  the load-bearing adapter↔prompt invariant is preserved by construction.
- **We reuse v5's food JSONL unchanged** (`finetune_data_v2/`) as the positive half — no
  regeneration, no risk of perturbing the 4768 food targets.
- v7 = a fresh LoRA on (v5 food data **+** negatives). Not a continuation of v5.

## Refusal schema — THE decision that touches iOS  ⚠️ needs sign-off

Negative examples get this exact completion (same JSON parse path, distinct shape):

```json
{"not_food": true}
```

Rationale for the minimal single-boolean form:
- Maximally learnable (one token pattern, nothing to hallucinate) and unambiguous to detect.
- No free-text `reason` field: a reason invites the model to over-explain / drift, and the
  UX only needs "No food detected." (If we later want a reason, it's an additive field.)

**iOS change (additive, small):** in `ScanJSONParser`, before strict-decoding as
`FoodScanResult`, peek for `not_food == true`. If set, short-circuit to a new
`.notFood` outcome that the scan flow renders as a "No food detected — try a photo of a
meal" state (NOT a 0-kcal loggable meal, and NOT the generic "Failed to parse JSON"
error). Files touched:
- `ios/SeeCal/Sources/SeeCalDomain/NutritionModels.swift` — add a `NotFoodResult` /
  parse branch; keep `FoodScanResult.validated()` untouched.
- `ios/SeeCal/Sources/SeeCalInference/QwenNativeRuntimes.swift` — route the refusal
  through instead of throwing.
- Scan flow / `MealResultSheet` — a lightweight "no food" terminal state with a "New scan"
  button (reuse the existing error-screen layout, friendlier copy).
- Unit tests for the new branch.

This is the only iOS-facing part of v7 and the one thing I want signed off before building.

## Negative dataset

**Source: COCO 2017 val** (5000 images, permissive; used for training, not redistributed).
COCO has a `food` supercategory (banana, apple, sandwich, orange, broccoli, carrot,
hot dog, pizza, donut, cake). **Any image containing a food-category annotation is
excluded** — training the model to refuse on those would cause exactly the over-refusal
we must avoid.

Target **~350 negatives** (~7% of the 4768 food dishes — inside the "un-clobber, don't
re-teach" budget we discussed). Composition, weighted to hard negatives:
- **~40% hard negatives (food-adjacent, no food):** images with `dining table`, `bowl`,
  `cup`, `fork`, `knife`, `spoon`, `wine glass` annotations but **no food** — empty
  tableware / set tables. Teaches "tableware present ≠ food present." Include top-down-ish
  crops where possible so the model can't shortcut on "meal framing."
- **~60% plain non-food:** electronics, furniture, animals, people, vehicles, indoor
  scenes — the mouse/keyboard/desk class from the actual failure.

New script `ml/download_negatives.sh` + `ml/make_negatives.py`:
1. Download COCO val2017 images + instances annotations.
2. Filter out any image with a food-supercategory annotation; sample ~350 by the split
   above; copy into `ml/negatives/<coco_id>.jpg` (gitignored — like the dataset).
3. Emit `ml/negatives.jsonl` records via the **imported** `USER_PROMPT`/`SYSTEM_PROMPT`
   and `build_record()` from `prepare_finetune.py` (guarantees byte-identical prompt),
   completion `{"not_food": true}`, one image each, no depth line.
4. Split negatives 80/10/10 → train/valid/**test** (test negatives are the refusal-recall
   eval set; never seen in training).

## Assembling v7 data (prepare_finetune.py untouched)

A thin `ml/make_v7_data.py` concatenates and shuffles:
- `finetune_data_v2/{train,valid}.jsonl`  (v5 food positives)
- negatives `{train,valid}.jsonl`
→ `finetune_data_v7/{train,valid}.jsonl`. Test stays two separate files:
`finetune_data_v2/test.jsonl` (food, for the accuracy gate) and the negatives test
(for refusal recall). Keeping `prepare_finetune.py` unmodified minimizes risk to the
positive half.

## Parity gate

Extend `ml/check_prompt_parity.py` to also run over `finetune_data_v7/train.jsonl`
(mixed food + refusal records). Both record types use the identical prompt and a JSON
completion, so both must pass the existing byte-identical / mask-prefix checks.
**Gate: parity PASSED before training.**

## Training config — reuse v5 exactly

Same base (`~/models/mlx-community/Qwen3.5-4B-MLX-4bit`), `--only-overhead`,
LoRA rank 16 / alpha 32, batch 4, `--train-on-completions`, ~1000 iters
(may nudge up slightly for the ~350 extra records; decide from the loss curve).
Output `adapters_v7`. Metal co-tenancy rule applies — nothing else GPU-heavy during the run.

Ladder (per project convention): tiny overfit/probe first (small iters on a handful of
food + negatives, confirm it learns to refuse at all and still emits food JSON), then the
full run.

## Validation gates (all must pass)

| # | Gate | How | Target |
|---|------|-----|--------|
| 1 | **Food accuracy holds** | `eval.sh adapters_v7 --limit 325` on the food test set | calories MAE ≈ 59 (no regression beyond noise vs v5's 59.0/31.4) |
| 2 | **Over-refusal ≈ 0** (critical) | count `not_food` predictions on the 325 food dishes | ~0% refused. Over-refusal is worse than the original bug. |
| 3 | **Refusal recall** | run over the held-out negatives test set | ≥90% correctly refused |
| 4 | **Real-world spot check** | a handful of real iPhone non-food photos (mouse, desk, empty plate) + real meals | refuses non-food, keeps food |
| 5 | **Parity** | extended `check_prompt_parity.py` | PASSED |

`infer.py`/`eval.sh` need a small addition: detect `{"not_food": true}` and count it as a
**refusal** (separate tally) rather than a `schema_failure`, so gates 1–3 read cleanly.

## Risks & mitigations

- **Over-refusal on real food** (esp. partial/unusual plates) → gate 2 + hard negatives
  that are food-adjacent-but-empty; keep negative count modest (~7%).
- **Framing shortcut** (model keys on "natural scene vs top-down" not "food vs not")
  → top-down empty-plate/table hard negatives; gate 2 is the detector.
- **COCO food leak** → strict food-supercategory exclusion + manual spot-check of the
  sampled set before training.
- **iOS regression** → refusal branch is additive; `FoodScanResult.validated()` unchanged;
  unit tests cover both branches.

## Execution order (nothing downloads/trains until gate sign-off)

1. **Sign off the refusal schema `{"not_food": true}` + iOS "No food detected" state.** ← here
2. `download_negatives.sh` → filter/sample → manual spot-check of ~350 negatives.
3. `make_negatives.py` → `make_v7_data.py` → extended parity gate (must pass).
4. Probe run → full `train.sh` → `adapters_v7`.
5. Gates 1–5. If green: convert for Swift, wire the iOS refusal state, ship-decision vs v5.

## Results

### v7 (221 negatives / 7.8%, 2 epochs) — 2026-07-27

Training clean (2 epochs, final train loss ~0.14). Gates on the committed test sets:

| Gate | v7 | Target | Verdict |
|---|---|---|---|
| Refusal recall (29 held-out non-food) | **100%** (29/29) | ≥90% | ✅ |
| Over-refusal (325 real-food dishes) | **0** false refusals | ≈0 | ✅ |
| Food calories MAE (325) | **73.0** (median 38.4) | ≈59 (v5) | ❌ regressed |

The refusal behavior is exactly right — every non-food image refused, zero real
meals refused. But food calories MAE regressed **59.0 → 73.0** (+24%; median
31.4 → 38.4), both moving together = a real distribution shift, not outliers,
well beyond the ~3.5 kcal v5↔v6 noise. 221 refusal records (7.8%) pulled the
shared LoRA weights and cost food precision. **Not shippable over v5 as-is.**

Raw: `ml/runs/eval_v7_food325.json`, `ml/runs/eval_v7_neg.json`.

### v7b (100 negatives / 3.7%, 2 epochs) — 2026-07-27 — **PASSES ALL GATES**

Refusal recall was a perfect 100% and over-refusal 0 even with 221 negatives, i.e.
the refusal is *trivially* learned — so v7b halves the training negatives to 100
(valid/test negatives kept identical so the gates stay comparable) via
`make_v7_data.py --neg-train-cap 100 --out-dir finetune_data_v7b`. Same v5 config,
2 epochs.

| Gate | v5 | v7 (221 neg) | **v7b (100 neg)** | Verdict |
|---|---|---|---|---|
| Calories MAE (325) | 59.0 (n=322) | 73.0 | **63.4** (n=324) | see pairing below |
| Calories median | 31.4 | 38.4 | **30.5** | ✅ better than v5 |
| Over-refusal (real food) | — | 0 | **0** | ✅ |
| Refusal recall (29 non-food) | — | 100% | **100%** (29/29) | ✅ |
| Parse failures | 3 | 1 | **1** | ✅ better than v5 |
| Macros MAE (prot/fat/carb) | 5.6/4.6/5.4 | 5.7/5.6/7.3 | 5.6/5.8/7.0 | ~tie |

**⚠️ Do not compare the raw MAE headline numbers.** v5's 59.0 is over n=322
(3 parse failures excluded) while v7b's 63.4 is over n=324 — *different dish sets*.
Paired on the 324 shared dishes (same method as the v6−v5 test):

```
v5  on shared 324 : MAE 62.5   median 35.2
v7b on shared 324 : MAE 63.4   median 30.5
paired diff (v7b − v5): mean +0.87 kcal   sd 68.8   t = 0.23   → NOT significant
median paired diff    : +0.15 kcal
v7  paired diff       : mean +10.48 kcal  t = 2.49   → significant regression
```

**Conclusion: v7b is a statistical tie with v5 on food accuracy (t=0.23) with a
better median, while gaining 100% not-food refusal and 0 over-refusal.** The +0.87
kcal mean gap is entirely outlier-driven — the 5 worst dishes alone account for
+3.84 kcal of it, meaning v7b is *better* than v5 across the rest of the
distribution. v7's regression (t=2.49) was real; v7b's is not.

**Gate 4 (real-world device spot check) PASSED 2026-07-28** — user-verified on
an iPhone build bundling `adapters_v7b_swift`: non-food photos surface the "No
food detected" state and real meals still analyse normally. This was the one gate
the held-out set cannot answer, since all 325 test dishes are cafeteria trays shot
on the same RealSense rig.

**v7b is SHIPPED**: converted (`adapters_v7b_swift`), `SHIPPING_ADAPTER` swapped in
`ios/App/copy_weights.sh`, dev-fallback order updated in `ModelAssetResolver.swift`.

**TRACK COMPLETE.** All five gates green.

Raw: `ml/runs/eval_v7b_food325.json`, `ml/runs/eval_v7b_neg.json`.
```
