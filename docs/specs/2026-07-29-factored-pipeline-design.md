# Factored nutrition pipeline — design

**Date:** 2026-07-29
**Status:** DRAFT — pending user review
**Supersedes:** the monolithic photo→JSON contract described in
`docs/architecture.md` (which remains the shipping system until the gates in
§8 pass). Companion history: `docs/training-history.md`,
`docs/plans/2026-07-28-visual-specialist-conditioned-qwen-plan.md`.

---

## 1. Problem statement

SeeCal's errors are not all equal. Users compensate easily for **soft
mistakes** (wrong amount of the right ingredient; mayo vs. sour cream) and
lose trust entirely on **hard mistakes**:

- **H1 — wrong ingredient/dish** (raspberries instead of tomatoes)
- **H2 — wrong nutrient distribution for a correctly named ingredient**
  (fat in strawberries)
- **H3 — wrong calorie density for a correctly named ingredient**
  (150 kcal for 100 g of cucumber)

The current system optimizes a single number (total-kcal MAE) that cannot
see this distinction, and its architecture — one LoRA decoder emitting every
number — makes H2/H3 possible because nutrition facts live in the adapter
weights as lossy memories. The design goal, in priority order:

1. Make H2/H3 **structurally impossible**, not statistically rare.
2. Minimize and **detect** H1, deferring residual cases to a one-tap confirm.
3. Only then optimize soft error (portions, totals).
4. No manual dataset gathering anywhere.

## 2. Lessons carried forward (v1–v8)

| Lesson | Source | Design consequence |
|---|---|---|
| Fine-tuning, not base-model choice, does the heavy lifting | Gemma E4B comparison | Keep Qwen3.5-4B |
| One dataset (Nutrition5k, one rig) currently trains *everything*; scene diversity is the weakness | BACKLOG #1 | Factor tasks so identification can train on diverse data |
| Depth as image tokens fails; structured conditioning helps | v6 dead, v8 shipped | Geometry enters via a specialist, never the decoder |
| Multi-task interference in one completion is real; dose matters | v7 (221 negs) vs v7b (100) | Separate capabilities into separate components/adapters |
| Small specialists are cheap and calibratable | S1 (9.5 MB, intervals) | SCALE is a specialist regression, not decoder output |
| Prompt/adapter byte-parity is load-bearing and fragile | training history | One frozen prompt contract, parity check from day one |
| Teach via the completion, never the prompt | v7b | New schema lives in completions; prompt designed once |
| Pair by dish id when comparing models | v5/v7b incident | All comparative evals in this project are paired |
| Never fabricate zeros for missing data | barcode path | RESOLVE misses are flagged, never silently defaulted |
| Sensor calibration is the whole difficulty | v6 LiDAR blocker | LiDAR is additive input later, never on the critical path |

## 3. Design principle

> Every number the user sees is either **looked up** (nutrition facts),
> **measured** (mass, with a calibrated interval), or **confirmed** (low
> confidence identifications). The VLM's only unsupervised claim is *what the
> food is and in what proportions*.

The error taxonomy maps onto the architecture: H2/H3 are knowledge errors →
eliminated by lookup + arithmetic. H1 is a perception error → attacked with
data diversity, detected via independent-model disagreement and confidence,
deferred via UX. Soft portion error is isolated in one measurable module.

## 4. Architecture

```
photo (+ LiDAR depth when available)
 │
 ├── IDENTIFY   Qwen3.5-4B + LoRA (new schema)
 │              → {not_food, container, items:[{name, share_pct}]}
 │              names + percentage shares ONLY — no grams, no calories
 │
 ├── SCALE      specialist regressor (S1 descendant, Core ML)
 │              → total_mass_g + calibrated interval (P10/P50/P90)
 │
 ├── RESOLVE    name → nutrition entry, no ML (bundled SQLite: USDA FDC
 │              Foundation + SR Legacy + FNDDS composites, alias table)
 │              → per-100g {kcal, protein, fat, carbs} + provenance
 │
 └── ASSEMBLE   deterministic Swift arithmetic
                grams_i    = share_i × total_mass
                nutrition_i = grams_i × per-100g_i
                intervals propagated; Atwater + sane densities BY CONSTRUCTION
```

### 4.1 IDENTIFY

- **Model:** Qwen3.5-4B + LoRA, retrained from scratch on the new completion
  schema. Prompt text designed once in Stage 0, frozen, enforced by a
  `check_prompt_parity.py` equivalent from the first training run.
- **Output schema (completion JSON):**
  `{"not_food": bool, "container": "plate|bowl|cup|tray|packaging|other",
  "items": [{"name": str, "share_pct": int}]}` — items sorted by share
  descending, shares are integers summing to 100, rounded to the nearest 5
  (coarse buckets avoid false precision and match label noise; see E4).
- **Scale invariance is the point:** shares (not grams) make any food photo
  with an ingredient list usable as supervision, which is what fixes OOD
  identification without manual gathering.
- **Confidence:** mean token logprob over each item's name tokens, plus
  optional k-sample agreement (E5). Low confidence → confirm UX.
- **Not-food refusal:** kept, 100-negative dose (the proven v7b recipe).

### 4.2 SCALE

- **Model:** S1's direct descendant, retargeted to predict **total plate
  mass** with quantile heads (P10/P50/P90) for verifiable interval coverage.
  Image-only in v1; depth/LiDAR and plate-reference features are additive
  inputs behind a capability flag, never blockers.
- **Role change from v8:** promoted from "context appended to the 4B prompt"
  to *the authoritative absolute-scale source*. S1's measured value in v8 was
  marginal (paired −1.8 kcal, CI −8.3..+4.7); it stays only because the new
  architecture needs calibrated mass from somewhere, and it must earn the
  accuracy role in E2. If Qwen's implied mass beats it, ASSEMBLE fuses both
  (SCALE still supplies the interval and the disagreement gate).
- **Disagreement gate:** if Qwen-implied plausible mass (from category priors
  of identified items) and SCALE's interval diverge hard, flag the meal for
  confirmation — our best automatic H1 detector.

### 4.3 RESOLVE

- **Data:** bundled SQLite built from USDA FoodData Central (Foundation +
  SR Legacy, generic foods) + FNDDS (composite dishes: lasagna, curry, …).
  Public-domain, a few MB after pruning to name/category/per-100g macros +
  portion hints. Open Food Facts remains the barcode path, unchanged.
- **Resolution ladder** (first hit wins; every rung records provenance):
  1. **Exact alias hit.** The alias table pre-maps the entire training
     vocabulary (Nutrition5k ~555 ingredients — which are themselves
     DB-derived — plus FoodSeg103's 103 classes), so in-distribution misses
     are rare by construction.
  2. **Lexical fuzzy match** (normalized token overlap + edit distance)
     above threshold θ_resolve against FDC+FNDDS names.
  3. **Category default:** FDC's category taxonomy → median per-100g profile
     for the category ("unknown pastry" → baked goods). Item marked
     *estimated* in UI.
  4. **`NutritionHypothesisProvider`** — an interface, category-default
     implementation in v1. Reserved slot for the FDC-distilled text-only
     "nutritionist" adapter (E7), built only if the miss-rate metric
     justifies it.
  5. **User resolution:** item renders as uncertain with one-tap search
     (reuses existing editing + OFF search infrastructure).
- **Hard rule:** unresolved is visible, never a silent guess; no fabricated
  zeros. Unresolved names flow into the existing privacy-safe diagnostics to
  grow the alias table across releases.

### 4.4 ASSEMBLE

Pure Swift, no ML. `grams_i = share_i × total_mass`;
`nutrition_i = grams_i/100 × per100g_i`; totals are sums of items (matches
the shipped "meal totals derived from ingredients" invariant). Interval
propagation: nutrition intervals scale linearly with SCALE's mass quantiles;
identification confidence gates UI treatment rather than widening numeric
intervals (simple, honest, explainable). Atwater consistency and physically
sane densities hold by construction — H2/H3 cannot be emitted.

### 4.5 What is explicitly NOT in v1

- No runtime segmentation model (segmentation is an *offline labeling tool*
  only — FoodSeg103 masks become share labels during data prep).
- No LiDAR on the critical path.
- No second adapter at runtime (interface reserved; also blocked on the
  fused-adapter constraint: the iOS pipeline fuses LoRA into base weights,
  so hot-swapping requires unfused adapters in mlx-swift — real engineering,
  deferred with E7).
- No on-device embedding model for RESOLVE rung 2 in v1 (lexical first;
  embedding matching is an experiment, E9).
- No cloud training. Local Mac only (~10–20k records per run at v7b
  throughput: 2.6k records ≈ 3.5 h). Recipe1M-scale identification data is a
  deferred scale-up (E10) if the OOD gate says identification is still the
  bottleneck.

## 5. Metric design (the contract — built first, in Stage 0)

All metrics computable from data already on hand: Nutrition5k per-ingredient
ground truth (name, grams, kcal, macros per dish), the FDC dump, and
NutritionVerse-Real for OOD. **All comparative numbers are paired by dish id.**

### 5.1 Ingredient matching (foundation for everything below)

Predicted items are matched to ground-truth ingredients per dish:

1. Canonicalize both sides through the alias table → FDC id + FDC category.
2. Same FDC id ⇒ **exact match**. Same category ⇒ **soft match** (mayo vs.
   sour cream: both "fats/dressings" — a soft mistake). Different category ⇒
   **hard mismatch** (raspberry vs. tomato).
3. Names outside the alias table fall back to lexical similarity with
   thresholds θ_soft > θ_hard (tuned once on a held-out slice of the
   training vocabulary, then frozen).
4. Assignment: greedy by gram weight (largest GT ingredient first), one
   prediction per GT ingredient.

### 5.2 Tier 1 — hard-mistake gates (target ≈ 0; gate, don't average)

| Metric | Definition | Catches |
|---|---|---|
| **HMR** (hard-misidentification rate) | gram-weighted fraction of GT mass matched to a *hard mismatch* or hallucinated cross-category item | H1 |
| **DVR** (density-violation rate) | fraction of emitted items whose implied kcal/100g falls outside [0.5×, 2×] the resolved DB density, or whose dominant macro contradicts the DB profile | H2, H3 |
| **AIR** (Atwater-inconsistency rate) | fraction of items/totals with \|4p + 9f + 4c − kcal\| > 15% of kcal | incoherent outputs |
| **Refusal recall / over-refusal** | unchanged from v7b gates (29 held-out negatives / 324 food dishes) | not-food |

DVR and AIR need **no ground truth** — they are internal-consistency checks,
runnable on any photo including OOD phone photos. In the new architecture
DVR and AIR are zero by construction; they are measured on the *monolith*
baselines (E0) and kept as regression tripwires on anything a model emits.

### 5.3 Tier 2 — soft metrics (optimize once Tier 1 is clean)

| Metric | Definition |
|---|---|
| **IDR / IDP** | gram-weighted identification recall / precision at the exact-match level (soft matches reported separately) |
| **Share MAE** | mean absolute error of share_pct on exact-matched items |
| **Mass MAE / MAPE** | SCALE's total-mass error (paired vs. Qwen-implied mass in E2) |
| **Interval coverage** | empirical coverage of SCALE's P10–P90 (target 80% ± 5) |
| **Conditional kcal MAE** | total-kcal error restricted to dishes with no Tier-1 violation — the decontaminated successor of today's headline number |
| **Resolution rate** | fraction of emitted names resolving at ladder rungs 1–2 (rung ≥3 usage tracked per release) |

### 5.4 Evaluation sets

- **test325** (Nutrition5k, committed test set) — in-distribution; `eval.sh`
  discipline unchanged (explicit `--limit 325`).
- **NutritionVerse-Real** (~889 real photos, weighed ground truth) — the OOD
  gate. Ingredient vocabulary must be alias-mapped during Stage 0.
- **COCO negatives** (29 held-out) — refusal, unchanged.
- **Unlabeled OOD smoke set** — any personal/CC photos, scored on DVR/AIR +
  resolution rate only (no GT needed). Optional CLIP-style image↔names
  agreement as an H1 proxy (E5b).

## 6. Training pipeline

### 6.1 Data preparation (all derived, no manual gathering)

| Source | Records | Labels derived | Notes |
|---|---|---|---|
| Nutrition5k overhead | ~2.6k train | names + shares from GT gram fractions; container="tray" mostly | Reuse `dataset_clean/`; **decide cafe1-only vs cafe1+cafe2 explicitly** — the tidy CSVs are cafe1-only (4,768 dishes; cafe2's 228 imaged dishes are dropped there), so derive from raw metadata if both cafes are wanted |
| FoodSeg103 | ~7.1k | names from class labels; shares from **pixel-mask area fractions** (weak label — area ≠ mass share; hence coarse 5% buckets + E3/E4) | The OOD-diversity workhorse: real-world photos, varied plating |
| COCO negatives | 100 train / 29 test | `not_food: true` | Existing corpus incl. `negatives_cull.txt`; proven dose |
| NutritionVerse-Synth | ~84k renders | reserved for SCALE (mass GT), not IDENTIFY, in v1 | E8 gates whether synthetic transfers |
| FDC dump | ~10k foods | RESOLVE SQLite + alias table + category medians; also the training set for the deferred E7 adapter | public domain |

New prep scripts (in `ml/`, run from `ml/` per convention):
`make_fdc_db.py` (FDC/FNDDS → SQLite + category medians),
`make_alias_table.py` (training vocab → FDC ids; misses reviewed once via
contact-sheet-style listing — this is curation of a derived table, not data
gathering), `make_identify_data.py` (N5k + FoodSeg103 + negatives → new-schema
JSONL), `make_scale_data.py` (N5k mass + optional NV-Synth → SCALE training
set).

### 6.2 IDENTIFY training

- Same discipline as v7b: `check_prompt_parity.py` (rewritten for the new
  frozen prompt) MUST print PASSED → `smoke_test.py` → `train.sh` with
  `DATASET=finetune_data_id_v1`.
- ~10k records ≈ overnight local run at measured throughput. Metal is
  single-tenant: no concurrent evals/LM Studio.
- Mixing: N5k + FoodSeg103 interleaved; negatives capped at 100. If
  FoodSeg's coarse vocabulary (103 classes) drags share quality on N5k
  dishes, mix ratio becomes a tuned hyperparameter (E3).
- Multi-view side_angles remain out: multi-image records are blocked
  upstream (mlx-vlm #1726) and single side frames need downscaling first —
  unchanged backlog item, orthogonal to this design.

### 6.3 SCALE training

Retarget the S1 pipeline (`runs/visual-specialist/`) from its current heads
to total-mass quantile regression (pinball loss at 0.1/0.5/0.9). Minutes per
retrain — iterate freely. Calibration measured, not assumed (§5.3 interval
coverage). NV-Synth added only if E8 shows positive transfer to NV-Real.

### 6.4 Conversion & bundling

Existing `convert.sh` path for the LoRA (new `--version id-v1` stamp);
`coremlcompiler` for SCALE; SQLite bundled as an app resource. Weights and
datasets stay gitignored and uncommitted, per repo policy.

## 7. Experiments

**Stage 0 (measure before building — all runnable this week):**

- **E0 — Baseline hard-mistake audit.** Score existing v5/v7b/v8 prediction
  logs with the §5 harness. Establishes today's HMR/DVR/AIR — the numbers
  this redesign exists to fix. (If per-item predictions weren't logged,
  re-run `eval.sh` once per adapter with full JSON output.)
- **E1 — Resolution rate audit.** Run all item names from E0 logs through
  the RESOLVE ladder against the FDC dump. Decides how much rungs 3–5
  matter and sizes the alias table work.
- **E2 — Mass bake-off.** Paired-by-dish: S1-descendant total-mass MAE +
  interval coverage vs. monolith-implied total mass on test325. Decides
  SCALE's role (authoritative vs. fused vs. dropped).

**Stage 2+ (component development):**

- **E3 — Data-mix ablation.** IDENTIFY trained on N5k-only vs. N5k+FoodSeg103,
  scored on IDR/HMR on test325 (in-dist) and NV-Real (OOD). The design's
  central bet — diversity fixes OOD identification — gets a number.
- **E4 — Share granularity.** 5%-bucket vs. exact-integer shares; hypothesis:
  buckets match label noise and improve stability.
- **E5 — Confidence signals.** Do name-token logprobs and k-sample agreement
  actually rank H1 errors? (Measured as AUROC against E0's hard mismatches.)
  **E5b:** CLIP image↔names similarity as a GT-free H1 proxy for OOD sets.
- **E8 — Synthetic transfer.** SCALE ± NV-Synth, evaluated on NV-Real.

**Deferred (gated, not scheduled):**

- **E6 — Synthetic-corruption DPO.** Rejected completions generated by
  hard-mistake corruption (cross-category name swaps, density corruption) —
  directly encodes "hard first, soft second" into a preference objective.
  Gated on mlx-vlm DPO support and on E0 showing residual H1 after Stage 3.
- **E7 — FDC-distilled nutritionist adapter** (text-only LoRA: name →
  per-100g). Gated on E1 miss-rate and on unfused-adapter support in the
  iOS pipeline.
- **E9 — Embedding-based RESOLVE rung 2.** Gated on lexical fuzzy-match
  miss analysis from E1.
- **E10 — Cloud-scale identification data** (Recipe1M-class). Gated on the
  NV-Real OOD gate still failing after E3's best mix.

## 8. Final system implementation & rollout

**Stage 0 — Contract.** Metric harness (`ml/score_harness.py` + tests),
frozen IDENTIFY prompt + parity checker, FDC SQLite + alias table, E0/E1/E2.
Nothing ships; everything after is judged by this stage's tooling.

**Stage 1 — Deterministic spine.** RESOLVE + ASSEMBLE in Swift (unit-testable
without any model), SCALE-v0 retargeted and calibrated. The spine can be
exercised end-to-end with the *current* v8 model's names as a shadow mode.

**Stage 2 — IDENTIFY retrain.** New-schema training run(s), E3/E4 ablations,
best candidate selected on paired Tier-1-then-Tier-2 criteria.

**Stage 3 — Integration & gates.** Swift pipeline wired end-to-end
(IDENTIFY ∥ SCALE, then RESOLVE → ASSEMBLE), confirm-UX for low-confidence
items and disagreement-gate hits. **Ship gates, all must pass (v7b
five-gate pattern):**

1. Tier-1: HMR ≤ baseline v8's, DVR = 0 and AIR = 0 by construction
   (verified end-to-end on device output), refusal gates unchanged green.
2. OOD: NV-Real conditional kcal MAE and IDR strictly better than v8's on
   the same set.
3. In-distribution: paired conditional kcal MAE within noise of v8 on
   test325 (bootstrap CI includes 0) — the accepted trade for structural
   hard-mistake elimination. An in-dist *regression* outside noise blocks.
4. Interval honesty: SCALE coverage 80% ± 5 on held-out.
5. Device validation: memory (cache-limit discipline unchanged), latency
   budget ≤ current scan flow, thermal spot check, user-verified session.

v8 remains the shipping system and rollback baseline until all five pass.
`adapters_v7b` retains its protected-rollback status.

**Stage 4 — Feedback loop.** Resolution-miss and confirm-tap telemetry
(privacy-safe diagnostics channel) drive alias-table growth and the E6/E7/E10
gating decisions.

## 9. Risks & open questions

- **Area ≠ mass** for FoodSeg-derived shares (a thin spread covers pixels but
  not grams). Mitigated by coarse buckets + N5k providing true mass-fraction
  shares; E3/E4 quantify the residual harm.
- **FoodSeg103 vocabulary is coarse** (103 classes) vs. N5k's ~555 — mixing
  may pull naming toward generic terms. Watched via IDP on test325.
- **OOD absolute scale is genuinely hard** (no metric reference in a phone
  photo). SCALE's honest failure mode is *wide intervals*, which the UI can
  represent; the monolith's failure mode was confident nonsense. Plate/bowl
  container prediction from IDENTIFY may serve as a weak scale prior.
- **Composite-dish resolution quality** (FNDDS name matching) is unproven —
  E1 measures it directly.
- **Latency:** two models run per scan (they already do in v8), but IDENTIFY's
  completion is shorter than today's full-nutrition JSON — net latency likely
  improves. Verified at Stage-3 gate 5.
- **Schema-break for existing users:** persistence already supports edited
  ingredients; ASSEMBLE output maps onto the existing meal model
  (ingredient-derived totals), so migration is display-layer only. Confirm
  during Stage 3.

## 10. Decision log

- Keep Qwen3.5-4B; retrain freely (retraining is cheap; local-only).
- VLM emits **names + share_pct only** — the load-bearing decision.
- S1 descendant (SCALE) stays but must earn the accuracy role via E2;
  intervals + disagreement gate justify it regardless.
- No runtime segmentation; masks are an offline labeling tool.
- RESOLVE ladder with visible-uncertainty rule; no silent guesses, no zeros.
- Hypothesis adapter (E7) is an interface in v1, not an implementation.
- LiDAR additive, never blocking.
- Eval contract (this doc §5) changes **before** the architecture, so the
  known in-dist MAE trade-off cannot masquerade as a regression.
