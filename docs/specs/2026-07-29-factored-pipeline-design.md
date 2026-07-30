# Factored nutrition pipeline — design

**Date:** 2026-07-29
**Status:** APPROVED — execution in progress (approved 2026-07-29)
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
 │              → {not_food, container, items:[{name, portion_units}]}
 │              names + relative portion units ONLY — no grams or nutrition
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
  "items": [{"name": str, "portion_units": int}]}` — items sorted by portion
  descending. Units are positive integers; they do not need to sum to a
  particular value. ASSEMBLE normalizes them to shares deterministically.
- **Compatibility rule:** the completed v1 percentage adapter may be
  evaluated through a strict legacy parser. Positive numeric percentages are
  merged by normalized name and converted to deterministic 5-point shares
  summing to 100. This salvages its learned identification behavior without
  making arithmetic closure part of the model's job.
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
- **Explicit uncertain-scale state:** after validation freezes a relative-width
  threshold, trigger confirmation when
  `(P90 - P10) / max(P50, ε)` exceeds it or the disagreement gate fires. The
  result sheet must present the mass as an approximate editable range and ask
  the user to adjust or accept before logging; it must not silently collapse a
  wide interval to a precise number. USDA/FNDDS typical portions may appear
  only as a clearly labelled serving-size hint after their item coverage is
  validated. They never replace SCALE automatically: the cross-domain audit
  shows raw portion priors regress Nutrition5K and NutritionVerse badly.

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

1. Canonicalize both sides through the committed, versioned
   `eval_taxonomy_v2` snapshot → FDC id + FDC category. Runtime RESOLVE never
   defines its own HMR/IDR report card.
2. Same FDC id ⇒ **exact match**. Same category ⇒ **soft match** (mayo vs.
   sour cream: both "fats/dressings" — a soft mistake). Different category ⇒
   **hard mismatch** (raspberry vs. tomato).
3. Names outside the frozen taxonomy fall back to the taxonomy's independent,
   frozen lexical implementation with
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
| **Share MAE** | mean absolute error after normalizing portion units (or legacy percentages) to shares |
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
| FoodSeg103 | ~7.1k | names from class labels; shares from **pixel-mask area fractions** (weak label — area ≠ mass share; hence coarse 5% buckets + E3/E4); `container="other"` because the dataset has no container labels | The OOD-diversity workhorse: real-world photos, varied plating. Container is diagnostic-only for the mixed arm; the default must not become a disagreement gate |
| FRB 2022 v2.0 teacher pilot | 5,000 selected images; 4,430 with accepted visible names | visible names + mapped container; equal relative units only | CC BY 4.0 user-provided archive. Recognition/schema supervision for a separate IDENTIFY arm; no mass claim |
| COCO negatives | 100 train / 29 test | `not_food: true` | Existing corpus incl. `negatives_cull.txt`; proven dose |
| NutritionVerse-Real v2 | 624 official-Train views / 158 scenes | total measured mass for SCALE in the non-commercial track only | split by scene; official Val is frozen for testing; never bundled |
| Food Portion Benchmark | 8,929 clean train + 2,170 clean validation images | total mass from summed positive per-object gram labels | CC BY-NC 4.0; source-capture grouped smartphone/RealSense data; two train records overlapping frozen test captures excluded; non-commercial track only |
| NutritionVerse-Synth | ~84k renders | reserved for SCALE (mass GT), not IDENTIFY, in v1 | E8 gates whether synthetic transfers |
| FDC dump | ~10k foods | RESOLVE SQLite + alias table + category medians; also the training set for the deferred E7 adapter | public domain |

New prep scripts (in `ml/`, run from `ml/` per convention):
`make_fdc_db.py` (FDC/FNDDS → SQLite + category medians),
`make_alias_table.py` (training vocab → FDC ids; misses reviewed once via
contact-sheet-style listing — this is curation of a derived table, not data
gathering), `make_identify_data.py` (N5k + FoodSeg103 + negatives → new-schema
JSONL), `make_scale_v2_data.py` (leakage-safe N5k side-view expansion plus
optional non-commercial sources → SCALE manifests), and
`make_fpb_scale_data.py` (FPB labels → grouped SCALE manifests).

### 6.2 IDENTIFY training

- Same discipline as v7b: `check_prompt_parity.py` (rewritten for the new
  frozen prompt) MUST print PASSED → `smoke_test.py` → `train.sh` with
  `DATASET=finetune_data_id_v1`.
- ~10k records ≈ overnight local run at measured throughput. Metal is
  single-tenant: no concurrent evals/LM Studio.
- Mixing: N5k + FoodSeg103 interleaved; negatives capped at 100. If
  FoodSeg's coarse vocabulary (103 classes) drags share quality on N5k
  dishes, mix ratio becomes a tuned hyperparameter (E3).
- The primary E3 policy is same epochs, so every included example is seen
  twice. The mixed arm therefore has about 2.85× more optimizer steps; this is
  an explicit policy/compute confound. An equal-step N5k control is deferred
  unless a mixed-arm win would change a later data-scaling decision.
- Adapter resume is disabled for this track. `train.sh` must observe
  32.464896M trainable parameters (within its small guard tolerance) before
  the first iteration or abort; the 366.9M resume shape is never accepted.
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
  central bet — diversity fixes OOD identification — gets a number. Decision
  order is frozen before scoring: (1) HMR under `eval_taxonomy_v2` plus
  refusal/over-refusal gates; (2) paired assembled conditional kcal MAE on the
  intersection of dishes that are Tier-1-clean and fully resolved in both
  arms, with completeness printed adjacent; in-distribution must remain
  within bootstrap noise and OOD must improve; (3) IDR/IDP/share MAE remain
  diagnostics. NV-Real's Tier-2 primary averages views within each dish and
  then averages 225 dishes; Tier-1 events use equal dish weights over pooled
  views, with per-view P90 diagnostic.
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
5. Resolution honesty: gram-weighted rung-1/2 rate ≥85% on test325, with the
   NutritionVerse-Real rate reported alongside.
6. Device validation: memory (cache-limit discipline unchanged), latency
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
- VLM emits **names + relative portion units only**. Deterministic
  normalization owns the sum-to-100 constraint. The v1 percentage adapter is
  retained as a normalized compatibility baseline.
- S1 descendant (SCALE) stays but must earn the accuracy role via E2;
  intervals + disagreement gate justify it regardless.
- No runtime segmentation; masks are an offline labeling tool.
- RESOLVE ladder with visible-uncertainty rule; no silent guesses, no zeros.
- Hypothesis adapter (E7) is an interface in v1, not an implementation.
- LiDAR additive, never blocking.
- Eval contract (this doc §5) changes **before** the architecture, so the
  known in-dist MAE trade-off cannot masquerade as a regression.

## 11. Execution record (2026-07-29)

Stages 0–1 are implemented without changing the shipping v8 path:

- The frozen IDENTIFY prompt/schema, derived-data builders, prompt-parity
  gate, real mlx-vlm smoke gate, FDC database builder, alias builder, and
  paired score harness are implemented and tested. The final E3 mix builder
  produced 7,677 train / 2,487 valid / 354 test records from Nutrition5K,
  the exact 4,983/2,135 FoodSeg103 image-mask splits, and the proven negative
  dose. The upstream FoodSeg archive endpoint returned HTTP 502, so the
  downloader pins the Apache-2.0 Hugging Face mirror to commit
  `34e1208e14bc3595d544fc8c3f3c6673253fd9ef` and records all four source
  object SHA-256 values. Prompt parity passed 8/8 and the real mlx-vlm smoke
  gate passed on the generated mix.
- IDENTIFY v1 deliberately uses the existing cafe1-derived Nutrition5K split,
  rather than silently adding cafe2 from raw metadata. This preserves exact
  dish pairing with the current v8/test325 baseline for E3. Cafe2 remains a
  separately measurable future mix extension.
- NutritionVerse-Real v2 is pinned and materialized from the public Kaggle
  release (CC BY-NC-SA 4.0; local archive SHA-256
  `22910bc84cb5840cb4a66be20d71ce2b7ba51f1d1a7f62108debf78fe7180780`).
  The derived OOD manifests contain all 889 images across 225 represented
  dish IDs, preserve measured per-item grams/source nutrition, and carry an
  explicit group id so comparative metrics aggregate by dish rather than
  overweighting viewpoints.
  The project owner subsequently authorized non-commercial training because
  SeeCal is free and open source. The official Train scenes are therefore
  allowed only in the separately marked non-commercial SCALE manifests;
  official Val remains frozen for OOD testing. Source images are never
  bundled or redistributed. A permissive-only manifest remains available.
- The pruned database was built from the official USDA Foundation 2026-04,
  SR Legacy 2018-04, and FNDDS 2021–2023 CSV releases. An execution audit
  found that FNDDS stores legacy nutrient numbers (203/204/205) in the
  `nutrient_id` column; the first importer accepted only modern IDs and had
  silently discarded the entire advertised FNDDS source. The corrected,
  tested importer contains 13,589 complete profiles (including 5,430 FNDDS
  composites), 196 category medians, and is 7.2 MB after derived aliases.
  Calories
  exposed by the factored path use 4/9/4 arithmetic over the looked-up USDA
  macros; the original USDA energy value remains alongside it as provenance.
- E0 on the 325 paired v8 outputs measured HMR 12.9%, DVR 40.5%, and AIR
  31.6%. Only 39/325 dishes were clean on all three Tier-1 checks. These
  numbers are diagnostic baselines, not a reinterpretation of the shipping
  56.8 kcal MAE.
- E1 resolved 17.6% of v8-emitted names at rungs 1–2, 82.1% at category
  defaults, and left 0.3% visible/unresolved. The alias database was rebuilt
  from train/validation completions only; neither NutritionVerse ground truth
  nor evaluation-emitted names are promoted. A reviewed 228-entry,
  source-backed alias artifact brings the actual 271-name training vocabulary
  to 258/271 (95.2%) rung-1 coverage. Thirteen intrinsically ambiguous or
  non-food labels remain category-default/unresolved instead of being mapped
  to unrelated USDA profiles. Candidate-emission resolution is measured
  separately after training.
- E2 found the existing S1 mass head better than Qwen-implied mass on the
  same 325 dishes: 35.9 g vs 43.7 g MAE (paired advantage 7.8 g), so SCALE is
  authoritative in v1. Its 74.2% raw interval coverage failed the honesty
  gate.
- SCALE-v1 was then retargeted to a mass-only P10/P50/P90 model, initialized
  from S1, calibrated, exported, and compiled to Core ML. On untouched
  test325 it reached 34.3 g MAE / 22.4% MAPE with 77.2% P10–P90 coverage,
  passing the 75–85% interval gate.
- Swift now contains the strict IDENTIFY parser, SQLite RESOLVE ladder,
  deterministic interval-aware ASSEMBLE arithmetic, visible unresolved
  state, confidence/disagreement confirmation gates, current-v8 shadow
  adapter, concurrent IDENTIFY/SCALE pipeline shape, and the mass-only Core
  ML runtime. None is selected by `ProductionFactory`; v8 remains shipping.
- A continuous 20-epoch, 32-record IDENTIFY memorization probe reached
  0.000082 training loss and 32/32 schema-valid generations, with HMR 0,
  IDR/IDP 1.0, and share MAE 0. The shorter 10-epoch probe's seven share-sum
  failures were undertraining rather than a contract defect.
- HMR/IDR now read committed `eval_taxonomy_v1`, frozen before alias
  promotion. Re-scoring the fixed E0 predictions reproduced HMR
  0.1292329217, IDR 0.7313183142, and IDP 0.7130800891 exactly; a runtime
  resolver change can no longer improve its own evaluation taxonomy.
- Nutrition5K's objective ≥5% retention rule keeps oil-type labels in 1,630
  of 3,244 completions (1,791 items: 1,558 olive oil, 82 vinaigrette,
  55 mayonnaise, 46 Caesar dressing, 42 butter, and 8 vegetable oil).
  Nothing is removed by subjective visibility review; OOD condiment
  hallucination is reported per E3 arm as the counter-diagnostic.

Subsequent fast-iteration corrections supersede the v1 contract where they
conflict with the bullets above:

- Full-arm behavior showed that requiring a 4B model to emit percentages
  summing to 100 was not reliable: the Nutrition5K-only arm had 18.8% schema
  failures and 25.2% repaired outputs under the v1 rules. IDENTIFY v2 therefore
  emits positive integer `portion_units`; deterministic code performs all
  normalization. Prompt parity checks both the retained v1 prompt and the v2
  prompt byte-for-byte.
- Scoring now uses `eval_taxonomy_v3.json`
  (SHA-256 `5c3cad9f733ad74ab99c2df2ae7cc54bef0526cf566ed9b1796ccbd6b3d71a27`)
  and performs exact matching before soft matching. Training labels follow a
  visible-label policy; a deterministic post-filter is available for the
  legacy adapter. The v3 diff is ground-truth-derived and reviewed: 2 entries
  added, 33 changed, none removed. Nutrition5K fast metrics are unchanged.
- The completed tuned v1 percentage adapter remains useful after compatibility
  normalization. On the frozen 64-record Nutrition5K slice it produced 64/64
  valid structures, 8/8 negative refusals, HMR 12.0%, IDR 77.0%, IDP 68.9%,
  and share MAE 5.8 points. The untuned Qwen baseline on the same slice had
  HMR 41.7%, IDR 34.6%, and IDP 36.6%. On the 63-view NutritionVerse slice the
  tuned adapter still had 33.6% equal-scene HMR, confirming that OOD
  identification remains unsolved.
- SCALE-v1's apparent in-domain success did not transfer: on the frozen
  NutritionVerse official-Val split (265 views / 67 scenes) it measured
  302.4 g record-level MAE, 300.0 g equal-scene MAE, and 6.0% P10–P90
  coverage. `make_scale_v2_data.py` now constructs leakage-safe,
  source-labelled manifests with all usable Nutrition5K side views. The
  permissive corpus has 39,352 train records / 4,293 groups; the
  non-commercial corpus adds 624 NutritionVerse official-Train views / 158
  scenes while preserving its 265-view / 67-scene official Val as test.
- Food Portion Benchmark (CC BY-NC 4.0; 14,083 images, smartphone and
  RealSense viewpoints, per-object gram labels) is the next SCALE source.
  Its importer groups related viewpoints and sums object weights. It is
  isolated to the non-commercial track and pinned by revision. Before any
  FPB training, the untouched FPB test partition was imported as a zero-shot
  gate. Of 2,196 images, 2,123 have complete positive weight labels across
  164 food/portion groups; 73 records with explicit `-1` unknown weights and
  one orphan label are reported and excluded.
- A matched SCALE-v2 probe using Nutrition5K plus NutritionVerse official
  Train succeeded without FPB: 36.9 g MAE on 325 N5K overheads, 41.3 g
  equal-group MAE on 1,179 held-out N5K side views, and 96.2 g equal-scene
  MAE on the frozen NutritionVerse official Val (265 views / 67 scenes).
  The N5K-only multi-view probe scored 44.2 g / 49.2 g / 331.8 g on the
  same three gates, so viewpoint expansion from one rig was not a substitute
  for measured phone-domain data.
- SCALE calibration is group-aware and source-specific. Each dish has total
  weight one regardless of view count, and finite-sample ranks use independent
  group count. Margins of 2.48 g for the controlled N5K rig and 79.26 g for
  phone photos produced held-out coverage of 80.3% N5K overhead, 82.4%
  equal-group N5K side, and 81.3% equal-scene NutritionVerse. The phone margin
  is the deployment default; the maximum remains the unknown-source fallback.
  Wilson 95% intervals are 75.6–84.3%, 77.7–86.2%, and 70.4–88.9%,
  respectively. Checkpoint selection is explicitly logged against the
  calibration/holdout manifest; official tests are recorded as excluded.
- The oracle assembly audit passes the hidden-label median-bias gate on
  test325: true mass + exact visible shares has +1.29% median kcal bias;
  visible exclusion removes a median 1.64% of kcal. Five-point share buckets
  cost only +1.28 kcal MAE. However, on the 72-dish intersection that is also
  Tier-1-clean for v8, the corrected-FNDDS true-mass oracle reaches 10.3 kcal
  MAE while current SCALE-P50 oracle assembly is 56.9 kcal versus v8's
  29.5 kcal.
  A mass-error sensitivity sweep puts the crossover just below 50% retained
  error: retaining half of current SCALE error gives 30.3 kcal MAE; retaining
  one quarter gives 18.8. SCALE therefore needs roughly a 52% error reduction
  (about 36.9 → 18 g MAE) before the oracle candidate beats v8. SCALE point
  accuracy is therefore the next blocker, ahead of IDENTIFY LoRA.
- FRB remains a matched IDENTIFY-v2 ablation, not the primary arm. The safe
  arm keeps only records whose complete teacher-visible truth resolves at
  rungs 1–2: 1,905 train / 212 validation records and 4,122/4,122 resolved
  item occurrences. The combined corpus is 9,582 / 2,699 / 354. Oversized
  training images are capped at a 1,024-pixel edge; this fixed a discovered
  zero-completion-token truncation and the post-cache smoke test passes.
- The corrected FNDDS database plus 38 reviewed aliases derived only from
  NutritionVerse official Train raises train rung-1/2 occurrence resolution
  to 92.6% and makes all 265 official-Val views assemblable. The primary raw
  official-Val oracle remains reported. A separately labelled quality slice,
  frozen from a Train-only semantic/nutrient rule, removes seven scenes using
  the corrupt `near-whole-chicken` template (480 kcal and 64.39 g carbohydrate
  per 100 g). On the remaining 60 scenes, true-mass/bucketed-share oracle MAE
  is 78.4 kcal and current SCALE-P50/bucketed-share MAE is 187.5 kcal.
- A five-fold, group-aware validation experiment selected a source-specific
  log-affine point correction for NutritionVerse, but it was rejected after
  frozen official-Val MAE worsened from 96.2 to 98.8 g and coverage fell from
  81.3% to 75.4%. Probe B remains unchanged.
- Probe B then failed FPB zero-shot decisively: 157.3 g record MAE, 165.7 g
  equal-group MAE, 55.3% equal-group MAPE, and 20.9% P10–P90 coverage
  (164-group Wilson 95% CI 15.4–27.8%). It predicts a narrow, low range
  (89 g median versus 206 g truth), underestimates 85% of records, and orders
  small < average < big correctly for only 23/40 complete food triads. A
  test-label-leaking multiplier diagnostic still leaves about 106 g
  equal-group MAE, so this is not accepted as a calibration problem. The next
  SCALE probe must learn from FPB train/validation imagery while keeping this
  FPB test partition frozen.
- A zero-training occupancy audit refined the diagnosis. Under the actual
  center-crop geometry, equal-group occupancy correlations (truth / Probe B)
  are 0.20/0.16 for Nutrition5K, 0.24/0.51 for NutritionVerse, and 0.83/0.83
  for FPB. Probe B therefore does see FPB portion-area signal, but maps it to
  a compressed gram range; on NutritionVerse it is over-coupled to framing
  relative to truth. The numerical target-support hypothesis is rejected:
  FPB group median/P90/max are 208/473/1,494 g versus NutritionVerse training
  315/724/1,669 g. No FPB group exceeds the NV training maximum.
- Exported NutritionVerse (889/889) and FPB test (2,196/2,196) images contain
  no EXIF, focal-length metadata, or camera model. Runtime iPhone intrinsics
  remain a possible later feature, but there is no corresponding training or
  frozen-evaluation input for Probe C.
- USDA typical portions are not accepted as an automatic SCALE fallback.
  Oracle-name/share audits show the literal sum-of-portions prior worsens
  Nutrition5K from 37.4 to 231.5 g MAE and NutritionVerse from 96.2 to
  151.5 g; arithmetic/geometric fusion also regresses both. It helps only the
  53%-covered FPB subset (170.0 → 120.5 g). The existing Swift
  `median(portion/share)` disagreement proxy is even less accurate and its
  threshold fires on 32.3% of N5K but only 16.4% of FPB. It remains an
  experimental independent signal, never a replacement estimate.
- Probe C completed as a two-run mini-factorial: C1 adds FPB with center crop
  unchanged; C2 uses the same data and
  50/25/25 effective exposure with aspect-preserving letterbox. Checkpoint
  selection minimizes worst-source equal-group MAPE, tie-breaks by mean
  source MAPE, and rejects any epoch more than 10% worse than the per-source
  best seen during that run. C1 is selected: N5K overhead/side MAE is
  40.1/42.0 g, NutritionVerse is 99.0 g, and FPB is 73.3 g / 35.7% MAPE.
  C2 scores 37.5/41.0 g, 100.3 g, and 80.0 g / 38.6% respectively and is
  rejected. C1 repairs FPB ordering to 37/40 triads (117/120 pairwise), so
  ordinal Probe D is parked. Width-normalized calibration follows point
  selection; the max-source phone-union rule covers NV at 84.0% and FPB at
  97.3%, requiring the explicit wide-interval confirm path.

Stage 2 adapter training and Stage 3 device/product gates remain blocking.
The local adapters, database sources, generated manifests, checkpoints, and
compiled model remain gitignored as required.
