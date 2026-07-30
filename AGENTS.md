# SeeCal — agent guide

Photo → calories/macros/ingredients, on-device, via a Qwen3.5-4B LoRA
fine-tune. Full story: top-level `README.md`. Architecture + the ml/iOS
contract: `docs/architecture.md`. Blog-ready narrative of every interesting
failure and result: `docs/blog-inputs.md`. This file is the short, working-memory
version for agents — see `docs/` for everything it points to instead of
repeating.

> Renamed from `CLAUDE.md` (2026-07-28) to the cross-agent `AGENTS.md`
> convention. If your harness only auto-loads `CLAUDE.md`, read this file
> explicitly at session start.

## Repo map

```
ml/       training pipeline (Python + mlx-vlm). ALL commands run from ml/ —
          see ml/README.md for setup -> prep -> train -> eval -> convert.
ios/      SeeCal SwiftPM package (ios/SeeCal) + thin XcodeGen app wrapper
          (ios/App). See ios/README.md and ios/SeeCal/README.md.
scripts/  build.sh, test.sh, gen-xcode.sh, make-appicon.py,
          release-{setup,testflight,appstore}.sh — the only supported way to
          build/test/release the app.
docs/     architecture.md, training-history.md, blog-inputs.md, third-party.md,
          BACKLOG.md, specs/ (binding app spec), design/ (prototype = binding
          visual spec), plans/ (dated planning docs, historical).
attic/    retired one-off debugging/patch scripts, kept for reference only.
```

## ⚠️ Local-only artifacts (NOT in git — read before assuming a clean clone works)

Model weights, datasets and adapters are gitignored by design. On this machine
they exist; on a fresh clone they do not, and some cost hours to rebuild:

| Path | Size | How to regenerate |
|---|---|---|
| `ml/adapters_v8_numeric_4b/` | 1.3 GB | retrain from `finetune_data_v8_numeric` (see visual-specialist plan) |
| `ml/adapters_v8_numeric_4b_swift/` | 128 MB | `cd ml && ./convert.sh adapters_v8_numeric_4b --version v8-conditioned` |
| `ml/runs/visual-specialist/deployment/SeeCalVisualSpecialist.mlmodelc/` | 9.5 MB | export S1 best checkpoint, then `xcrun coremlcompiler compile` |
| `ml/runs/factored/scale-v1/` | 33 MB | `make_scale_data.py`, then `python -m visual_specialist.scale train/evaluate/export` |
| `ml/runs/factored/scale-v2-probe-b-nv-1024/` | ~17 MB | regenerate `datasets/scale_v2_nc_1024`, then use the run config preserved beside the checkpoint |
| `ml/datasets/fdc/seecal-nutrition.sqlite` | 7.2 MB | `cd ml && ./download_fdc.sh`, then apply the reviewed base and NutritionVerse official-Train alias files with `make_alias_table.py` |
| `ml/datasets/foodseg103/` | 2.3 GB incl. download cache | `cd ml && .venv/bin/python download_foodseg103.py` |
| `ml/datasets/nutritionverse-real/` | 2.1 GB incl. archive | `cd ml && ./download_nutritionverse_real.sh`, then `make_nutritionverse_eval.py` |
| `ml/finetune_data_id_v2_frb_resolved/` | ~15 MB + image cache | regenerate the database/aliases first, then run the fully-resolved, 1024-edge command in `ml/README.md` |
| `ml/adapters_v7b/` | 1.3 GB | **~3.5 h retrain** (see "Reproducing v7b") |
| `ml/adapters_v7b_swift/` | 124 MB | `cd ml && ./convert.sh adapters_v7b --version v7b` |
| `ml/adapters_v5/` | 1.3 GB | ~3.5 h retrain on `finetune_data_v2` |
| `ml/Nutrition5K/` | 22 GB | `cd ml && ./download_dataset.sh` |
| `ml/dataset_clean/` | 2.6 GB | `cd ml && ./prep.sh` |
| `ml/negatives/`, `ml/coco/` | 48 + 19 MB | `cd ml && ./download_negatives.sh` then `make_negatives.py jsonl` |
| `ml/finetune_data_v7b/` | 5.1 MB | `make_v7_data.py --neg-train-cap 100 --out-dir finetune_data_v7b` |

**`adapters_v8_numeric_4b` is the shipping adapter; `adapters_v7b` remains the
protected rollback baseline.** Both exist only on this machine. Do not delete
either casually.

## Conventions

- **Run every `ml/` command from `ml/`.** JSONL image paths are relative to
  that directory, not the repo root — running from elsewhere silently breaks
  image loading (fails loud in `smoke_test.py`, not silently, if you do it
  right).
- **Spec wins over code for the app.** `docs/specs/2026-07-26-app-spec.md`
  and `docs/design/prototype/seecal-prototype.html` are binding — the
  prototype HTML is the visual spec (colors, spacing, motion), not a
  reference sketch. If the code and the spec/prototype disagree, the
  spec/prototype is right until a human says otherwise.
- Never commit datasets, model weights, or adapters (`.gitignore` already
  covers `ml/Nutrition5K/`, `ml/adapters*/`, `*.safetensors`, `ml/runs/`,
  `ml/coco/`, `ml/negatives/`, `ml/finetune_data_v7*/`).
  Never commit release credentials (`.secrets/`).
- **Delegation policy (user's standing rule, 2026-07-27):** Opus for complex
  tasks and planning; Sonnet for simple, well-specified tasks. **Do not use
  Fable.** Never dispatch a cheaper model on open-ended design.

## Build & test

```bash
cd ml && .venv/bin/python -m pytest tests/        # ml pipeline unit tests (90)
cd ios/SeeCal && swift test -Xcxx -DFMT_CONSTEVAL= # Swift package tests (219; 3 env-gated skips)
scripts/test.sh                                    # both of the above + iOS build check
scripts/test.sh --skip-build                       # skip the slow xcodebuild step
scripts/build.sh --device                          # signed device build (bundles weights)
scripts/gen-xcode.sh                               # regenerate + open Xcode project
```

`-DFMT_CONSTEVAL=` works around Xcode 26's clang being stricter about
`consteval` than the `fmt` vendored inside `mlx-swift`. If you open
`ios/SeeCal` directly in Xcode instead of the CLI, the auto-generated scheme
is named `SeeCal-Package` — add the same flag to its build settings or
you'll hit the same error there. The app wrapper's own xcodebuild scheme
(via `scripts/build.sh`, `ios/App/project.yml`) is `SeeCal`, already wired
with the flag. `ios/SeeCal/Package.swift` pins a **fork**,
`ekabanov/mlx-swift@seecal-fmt`, which carries the flag in Cmlx's own
manifest — SwiftPM packages don't inherit project-level C++ flags, so the fork
is the only way Xcode GUI builds work.

Device builds need `.secrets/release.env` (written by
`scripts/release-setup.sh`) for `BUNDLE_ID` / `DEVELOPMENT_TEAM` / `MODELS_DIR`.
`project.yml` reads these via XcodeGen `${VAR}` substitution, which is why
regenerating through `scripts/gen-xcode.sh` (not bare `xcodegen`) is required —
otherwise the signing team is wiped to the placeholder on every regen.

**macOS test runs never compile the `#if os(iOS)` camera code**
(`AVFoundationCaptureService.swift`). Only an iOS build typechecks it — that's
why `scripts/test.sh` includes a build check. Don't claim camera changes are
verified off a green `swift test`.

## Key gotchas

- **Adapter↔prompt byte-parity is the single most load-bearing invariant in
  this project.** Training and inference must build the *exact* same prompt
  text (no system message, same vision-token placement) or the LoRA adapter
  learns a mapping the app never actually sends it. Enforced by
  `ml/check_prompt_parity.py` and `ml/smoke_test.py`. See
  `docs/architecture.md` for the full contract and
  `docs/training-history.md` for every way this broke before it was fixed.
- **Teach new behaviour via the COMPLETION, never the prompt.** v7b adds
  not-food refusal with a prompt byte-identical to v5's, so parity holds by
  construction and the iOS app needed no prompt change. Reuse this pattern.
- **Always pair by dish id when comparing adapters.** Raw MAE headlines are
  computed over different n (parse failures are excluded per-adapter), so
  v5's "59.0 (n=322)" vs v7b's "63.4 (n=324)" is an unlike-set comparison that
  looks like a regression and isn't — on the 324 shared dishes v5 is 62.5.
  `ml/runs/eval_full/eval.log` has per-dish errors for pairing. This mistake was
  actually made mid-session and had to be corrected; don't repeat it.
- **`eval.sh` requires `--limit`.** `infer.py`'s own `--limit` silently
  defaults to 20 and once truncated a full overnight eval without warning.
  `eval.sh` refuses to guess — pass it explicitly (325 for the full
  committed test set).
- **Metal memory doesn't swap.** Don't run `train.sh` alongside anything
  else GPU-heavy (LM Studio, `eval.sh`, a second training run) — co-tenancy
  silently OOM-kills training instead of failing cleanly. Evals are also
  single-tenant: chain them, don't parallelise.
- **`infer.py` output is block-buffered through a pipe.** Piping an eval into
  `tee` means per-dish lines don't appear until the run ends, and the summary
  JSON is only written at the very end. To judge liveness mid-run, check the
  process's accumulating CPU time rather than assuming a hang.
- **torchvision must be installed explicitly.** mlx-vlm's own `[train]`
  extra doesn't pull it in, but the Qwen3VL processor imports it at init
  time. `ml/setup.sh` installs torch/torchvision as a separate step for
  exactly this reason.
- **iOS memory: cap the MLX cache.** `MLX.Memory.cacheLimit` defaults to
  roughly the device memory limit, so MLX retains GBs of freed buffers and the
  app gets jetsam-killed (peaks were 4.6–5.5 GB). Fixed in
  `MLXQwen35RunnerBuilder` with `cacheLimit = 48MB` plus `clearCache()` after
  fuse/warmup/each inference, and a memory-warning handler in
  `ProductionRootView`. Verified on device: a real memory warning fired and the
  app survived. The Xcode debugger itself inflates memory — test non-attached
  before believing a regression.
- **MLX cannot create a Metal device on the iOS Simulator** — an uncatchable
  C++ `SIGABRT`, not a Swift error. `ProductionRootView` uses
  `#if targetEnvironment(simulator)` to fall back to the mock engine. Never try
  to "fix" simulator crashes with try/catch.
- **App icons must be full-bleed, opaque, square-cornered 1024².** iOS applies
  its own squircle mask. `scripts/make-appicon.py` converts mockups (rounded
  corners + drop shadow + white page) into valid assets; a missing 120×120
  variant previously failed App Store Connect validation (error 90022), which
  is why `project.yml` sets `CFBundleIconName`.

## Current state (2026-07-30)

- **Factored pipeline migration (2026-07-29): Stage 0–1 implemented in
  shadow, not shipping.** Frozen IDENTIFY schema/parity/smoke gates, USDA
  SQLite RESOLVE ladder, paired E0/E1/E2 harness, deterministic Swift
  ASSEMBLE, confirmation/disagreement gates, concurrent pipeline shape, and
  mass-only Core ML runtime are present. Baseline v8 hard-mistake audit:
  HMR 12.9%, DVR 40.5%, AIR 31.6%. The original percentage IDENTIFY contract
  failed closure at full scale and is superseded by v2 `portion_units`;
  deterministic Python/Swift code normalizes units to 100. The finished
  percentage adapter remains a normalized compatibility baseline (fast
  Nutrition5K HMR 12.0%, NutritionVerse HMR 33.6%). SCALE-v2 trained on
  N5K + NutritionVerse official Train reaches 36.9 g N5K overhead MAE,
  41.3 g N5K side equal-group MAE, and 96.2 g NutritionVerse official-Val
  equal-scene MAE, with calibrated coverage 80.3% / 82.4% / 81.3% (Wilson
  95% intervals 75.6–84.3 / 77.7–86.2 / 70.4–88.9). The oracle audit passes
  hidden-label median bias (+1.29%) but shows SCALE is now the blocker:
  true-mass oracle assembly is 10.3 kcal MAE on the shared clean set, current
  SCALE-P50 oracle is 56.9, and v8 is 29.5. A validation-selected SCALE point
  correction was rejected after worsening frozen NV MAE (96.2 → 98.8 g).
  The sensitivity sweep says SCALE must cut its current mass error by about
  52% (roughly 36.9 → 18 g MAE) before the oracle candidate crosses v8.
  A fixed FNDDS importer now retains 5,430 composite rows; the database has
  13,589 profiles/196 category medians, and train-only NV aliases make its
  official Val fully assemblable. The leakage-safe FRB ablation is 1,905/212
  records with 4,122/4,122 item occurrences resolving at rungs 1–2.
  Probe B also fails FPB zero-shot: 165.7 g equal-group MAE / 55.3% MAPE /
  20.9% interval coverage on 2,123 clean-weight test images (164 groups).
  Occupancy diagnostics show a compressed FPB mapping rather than missing
  numerical target support; NV predictions are over-coupled to framing.
  Simple USDA portion-prior fallbacks/fusions regress N5K and NV and are
  rejected. FPB clean training/validation is now 8,929/2,170 records with
  source-capture grouping, unknown weights excluded, and two frozen-test
  capture overlaps removed. Probe C1 (+FPB center crop) and C2 (matched
  letterbox) completed with worst-source equal-group-MAPE selection and a 10%
  Pareto guard. C1 wins: N5K overhead/side 40.1/42.0 g, NV 99.0 g, and FPB
  73.3 g / 35.7% MAPE; C2 is 37.5/41.0 g, 100.3 g, and 80.0 g / 38.6%.
  C1 repairs FPB size ordering from 23/40 to 37/40 triads, so ordinal Probe D
  is parked. Width-normalized phone-union calibration exposes nonexchangeable
  domains: pooled undercovers NV (70.1%); max-source reaches NV 84.0% but FPB
  97.3%, reinforcing the wide-interval confirm path. Oracle assembly with C1
  mass improves the 72-dish shared-clean N5K candidate from 56.9 to 38.5 kcal
  MAE but still trails v8 at 29.5; all-N5K barely moves (64.2 → 63.4) and the
  NV quality slice regresses (187.5 → 191.2). On a deterministic
  one-view-per-scene NV screen, C1 mass beats v8-implied mass 88.4 vs 156.8 g
  MAE on 66 shared valid scenes and wins 66.7% of scenes; one v8 generation
  fails parsing. The teacher-required regret decomposition changes the
  binding-constraint decision: the true-mass floor is 53.5% of total C1
  calorie MAE on all complete N5K groups and 59.7% on raw NV Val, versus
  26.7% on clean-72 and 41.0% on the NV quality slice. Per the pre-agreed
  half-total rule, IDENTIFY-v2 LoRA resumes with a fresh current-contract
  memorization gate, FoodSeg primary arm, then matched +BF2 ablation. The
  FoodSeg manifests now cap images at 1,024 px after the first tiny run caught
  a 2,048-token truncation mismatch before any weight update. The corrected
  20-epoch probe reached loss 0.000125 and reproduced 32/32 names, containers,
  and portion units exactly with no parse/schema failure or repair; the
  two-epoch 7,677-record FoodSeg primary run is active at
  `runs/factored/e3-v2-foodseg-1024/adapter`. USDA serving
  evidence remains gate-only; Swift now sums one selected item portion and
  never averages across serving kinds. Stage-3
  device/product gates still block shipping; v8 remains the shipping/rollback
  system.

- **Shipping system: conditioned 4B + S1 specialist.** The 4B adapter is
  `adapters_v8_numeric_4b`, stamped `v8-conditioned`; the compiled specialist
  is `runs/visual-specialist/deployment/SeeCalVisualSpecialist.mlmodelc`.
  Full test325: 56.8 kcal MAE / 29.7 median, 0 parse failures, 29/29 held-out
  negative refusal. Paired with v5 on 322 dishes: −1.8 kcal (95% bootstrap CI
  −8.3 to +4.7), a directional but non-significant improvement. Swift runs the
  specialist first and appends its calibrated mass/calorie/macro intervals.
  Exact contract and results:
  `docs/plans/2026-07-28-visual-specialist-conditioned-qwen-plan.md`.
- **Rollback baseline: `adapters_v7b`** — v5's food data + 100 not-food
  negatives. Emits `{"not_food": true}` on non-food photos. **All five gates
  green:** 100% refusal recall (29/29 held-out negatives), 0 over-refusal on
  324 real-food dishes, food accuracy a statistical tie with v5 (paired
  v7b−v5 = +0.87 kcal, t=0.23) with a BETTER median (30.5 vs 35.2) and fewer
  parse failures (1 vs 3), parity PASSED, and a user-verified device spot check.
  Track complete: `docs/plans/2026-07-27-v7-notfood-plan.md`.
- `adapters_v7` (221 negatives) is dead: identical perfect refusal but a REAL
  food regression (paired +10.48 kcal, t=2.49). **100 negatives is the right
  dose** — refusal is trivially learned, so if over-refusal ever appears on
  device, cut negatives further rather than abandoning the approach.
- `adapters_v6` (depth-augmented) is a statistical tie trending slightly worse
  (paired v6−v5 = +3.5 kcal, t=0.89 on 322 shared dishes) and is not shipped.
  Full 325-dish v5-vs-v6 eval DONE (2026-07-27); **depth track closed**. Raw
  results in `ml/runs/eval_full/`. `adapters_v1`–`v4` are all confirmed dead —
  see `docs/training-history.md`.
- **Base-model choice settled.** Untuned Gemma 4 E4B scored MAE 159.4 vs
  untuned Qwen 83.4 and tuned Qwen v5 54.4 (same 50 dishes) — ~3× worse — and
  E4B is 5.15 GB plain / 7.48 GB OptiQ, too big to bundle. Qwen3.5-4B stays.
  Fine-tuning, not base-model choice, does the heavy lifting.
- **iOS app**: full product build-out complete against the prototype spec.
  219 Swift (3 env-gated skips) + 90 ml tests; `scripts/build.sh` /
  `scripts/test.sh` green. The signed device build bundles the
  `v8-conditioned` adapter and 9.5 MB compiled Core ML specialist. The real
  specialist Core ML scaffold passes on macOS; physical-device
  latency/memory/thermal validation remains required.
- Camera-first manual and barcode entry shipped (2026-07-28): Scan still opens
  the viewfinder; Manual opens a photo-less editable draft, while EAN/UPC/Code
  128 detection shares the photo capture session and resolves normalized GTINs
  through a cached Open Food Facts v3 lookup. Package-label nutrition stays
  source-backed and resettable; missing fields fall back to manual entry without
  inventing zero values.
- Editable nutrition shipped (2026-07-28): meal and ingredient names, grams,
  calories/macros, manual add, delete with Undo, model reset, corrected-density
  scaling, backward-compatible persistence, and confirmed logged-meal deletion
  with photo cleanup. Meal totals remain derived from ingredients. The unused
  LiDAR setting was removed; depth affordances are capability-driven.
- Camera capture hardened (2026-07-28): watchdog timeout on `capturePhoto()`,
  `AVCaptureSession.runtimeErrorNotification` observed, and no permanent failure
  latch in `configureIfNeeded()`. Exactly-once completion lives in
  `PendingCaptureRegistry`, kept platform-agnostic so it is unit-testable on the
  macOS host.

## Reproducing v7b from scratch

```bash
cd ml
./download_dataset.sh                     # Nutrition5K (22GB)
./prep.sh                                 # dataset_clean + finetune_data_v2
./download_negatives.sh                   # COCO annotations + 300 sampled negatives
.venv/bin/python make_negatives.py jsonl  # applies negatives_cull.txt (23 food leaks)
.venv/bin/python make_v7_data.py --neg-train-cap 100 --out-dir finetune_data_v7b
.venv/bin/python check_prompt_parity.py   # MUST print PASSED before training
.venv/bin/python smoke_test.py --data finetune_data_v7b/train.jsonl --records 8 --batch-size 4
DATASET=finetune_data_v7b OUTPUT_PATH=adapters_v7b ./train.sh   # ~3.5h, ~58GB Metal
./eval.sh adapters_v7b --limit 325 --out-json runs/eval_v7b_food325.json
./eval.sh adapters_v7b --limit 29 --test-set negatives/test.jsonl --out-json runs/eval_v7b_neg.json
./convert.sh adapters_v7b --version v7b   # -> adapters_v7b_swift for iOS
```

Negatives come from COCO val2017 with **every food-supercategory image
excluded**. COCO annotates only 10 food types, so dining scenes with
un-annotated food (rice, plated meals, produce) still slip through — 23 such
leaks were found by visual review and are listed in `ml/negatives_cull.txt`
(regenerate the review sheets with `make_contact_sheet.py`). Training refusal on
a photo that contains food is exactly the over-refusal failure to avoid.

## Open work

See `docs/BACKLOG.md` for detail. Highest-value items:

1. **An honest out-of-distribution test set.** All 325 test dishes are
   cafeteria trays on one RealSense rig, so our MAE almost certainly flatters
   real phone photos. NutritionVerse-Real (889 real images, weighed ground
   truth) is the best candidate. Note Nutrition5K is *not* single-food —
   63.3% of dishes are multi-ingredient (mean 5.71, max 34); the weakness is
   scene diversity, not mixed food.
2. **On-device depth (v6)** — blocked on calibrating iPhone LiDAR to the
   Nutrition5K rig scale; that calibration is the entire difficulty.
3. **Further multi-view Qwen training (v9?)** — `side_angles` (cameras A–D, ~115
   frames/dish) are downloaded but unused: a 1920×1080 side frame is ~2040
   image tokens against `--max-seq-length 2048`, so those records would
   truncate. Needs downscaling first. Would take training data from 2,594 to
   ~9,400 records.
4. **HF publish of the fused model: DEFERRED** by the user ("wait with
   publishing"). Ready via
   `ml/fuse.sh adapters_v7b --out-path fused_v7b --upload-repo <id>`.
   Any publish is a separate user-confirmed step.
