# Blog inputs — the interesting parts of building SeeCal

Raw material, roughly chronological, written to be self-contained: everything needed
to draft a post is in this file. Numbers are real and reproducible from the repo
(ledger: `.superpowers/sdd/progress.md`; history: `docs/training-history.md` after R6).

## Context: what this project is
**SeeCal**: point an iPhone camera at a meal, get calories, macros, and a
per-ingredient breakdown — entirely on-device, no cloud. Two halves:
- **ML pipeline** (`ml/`): LoRA fine-tuning of **Qwen3.5-4B** (a natively multimodal
  4B-parameter open model, Apache 2.0) on **Nutrition5K** (Google's dataset of ~5,000
  real cafeteria dishes, each weighed ingredient-by-ingredient with ground-truth
  calories/macros, photographed overhead with an RGB-D camera; CC BY 4.0). Training
  runs locally on a Mac Studio (M3 Ultra, 96GB unified memory) using **mlx-vlm**
  (Apple's MLX ecosystem's vision-language training library). LoRA rank 16, alpha 32
  (effective scale 2 under the modern alpha/rank convention), batch size 1,
  seq len 2048, LR 1e-4, loss on completions only. The model's answer format is a
  strict JSON schema: `total_calories, protein_g, fat_g, carbs_g, items[{name,
  estimated_grams, calories, protein_g, fat_g, carbs_g}]`.
- **iOS app** (`ios/`): SwiftUI app running the fine-tuned model via **mlx-swift**;
  the base model ships 4-bit quantized (~2.3GB) inside the app bundle, and the LoRA
  adapter (~35MB) is loaded and fused into the weights at app launch.

A critical invariant ties the halves together: the prompt the app sends at inference
must be **byte-identical** to the training prompt prefix (same folded-in system text,
same image-token placement, no system message — Qwen3.5's chat template forbids array
content in the system role while the datasets layer demands uniform types, so the
system prompt is folded into the user turn). A committed parity-check script asserts
this against the actual trainer code path.

The work described below spans one intensive day of orchestrated multi-model
development on top of months of prior (mostly failed) training attempts, all
reconstructed and verified from the repo itself.

## The four dead adapters (and four different silent failure modes)
Fine-tuning a 4B multimodal model locally produced four consecutive adapters that
trained "successfully" and were garbage, each for a *different* silent reason:
1. **Run 1**: the `<image>` placeholder in the training JSONL was tokenized as literal
   text. The model learned a text→JSON mapping with zero visual grounding.
2. **Run 2**: a framework quirk (`use_embedded_images` for qwen models) meant the
   vision encoder never ran during training — no `pixel_values` in any batch. Loss
   went down anyway.
3. **Run 3**: `--train-on-completions` was silently disabled because the default
   assistant-token id (77091) decodes to `'[hash'` in Qwen3.5's tokenizer and never
   appears in any sequence. Loss was computed over image tokens + the prompt + the
   answer, and the model collapsed to predicting the most frequent ingredient —
   olive oil, which appears in 48% of Nutrition5K dishes. An adapter that answers
   "olive oil" to everything.
4. **Run 4**: died overnight between iteration 800 and 900, unnoticed; reconstructed
   months later purely from file timestamps (no log survived). Bonus bug: an
   `--iters`-vs-dataset-size clamp meant it only ever saw the first 1000 of 2594
   training examples.

Lesson that shaped everything after: **loss going down proves nothing** for multimodal
fine-tuning. Every failure was invisible in the loss curve.

## The validation ladder
The fix for "every debug cycle costs a day": a ladder of gates before any long run —
a ~1-minute smoke test that assembles one real training batch and asserts the things
that actually failed historically (pixel_values present, image-token count == vision
grid product, sequence not truncated, completion mask sane and excluding image
tokens), then a 32-dish overfit run (can it memorize at all?), then a 1000-iteration
probe with a real eval, then the full run. Roughly 1.5 hours of gates replaced
multi-day blind runs. v5 — the first working adapter — passed every rung first try.

## Don't trust your own docs
The project's CLAUDE.md confidently documented five hand-patches to the ML framework
as load-bearing. Re-verification against pristine checkouts showed they *were* real —
but the newer upstream release had fixed all five, and two NEW migration traps had
appeared (image loading moved to a different JSONL field, silently; the LoRA scale
convention changed from bare-alpha to alpha/rank, silently changing effective scale
from 32 to 2). "Verify claims empirically before building on them" became a standing
rule after that.

## v5: the payoff
Calorie MAE 54.4 kcal (median 36.2) on held-out dishes vs 83.4 for the base model
(−35%), carbs error −61%, ~one parse failure in 50. First working adapter in five
attempts. Total training time ~3.5h on an M3 Ultra at ~0.45 it/s, peaking ~47–58GB
of Metal memory.

## Speed experiments: batching made it slower
Counterintuitive result: batch size 4 *lost* to batch size 1 (0.336 vs 0.452
samples/s) because dish JSON answers have wildly variable lengths and padding waste
dominates. Disabling gradient checkpointing bought +15% throughput for ~5× the
memory (10.8 → 47GB) — adopted, with a hard "nothing else on the GPU" rule after a
batch-4-no-checkpoint experiment estimated ~160GB and OOM-killed the whole machine.
Also: a benchmark run was nearly poisoned by WindowServer eating 38.8% GPU
(a webcam app was suspected); after a restart the numbers were identical anyway.

## The glass platform trap (depth track)
Nutrition5K's depth camera looks down at dishes on a **glass** platform. Infrared
depth sees *through* glass: the measured "table" is the real table 4.8cm below the
capture plane. A plane fit to border pixels therefore inflates every food height by
~48mm — producing 1–4 *liter* volume estimates per plate. The insidious part: volume
still correlated with mass (r=0.57) because a constant bias preserves correlation —
the correlation gate PASSED on physically absurd geometry. The discriminating test
was a physics sanity check: implied density (mass/volume) must land in 0.3–1.2 g/ml.
Broken geometry gave 0.06 g/ml — food fifteen times lighter than water. After the
fix (central-disk reference mode, focal length re-derived as 465.1px from the paper's
published per-pixel area rather than the spec-sheet 616): volumes 80–840 ml,
density 0.37 g/ml.

## The probe paradox
Full context: portion size is the hard part of photo calorie estimation (the model
can identify chicken, but 90g vs 150g is a guess), and Nutrition5K ships depth maps,
so the plan was to feed the model a portion-size hint derived from depth — a text
line appended to the prompt: "Estimated food volume from depth sensor: ~412 ml
(max height 41 mm)." — with 12% of training examples deliberately lacking the line
so phones without LiDAR stay in-distribution. The experiment design used matched
probes: 1,000-iteration training runs of with-depth vs without-depth, evaluated on
the same 50 held-out dishes, because comparing a cheap probe against the fully
trained baseline would be unfair in the other direction (the existing full model
had 5× the training).

Result: the depth probe beat the matched no-depth probe by **33%** calorie MAE
(68.6 vs 102.3). Decisive — so the depth variant graduated to a full 2-epoch run.
At full training: **statistical tie** with the no-depth model (59.2 vs 54.4 MAE on
50 dishes; paired per-dish difference +5.2 kcal, t=0.56, wins 26/loses 22). The
feature accelerated early learning and then converged away — given enough steps,
the model extracts equivalent portion signal from pixels alone. The gate held:
the depth track was stopped rather than shipping an iOS LiDAR capture pipeline,
a Swift port of the plane-fit geometry, and a second sensor dependency for zero
measured gain. Consistent with the literature on closer reading: Nutrition5K's own
depth ablation gained from *ground-truth-calibrated* depth; ours was estimated from
a plane fit with a plate-inclusive bias.

## Bugs found upstream and down
- mlx-vlm's trainer cannot train on multi-image records at all (grid tensor shape
  `(1,2,3)` where the vision tower expects `(2,3)`) — in the release AND git main;
  reported upstream with root cause and fix. It's why depth-as-image was never trained.
- The trainer's completion mask is off by ~2 tokens (the mask prefix ends with
  `<think>\n` that the actual training text doesn't contain) — benign, identical
  across all experiment arms, but the first two JSON tokens escape the loss.
- Our own eval script's `--limit` defaulted to 20 — silently turning a planned
  overnight 325-dish eval into a 20-dish one. Caught by reading the log mid-run;
  the new eval entrypoint refuses to run without an explicit limit.

## Data archaeology
Context: the pipeline reads two "tidy" CSVs (`dish_nutrition_values.csv`,
`dish_ingredients.csv`) that had always just been *there* on disk. While writing the
open-source dataset-download script, it emerged that these are not part of the
official Nutrition5K distribution at all — they were derived locally, at some point,
from Google's raw variable-width metadata format, and no converter existed anywhere.
A fresh clone of the "open-source-ready" repo could not have gone from download to
training. Investigating the derivation surfaced the second find: the tidy files
derive from `dish_metadata_cafe1.csv` ONLY. Cafe2's 238 dishes (228 of them with
usable overhead imagery) were silently dropped — apparently an oversight, years old,
inherited by every experiment since, including all published metrics. Two independent
investigations (one spawned specifically to write the converter, one inside the main
reorg task) converged on the same conclusion and the same verification bar: the new
converter's default mode reproduces the historical CSVs **byte-for-byte** from the
raw files (so all existing metrics remain comparable), and a `--cafes all` flag
recovers the lost 228 dishes for future runs. The eval/train split, adapter
checkpoints, and published numbers all stay honest because history was matched
exactly rather than "fixed" in place.

## Evaluation is the hidden cost
Training the adapter: ~3.5h. Evaluating it *properly*: ~2.5h of GPU time per adapter
per full 325-dish test set, because each dish is a full multimodal generation
(~15–30s each at ~220 output tokens of JSON). The 50-dish evals used for gating run
~25 minutes and resolve only ~±15 kcal differences — which is exactly why the
depth-track tie needed the full paired eval to call. Statistical discipline
(paired per-dish differences on identical dish sets, not headline MAEs from
different runs) changed the conclusion at least once.

## Prototype-driven product development
The iOS app was designed as a clickable HTML prototype first — iterated live in
~15 rounds of feedback (tab bar layout, a kg-per-week slider with a recommended
band replacing preset chips, onboarding wizard, goal math shown transparently as
"BMR 1,743 × 1.55 = 2,701 − 550 = 2,150"). Then the prototype file itself was
committed as the *binding visual spec*: implementation agents had to read its CSS
and match it, and the exact user-facing strings are pinned in unit tests down to
the typographic minus sign. Product decisions that came straight from prototype
questions: analysis continues in the background if you navigate away (a 20-second
on-device inference is too expensive to discard) and surfaces as a tap-to-review
banner; nothing is ever logged without explicit confirmation; every AI estimate is
an editable draft (gram steppers rescale per-item nutrition linearly).

## Multi-agent development, honestly assessed
The whole day ran as tiered multi-model orchestration, with an explicit division of
labor set by the human: the top-tier model (the orchestrator) did research passes,
specs, execution plans, decision gates, and reviews — and personally implemented the
few load-bearing correctness steps (the prompt-parity gate, the paired statistics);
a mid-tier model implemented the complex tasks (the concurrency-heavy scan flow, the
onboarding surface, release tooling); a cheaper model implemented well-specified
single-concern tasks. Each task = one fresh agent with a tight brief pointing at the
spec sections and exact files; tasks ran strictly serially on shared code. Two rules
made it work: **specs make all design decisions** (implementers were explicitly told
"the spec wins over existing code; report BLOCKED rather than improvise"), and
**nothing is trusted from a report** — the orchestrator re-ran every test suite
itself after every task, which matters because agents' reports were uniformly
confident and occasionally the environment (stale editor diagnostics, broken venv
shebangs) would have misled a trusting reader.

The independent whole-branch review at the end earned its cost concretely: it found
the history chart rendering **upside-down** (bars anchored to the top of the chart
area while the goal line measured from the bottom — a one-line SwiftUI frame-
alignment bug that shipped because nothing tested view geometry), a failed inference
permanently wedging the scan flow (the error screen's only affordance was Retry),
a double-tap on "Log meal" persisting the meal twice (each commit minted a fresh
UUID), and a camera-close race that could start an unwanted 30-second inference on
an abandoned capture. Nine findings, all real, all fixed with regression tests in
one wave. Final state: 171 Swift tests + 41 pipeline tests, plus an iOS-destination
build check that exists specifically because macOS test runs never compile the
`#if os(iOS)` camera code — a gap one agent flagged in its own report.

Also honest: the orchestrator made its own mistakes — launching the "overnight full
eval" with the script's silent 20-dish default limit, and a working-directory slip
that briefly aimed a git command at the wrong tree. Both caught by verifying rather
than assuming.

## Deployment math
A 4B model at 4-bit is ~2.3GB. Apple's app-size cap is 4GB (the 500MB limit applies
only to executable code), so the weights ship *inside* the app — no downloader, no
hosting. LoRA adapters fuse into the quantized weights at load time, so model-quality
updates are ~35MB app updates. On-device inference: 15–30s per photo on an
iPhone 15 Pro-class device (the wait became a designed, staged progress screen).

## Numbers people ask about
- Dataset: ~4,768 dishes with per-ingredient ground truth (Nutrition5K, CC BY 4.0).
- Training: 2,594 examples, 2 epochs ≈ 5,188 iterations ≈ 3.5h on M3 Ultra.
- Best adapter: calorie MAE 54.4 / median 36.2 kcal (50-dish eval; 325-dish pending).
- Five training runs to get one good adapter; zero of the four failures visible in loss.
