# SCALE Probe C — teacher ruling

**Date:** 2026-07-30
**Reviewing:** `docs/reviews/2026-07-30-scale-issues-teacher-feedback-response.md`
**Prior feedback:** `docs/reviews/2026-07-30-scale-issues-teacher-feedback.md`

## Corrections accepted — the diagnostics did their job

Two of my sub-hypotheses are refuted by measurement and I withdraw them:

1. **The mass-support gap is dead.** NutritionVerse training (median 315 g,
   P90 724 g) covers FPB's range; only 3.7% of FPB groups exceed the NV P90.
   The failure was domain *mapping*, not label range — and C1 proves it, since
   adding FPB images (whose targets were already numerically supported) cut
   FPB MAE 55.8%.
2. **The letterbox/geometry hypothesis under-delivered exactly as the
   occupancy simulation predicted.** C2 rejected on the evidence; the
   diagnostic-before-run discipline saved that from being a confounder.

The occupancy audit also *sharpened* the diagnosis usefully: FPB was
compressed-mapping (signal present, r = 0.83, mapping flat), while
NutritionVerse shows a genuine occupancy shortcut (prediction twice as
occupancy-coupled as truth). Keep that audit as a standing per-probe artifact —
it's now the cheapest lens we have on *what* SCALE learned.

## Approvals

- **C1 selected, C2 rejected, Probe D parked** — all approved on the stated
  evidence. 37/40 triads and 117/120 pairwise orderings means the ordinal
  auxiliary loss has nothing left to teach; parking it is correct.
- **Portion-prior conclusion approved**: experimental independent evidence
  only, no production fusion, no automatic fallback. Note the picture moved
  under it: after C1, SCALE beats the prior on FPB too (73.3 vs 120.5 g), so
  the fusion urgency from my previous feedback is gone. Two riders:
  (a) **replace the Swift disagreement-gate comparator** with the audit's best
  aggregation (`sum_item_portions`) — the 412 g `median(portion/share)`
  formula explains the 32.3% false-fire rate on Nutrition5K; the gate keeps
  its gate-only role but needs a sane comparator, then re-measure fire rates;
  (b) the pizza example (112 g slice vs 897 g whole → 504.5 g median) is a
  data-shape lesson worth a regression test: portion aggregation must never
  average across portion *kinds*.
- **Calibration stopgap approved**: domain-robust max-multiplier with
  NV 84.0% / FPB 97.3% conservative is the right side to err on, and the
  conclusion stands — conformal cannot make shifted domains exchangeable, so
  the confirm/fallback UX carries the tail. Threshold validation on
  validation splits only.
- **v8-on-FPB baseline: dropped, don't spend the hours.** The NV screen
  settled the question it existed to answer (v8 implied mass 156.8 g vs C1
  88.4 g, C1 wins 66.7% of scenes, paired −68.4 g): decoder mass is decisively
  worse OOD, the Qwen-mass alternative is closed, and no decision changes
  based on how badly v8 does on a second phone domain. Log it as answered.

## The one blocking demand: decompose the calorie regret before any
## binding-constraint claim

The regret table contains an unexplained anomaly that the report glosses:
C1 improves the shared-clean-72 slice dramatically (56.9 → 38.5 kcal) but
moves *all* N5K complete groups only 64.2 → 63.4, and NutritionVerse not at
all — while FPB mass improved 2×. "SCALE remains the binding constraint" is
asserted, not shown, and the promotion gate depends on it.

Required: extend the oracle assembly table with a **true-mass floor column
per slice**, and report the decomposition explicitly:

- `total regret = (assembly with true mass)  ← non-mass error (resolution,
  share bucketing, taxonomy)`
- `+ (assembly with C1 mass − assembly with true mass)  ← mass-attributable
  error`

for: N5K all-complete-groups, N5K shared-clean-72, NV quality slice, NV raw
Val. The machinery exists; this is a scoring run. It will explain the
72-vs-all anomaly (prediction: the all-groups slice is dominated by
resolution/share error that mass cannot touch) and it converts the
IDENTIFY-pause question from opinion to arithmetic.

**Decision rule, agreed in advance:** for each domain, if the true-mass floor
exceeds half of total regret, the binding constraint there is the
IDENTIFY/RESOLVE side, and the IDENTIFY-v2 LoRA (with the FoodSeg mix and the
BF2 vocabulary gate) resumes — not because mass is solved, but because the
marginal kcal per unit of work has moved. If mass still dominates
everywhere, the pause holds and Probe E below is the next lever.

## Probe E — the forgotten free dataset

The original design reserved **NutritionVerse-Synth (~84k rendered dishes,
exact mass ground truth) for SCALE, gated by E8** — never run. With N5K, NV,
and FPB now all in the mix, NV-Synth is the last large no-gathering mass
dataset on the shelf, and C1 just demonstrated that adding a genuinely new
source is worth ~2× on the target domain. Run Probe E as a data-only arm
under the C1 protocol (same exposure discipline — synthetic must not dominate;
start ≈10–15% effective exposure), frozen tests unchanged, minimax-MAPE
selection, reject on any real-domain regression. Synthetic-to-real transfer
may fail — E8 was gated for that reason — but it is the cheapest remaining
probe with a plausible large payoff, and it needs no license caveats.

## Standing state after this ruling

1. Regret decomposition table (scoring only) — **blocking**, runs first.
2. IDENTIFY LoRA resume/hold decided by the pre-agreed rule above.
3. Probe E (+NV-Synth) — next SCALE lever, independent of 1–2.
4. Gate comparator swap + fire-rate re-measurement — small, non-blocking.
5. Fallback UX threshold validation — proceeds as planned.
6. v8-FPB baseline — closed without running.

The discipline in this cycle — frozen tests, pre-registered rejection rules,
diagnostics before runs, refuting the teacher with histograms — is exactly
what the review structure is for. Keep it.
