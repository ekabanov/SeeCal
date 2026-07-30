# Regret decomposition — teacher ruling

**Date:** 2026-07-30
**Reviewing:** `docs/reviews/2026-07-30-scale-probe-c-teacher-ruling-response.md`
**Prior ruling:** `docs/reviews/2026-07-30-scale-probe-c-teacher-ruling.md`

## Approvals

The decomposition was run as specified, the pre-agreed rule was applied
without relitigating, and the execution hygiene continues to be exemplary —
the token-ceiling catch before any weight update, the generation-based (not
loss-based) v2 memorization gate, the 4,122/4,122 leakage-safe vocabulary
gate, and the portion-kind ingestion fix are all exactly right. IDENTIFY-v2
resumption, the restart sequence, the comparator swap, and Probe E's queue
position are approved.

## One correction: the floor points at RESOLVE, not at the IDENTIFY LoRA

The half-total rule fired, and resuming IDENTIFY work is fine — but be
precise about what the floor *is*. Both columns of the decomposition use
**oracle names and shares**. Therefore the true-mass floor is error that
survives *perfect* identification: resolution quality, share bucketing, and
label issues. A better IDENTIFY model cannot lower the floor by a single
kcal — it can only close the (currently unmeasured) gap between the real
model and the oracle, which sits *on top of* these numbers.

So the branch label "IDENTIFY / RESOLVE" resolves to: **the LoRA is justified
as reaching-the-floor work; the floor-lowering work is resolver-side.** Two
consequences:

1. **Required next scoring run — floor attribution.** For N5K all-complete
   (floor 33.9 kcal) split the floor into: (a) share bucketing — oracle exact
   shares vs. 5-point buckets; (b) items resolved at rung ≥3 — their kcal
   contribution under category defaults; (c) residual rung-1/2 density
   mismatch (raw-vs-prepared variants, the BF1 mechanism). Same machinery,
   no training. Whichever component dominates defines the resolver workstream
   (finer buckets / alias & FNDDS-composite expansion / prepared-variant
   aliasing respectively).
2. **When the primary arm finishes, report the model-vs-oracle gap in the
   same units:** assembly with model IDENTIFY + C1 mass vs. assembly with
   oracle IDENTIFY + C1 mass, per slice. That is the number the LoRA is
   accountable for, and it's currently absent from every table.

## Discount the NutritionVerse-raw trip

NV raw official Val's floor (205 kcal, 59.7%) includes ground-truth nutrient
label noise — the reason the quality slice exists. A floor inflated by label
noise is not model-fixable by either branch, so NV-raw must not drive work
prioritization; the quality slice remains the NV signal (41% — SCALE side).
The resume decision stands on the clean N5K-all evidence alone, which is
sufficient. State this discount in the next report so the rule's precedent
stays clean.

## Non-blocking: disagreement-gate operating point

29.15% in-domain fire rate is defensible as a conservative signal but is a
real UX tax if wired straight to a confirm prompt. Before product
integration: an operating-point analysis on validation data — gate fires vs.
actual assembled-kcal regret (precision/recall or a small ROC), threshold
chosen for target precision. Not blocking any current training.

## Standing state

1. Floor attribution scoring run — next, resolver workstream defined by it.
2. IDENTIFY-v2 primary arm — running; on completion, E3 evaluation under the
   v3 taxonomy plus the model-vs-oracle gap table.
3. Probe E (+NV-Synth) — queued behind the primary arm per Metal
   single-tenancy, order at the agent's discretion.
4. Gate operating-point analysis — before product wiring.
5. v8 ships unchanged; all of the above remains shadow.
