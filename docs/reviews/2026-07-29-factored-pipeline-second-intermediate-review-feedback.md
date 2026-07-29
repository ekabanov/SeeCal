# Factored pipeline — teacher feedback on second intermediate review

**Date:** 2026-07-29
**Reviewing:** `docs/reviews/2026-07-29-factored-pipeline-second-intermediate-review.md`
**Binding design:** `docs/specs/2026-07-29-factored-pipeline-design.md`
**Prior feedback:** `docs/reviews/2026-07-29-factored-pipeline-intermediate-review-feedback.md`

Overall: the two big calls made since the first review are both right. SCALE's
domain overfit (34.3 g in-domain → 300 g / 6% coverage on phone photos) is the
most important empirical result of this track — it is exactly the failure the
OOD gate existed to catch, and Probe B's response (source-balanced mixing,
96.2 g / 81.3% calibrated) is the correct shape of fix. The IDENTIFY-v2
contract change (relative `portion_units`, closure moved to deterministic
code) is endorsed without reservation: v1's 18.8% schema failures + 25.2%
repair rate blew far past the ≤5% repair gate, and moving arithmetic out of
the decoder is this design's own principle applied to its own schema. The
healthiest single number in the report is MAPE parity (~27% on all three
held-out sets) — relative scale has genuinely been learned; the domains differ
in absolute spread, which calibration now represents honestly.

One prior ruling was overridden mid-stream and needs to be re-earned with
evidence (BF1 below).

---

## Blocking findings

### BF1 — Hidden-ingredient exclusion reverses ruling A without quantifying the kcal bias it introduces

Training truth now "excludes hidden recipe ingredients and seasonings."
First-review ruling A kept measured ingredients precisely because, in factored
assembly, dropping measured oil redistributes its mass onto low-density
visible items → **systematic kcal underestimation**, the exact quiet-bias
class this architecture exists to kill. The new evidence (62.5% OOD
hidden-condiment emission; filtering barely moves HMR) is real and justifies
*not training the model to recite invisible items* — but the bias consequence
was never measured, and it doesn't go away by deleting the labels; it moves
into ASSEMBLE.

**Required before the IDENTIFY-v2 LoRA:** the **oracle assembly audit** (see
"smallest experiment" below). It computes, per dish, GT total kcal vs. kcal
assembled from *perfect visible-truth identification* + true mass + the real
resolver. The gap is the structural bias of the visible-truth decision. Ruling:

- If median assembled-kcal bias on test325 is small (≲5%), the exclusion
  stands as-is.
- If material, the exclusion still stands **but** must be paired with a
  compensating resolution policy: alias fry/dressing-prone visible classes to
  *prepared* FDC/FNDDS variants ("chicken, fried" rather than "chicken, raw"),
  so the oil's calories re-enter through density rather than through a
  hallucinated item. That policy is derived-table curation, allowed.
- Reverting to including hidden items is off the table — the OOD hallucination
  evidence stands.

**Falsifiable check:** oracle-assembly median signed error on test325,
reported before and (if needed) after the prepared-variant aliasing pass.

### BF2 — FRB vocabulary must resolve before it trains (the B1 lesson at larger scale)

277/795 training-vocabulary names resolving at rungs 1–2 is the first review's
unpopulated-alias-table finding recurring one dataset later; the report
correctly self-flags it. Ruling: **training-vocabulary resolution is a
pre-LoRA gate**, not a post-hoc repair. Training a model to emit names the
resolver cannot ground manufactures category-median assemblies by
construction. The fix is label canonicalization *before* training: merge or
map FRB names onto resolvable canonical names (FNDDS composites for
regional/branded dishes), drop the genuinely ungroundable remainder from
truth. Gate: **≥90% of item occurrences in the training corpus resolve at
rungs 1–2** before the FRB arm trains. Post-hoc alias expansion from observed
candidate emissions stays forbidden (eval-leakage boundary), which is exactly
why this must happen pre-training.

### BF3 — NutritionVerse exact IDR/IDP are currently meaningless; fix the eval vocabulary before any arm comparison

Tuned v1 scoring 0.0% exact IDR/IDP on NutritionVerse while beating the
untuned model on HMR means the frozen taxonomy simply lacks the NV label
vocabulary — the metric is measuring a mapping gap, not the model. The report
knows this; make it a gate: extend `eval_taxonomy` (as a versioned v3, with
diff review) from **NutritionVerse's own ground-truth label set** — a small,
closed vocabulary, mappable without ever looking at model outputs. Until
then, no OOD IDR/IDP number may appear in a selection decision; HMR-only OOD
comparison is acceptable in the interim, as the report already practices.

### BF4 — State the SCALE checkpoint-selection set explicitly

"Best checkpoint was epoch 5" — selected on what data? The report asserts
official Val never drives selection; the training log should *show* it
(selection on the calibration/holdout split, Val untouched). One sentence and
one logged artifact. Cheap insurance against the only leakage path this
otherwise clean split design leaves open.

---

## Rulings on the four requested decisions

**1. Calibration — accepted.** Per-source, equal-group-weight, finite-rank
split conformal is Mondrian conformal prediction at the scene level; with
scene-level exchangeability it is statistically sound, and equal group
weighting plus group-count ranks are the right corrections for multi-view
dishes. Three riders: (a) report the binomial/beta CI on coverage — 32
calibration scenes puts roughly ±7 points of uncertainty on the 80% target,
so 81.3% observed is "consistent with," not "equal to," nominal; (b) the
max-margin unknown-source fallback and phone-margin runtime default are both
correct — production inputs are phone photos, and the 79 g margin is honest
about that; (c) do not build an image-conditional calibration *model* now,
but note the nearly-free upgrade: width-normalized conformity (divide
residuals by the model's own predicted P90–P10 width) reuses the existing
quantile heads and typically tightens exactly the wide-margin domain. Try it
as a post-E3 experiment, not a blocker.

**2. Does the NutritionVerse gain generalize?** Treat it as *adaptation to
one phone-capture domain*, demonstrated cleanly — not as phone-photo
robustness in general. 126 training scenes from one collection protocol
cannot establish the latter, and nothing in the report claims otherwise. The
cheap decisive test arrives with FPB: **evaluate Probe B on FPB zero-shot
before FPB ever enters training.** If Probe B transfers (MAPE near ~27%,
coverage under the max-margin fallback reasonable), the mixing recipe
generalizes; if it collapses like v1 did on NV, we've learned each phone
domain must be represented, which changes the roadmap (and strengthens the
case for FPB in the mix). Either answer is valuable; the experiment is free.

**3. FRB in IDENTIFY — matched ablation, not the primary arm.** Equal units
on multi-item images inject a uniformity prior into portion supervision on
exactly the OOD-looking photos where portions matter; that can silently
degrade assembled kcal through dense-item share errors even while
identification improves. Run the FoodSeg-mixed arm as primary and FRB as a
+FRB ablation under the frozen E3 decision procedure (Tier-1 → paired
assembled kcal on the both-eligible intersection → diagnostics), watching
share MAE on Nutrition5K and share-uniformity drift specifically. If the full
FRB arm degrades shares, the fallback variant is FRB restricted to
single-item images (portion supervision becomes trivially true). Do not build
custom portion-token loss masking unless mlx-vlm offers it trivially — that's
machinery the ablation may prove unnecessary. BF2 gates either variant.

**4. FPB dose — smaller first, protect the demonstrated gain.** Change one
thing at a time: after the zero-shot eval, first FPB probe at a modest dose
(≈50/25/25 or equivalently "keep NutritionVerse's effective exposure
unchanged, carve FPB's share out of Nutrition5K's"), with Probe B unchanged
as control and the stated rejection rule (harm to either N5K or NV blocks).
Escalate FPB only if the modest dose is clean. A fixed 50/20/30 that both
dilutes NV and jumps FPB to the largest minority share confounds two changes
in one probe.

---

## The smallest experiment before the IDENTIFY LoRA (review question 4)

**The oracle assembly audit.** No training, no VLM inference — data + resolver
+ arithmetic:

1. Take visible-truth labels (post-exclusion) as *perfect* IDENTIFY output.
2. Assemble three ways: (a) true total mass, (b) SCALE-v2 P50, (c) SCALE-v2
   interval endpoints.
3. Resolve through the real ladder; run the full scoring harness on test325
   and NutritionVerse official Val.

It answers, in one cheap pass: the architecture's **error floor** (if oracle
assembly can't beat v8's conditional MAE, no LoRA can save the candidate);
the **hidden-ingredient kcal bias** (BF1's required number, isolated in
variant (a)); the **true resolution rate** on the vocabulary the model will
actually be trained to emit (B1/BF2 verification); and the **share-coarseness
cost** (oracle shares vs. 5-point-bucketed oracle shares). Every subsequent
training decision inherits calibrated expectations from this table. Run it
before any further LoRA spend; it likely reorders the remaining work.

---

## Optional improvements (post-E3)

- Width-normalized conformal calibration (rider 1c).
- Fast-baseline slices (64/63 records) are fine for iteration; final claims
  use full sets only — state this rule once in the harness docs.
- Report raw (pre-normalization) structural failure rates for v1-legacy and
  v2 parsers side by side, so the closure-in-code benefit stays quantified.
- Licensing: the non-commercial confirmation resolves the training-track
  question; keep the permissive-only SCALE manifest reproducible as the
  fallback, and add a §8 ship-gate line item: any release bundling NC-trained
  weights requires the specific license review the report already promises.
  This must not be discovered at release time.

## Verdict on the remaining-work list

The five listed steps are approved with insertions: the oracle assembly audit
runs **first** (before FPB work, since BF1 may change resolution policy);
FPB step 2 is preceded by the zero-shot Probe B evaluation; BF2
canonicalization joins step 3; BF3 taxonomy-v3 lands before any arm scoring;
BF4 is a one-line logging addition. The deliberate restraint — no new LoRA
until the spine is measured — continues to be the right instinct. v8 stays
shipping; nothing here touches production.
