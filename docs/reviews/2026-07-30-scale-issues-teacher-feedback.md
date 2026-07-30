# SCALE issues — teacher feedback

**Date:** 2026-07-30
**Reviewing:** `docs/reviews/2026-07-30-scale-issues-teacher-report.md`
**Binding design:** `docs/specs/2026-07-29-factored-pipeline-design.md`
**Prior feedback:** `docs/reviews/2026-07-29-factored-pipeline-second-intermediate-review-feedback.md`

First, name the good news the report undersells: **true mass + bucketed
visible shares assembles to 10.3 kcal MAE vs. v8's 29.5** on the shared-clean
intersection. That is the oracle-audit result, and it validates the entire
architecture — identification, shares, resolution, and arithmetic are already
good enough to beat the monolith by ~3× *if* mass arrives. It also retires the
BF1 worry in-domain: hidden-ingredient bias is inside a 10.3 kcal floor.
SCALE is now the provable single bottleneck, and the zero-shot FPB discipline
(frozen test, run before training on it) is exactly how this was supposed to
be caught. Good work.

## Diagnosis (question 1, and the report's framing question)

Yes — representation failure, and specifically **prior-dominated output**,
with a contributing **mass-support gap**. The evidence is conclusive:

- Predictions barely respond to portion (94→104→116 g against truth
  141→256→376 g): the features carry almost no portion signal in the FPB
  domain, so the head falls back to the training-mass prior (~90–120 g,
  which is where Nutrition5K lives).
- 85% underestimation with median truth 206 g: FPB masses sit at the edge of
  and beyond the training distribution's support — the model has rarely been
  asked to say "400 g" at all.
- A *test-leaking* affine correction still leaves 106 g: no calibration-family
  fix exists, confirming experiment 4's conclusion from the other direction.

So the priority order for intervention is: **data/support first (FPB train),
geometry second (framing cues), objective third (ordinal), architecture last.**
Calibration work is correctly parked.

## Answers to the eight questions

**2. Direct regression vs. volume×density decomposition.** Keep direct
regression as the trained target for now — a true decomposition is not
trainable with the labels we have (no volume ground truth anywhere in the
inventory), and a learned "density" head would reintroduce semantic knowledge
into a model that shouldn't own it. But adopt the deterministic half of the
decomposition at ASSEMBLE, as ruled previously: an FNDDS **name+container-
conditional portion prior** (typical serving masses — a lookup, no training)
fused with SCALE's interval. That injects the semantic factor through a table,
keeps SCALE image-only, and preserves the disagreement gate — indeed it *is*
the gate: geometric estimate vs. semantic prior diverging is the flag. Do not
condition the SCALE network on IDENTIFY output (option 3 in the report's list)
— that couples the two error sources the gate needs independent. The
oracle-audit harness can evaluate prior-only, SCALE-only, and fused mass
*today* with zero training; add the fused variant to the standing audit.

**3. Smallest decisive experiment.** Three parts, in order:

1. **Zero-training diagnostic first — the occupancy analysis** the report
   already proposes. Per domain, correlate: (a) truth vs. food-pixel/plate
   occupancy, (b) Probe B prediction vs. occupancy. If prediction tracks
   occupancy everywhere but truth only tracks it on the rig, the crop/framing
   hypothesis (A) is confirmed before spending any run. Also report the
   training-set mass histogram against FPB's — quantifies the support gap.
2. **Probe C as a two-run mini-factorial, not one confounded run:**
   C1 = +FPB train, preprocessing unchanged; C2 = +FPB train + letterbox.
   The report is right that mixing both changes weakens attribution — but the
   fix is to run both arms, not to choose one. These are minutes-to-hours
   MobileNet runs; attribution is worth two of them. Everything else in the
   Probe C proposal (frozen tests, 50/25/25 effective exposure, group-aware
   sampling, calibrate-after-selection, reject-on-regression) is approved
   as written.
3. Letterbox note: it preserves aspect ratio but still normalizes absolute
   scale. The principled monocular-scale fix is **camera-intrinsics
   awareness** (focal length / FOV as auxiliary features) — check whether FPB
   and NutritionVerse retain EXIF; iOS provides intrinsics at runtime for
   free. Flag as the next geometry step after C2, not part of it.

**4. Ordinal/pairwise auxiliary loss — principled, but sequenced.** The
constraint "more of the same food in the same setting weighs more" is a
physical invariant, not an FPB artifact; a within-family pairwise ranking
head is legitimate and the 22/40 result shows exactly what's missing. But it
is a *third* variable — run it as Probe D only if C1/C2 leave within-family
ordering broken. Auxiliary head is training-only, dropped at export; runtime
stays image-in, quantiles-out.

**5. Capacity.** MobileNetV3-Large is almost certainly not the binding
constraint — in-domain error is fine and the failure signature (prior
regression, no portion sensitivity) is a data/geometry signature, not
underfitting. 224×224 *resolution* is a more plausible limit than parameter
count (plate rims, utensils, and container boundaries are thin scale cues).
Sequence: data (C1) → geometry (C2) → 320 px input as a cheap C3 if the
occupancy diagnostic implicates fine cues → architecture only if all of that
plateaus.

**6. Checkpoint selection.** Minimax over per-source **equal-group MAPE**
(worst source decides), tie-broken by the mean. MAPE is already
scale-normalized, which neutralizes the very different absolute mass
distributions; minimax prevents pooled selection from quietly sacrificing the
smallest source. Add a Pareto guard: reject any checkpoint that regresses any
single source >10% relative from its per-source best seen during the run.

**7. Uncertainty on unsupported domains — both, layered, plus an honest
product path.** (a) Calibrate on the union of phone domains once FPB train
lands, and adopt width-normalized conformity (the quantile heads make it
nearly free) so margins adapt per image. (b) Accept that *no* conformal
method covers a truly unsupported domain — FPB's 20.9% under an
NV-calibrated margin is the proof, and a wider constant margin cannot fix
point-level bias. So the runtime needs a third state: when the name-prior vs.
SCALE disagreement fires or the relative interval width exceeds a threshold,
**degrade to the FNDDS portion prior with an explicit confirm affordance**
("looks like ~1 serving of X — adjust?") instead of presenting a precise
number. "Mass unavailable, ask" is a feature, not a failure; it is the
confirmed-not-guessed principle applied to scale.

**8. The 18 g crossover is an anchor, not a constraint.** Three reasons it
must not become the optimization target: it is Nutrition5K-specific; the
72-dish intersection was conditioned on *v8's* Tier-1 cleanliness (selection
biased toward v8 — recompute on the both-eligible intersection per the
standing pairing rule); and it presumes v8 is the bar everywhere, while v8's
own mass behavior on NV/FPB has still never been measured — that baseline was
requested in the previous feedback and remains outstanding. Gate on
**downstream assembled calorie regret, per domain, via the oracle-audit
harness with SCALE swapped in** — that is the design's own Tier-2 metric and
it automatically prices mass error by its caloric consequence. Run the same
sweep on NV/FPB so the crossover is known per domain rather than extrapolated
from the rig.

## Consolidated prescription

1. Occupancy + mass-histogram diagnostics (no training).
2. v8 implied-mass baseline on NV-Val and FPB-test (scoring run, still owed).
3. FNDDS portion-prior: build it, add prior-only / SCALE-only / fused variants
   to the oracle assembly audit on all three domains (no training).
4. Probe C1 (+FPB, geometry unchanged) and C2 (+FPB + letterbox), minimax-MAPE
   checkpoint selection, calibration after selection, union-of-phone-domains
   conformal with width normalization.
5. Probe D (pairwise ordinal head) only if within-family ordering is still
   broken after C2.
6. Product fallback path (prior + confirm) specced regardless of probe
   outcomes — ambiguity is irreducible on some photos and the UX must own it.
7. IDENTIFY LoRA stays paused until the per-domain calorie-regret table from
   step 3+4 says mass is no longer the binding constraint — the report's own
   stop rule, endorsed.

The one framing correction to carry forward: the question is no longer
"does the factored architecture work" — 10.3 kcal at the floor says it does —
but "how much mass error can the phone domain be brought down to, and does
the honest-fallback UX cover the rest." That is a narrower, better problem
than the one this track started with.
