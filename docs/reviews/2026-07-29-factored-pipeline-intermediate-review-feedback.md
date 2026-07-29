# Factored pipeline — teacher feedback on intermediate review

**Date:** 2026-07-29
**Reviewing:** `docs/reviews/2026-07-29-factored-pipeline-intermediate-review.md`
**Binding design:** `docs/specs/2026-07-29-factored-pipeline-design.md`

Overall: execution quality is high. The corrections log shows the right
discipline (interleaving fix, gram-weighted HMR restoration, Atwater-by-
construction category defaults), the anti-leakage stance on alias promotion is
correct, and E0 is the headline result of the project so far — **v8 leaves only
39/325 dishes Tier-1-clean (HMR 12.9%, DVR 40.5%, AIR 31.6%)**. That number is
the empirical justification for the whole redesign; keep it front and center.

Findings below in the four requested buckets, then rulings on A–I.

---

## 1. Blocking correctness problems

### B1 — The alias table is effectively unpopulated; RESOLVE rung 1 is empty in practice

**Violated invariant:** design §4.3 — "the alias table pre-maps the entire
training vocabulary … in-distribution misses are rare by construction."
572 source-vocabulary names sit unpromoted in the review file, which is why E1
lands at 16.5% exact. As implemented, "looked up" degrades to "category-
median-guessed" for ~82% of items — that quietly hollows out the central
H2/H3-elimination claim while the report's architecture summary still asserts
it.

**Required fix (before any factored-candidate assembly evaluation; training
may proceed meanwhile):** promote the 572 source-vocabulary names, prioritized
by gram-weighted frequency in the training corpus. This is the one-time review
of a derived table that the design explicitly sanctions; it is not manual data
gathering. Keep the existing rule that **eval-emitted names are never
promoted** — that stays as the anti-leakage boundary.

**Falsifiable check:** after promotion, ≥95% of training-vocabulary names
resolve at rung 1; then re-run the E1 audit on the *factored candidate's*
emissions (not v8's — see ruling E) and report gram-weighted rung-1/2 rate.

### B2 — HMR's taxonomy must be decoupled from the runtime resolver and frozen

**Violated invariant:** the eval contract is the judge of the architecture; a
component under test must not define its own scoring (design §5 exists
precisely so trade-offs "cannot masquerade"). Today a resolver heuristic
change can reclassify hard vs. soft matches and thereby improve HMR — a
Goodhart loop, and finding F shows you already sense it.

**Required fix (before comparing E3 arms):** snapshot the name→FDC-id/category
mapping into a committed, versioned `eval_taxonomy_v1` artifact. The scoring
harness reads only that artifact; the runtime resolver may evolve freely.
Changes to the eval taxonomy require an explicit versioned diff, never a
side effect of resolver work.

**Falsifiable check:** (a) re-scoring the E0 prediction set under the frozen
taxonomy reproduces the published numbers (or revises them once, with a
reviewed diff); (b) add a regression test that mutates a resolver heuristic
and asserts HMR on a fixed prediction set does not move.

### B3 — Trainable-parameter assertion in `train.sh`; resume disabled

**Violated invariant:** a training run must train the intended LoRA parameter
set. The observed resume produced 366.9M trainable parameters vs. the fresh
run's 32.5M — adapter stacking or base unfreezing, silently.

**Required fix:** `train.sh` logs trainable-parameter count at startup and
hard-aborts if it deviates from the expected LoRA count (±tolerance). Disable
resume for this track (guard + doc note) until mlx-vlm adapter-resume
semantics are characterized — that characterization is off the critical path,
and upstream already has one known defect we track (multi-image, mlx-vlm
#1726), so distrust-by-default is warranted. The assertion converts this
failure class from silent to loud regardless of future resume policy.

---

## 2. Experiment-design changes required before the second E3 arm

1. **Freeze the eval taxonomy first** (B2). Arm selection on a movable metric
   is not a selection.
2. **Fix the decision procedure for E3 now, in writing:** Tier-1 gates first
   (HMR under the frozen taxonomy, refusal recall/over-refusal), then **paired
   assembled conditional kcal MAE on the intersection of dishes that are
   Tier-1-clean and fully resolved in both arms** (in-distribution must be
   within bootstrap noise; OOD must improve), then IDR/IDP/share MAE as
   diagnostics only. Full factored assembly is the Tier-2 decision metric
   because it prices share errors by caloric consequence — this settles
   questions D and I structurally.
3. **Implement the share-closure repair in the evaluator now** (ruling B
   below), so both arms report parse / repaired / rejected rates under
   identical rules. Runtime adoption is a Stage-3 decision; measurement must
   not wait.
4. **State the FoodSeg container-labeling policy in the report.** The frozen
   contract requires `container`, and FoodSeg has no container labels. If the
   prep script defaults them (e.g., all "plate"/"other"), the model learns a
   spurious corpus→container correlation and the field's disagreement-gate
   value dies. Pseudo-label offline (zero-shot pass) or document the default
   and demote container to diagnostic for the mixed arm — but say which.
5. **No equal-step control required before the mixed arm** (ruling C). Log the
   2.85×-steps confound explicitly in the E3 write-up; an equal-step N5K arm
   (E3b) is a cheap post-hoc add only if the mixed arm wins *and* attribution
   would change a future decision (e.g., whether E10 cloud-scale data is
   worth pursuing).

---

## 3. Improvements that can wait until after E3

- **Add a resolution ship gate to §8:** gram-weighted rung-1/2 resolution
  ≥85% on test325 (proposal — recalibrate once the candidate's true rate is
  known post-B1), with the OOD rate reported alongside. Rationale in ruling E.
- **SCALE OOD honesty:** step 3 of the plan must report NV-Real interval
  *coverage*, not just MAE. Coverage collapse doesn't block E3, but the
  design's "honest failure mode is wide intervals" claim then requires a
  conservative conformal offset for OOD before Stage 3 gates.
- **License discipline for NutritionVerse-Real (CC BY-NC-SA):** evaluation-only,
  never bundled, never trained on, never redistributed. Add a note to the
  data-provenance table so it can't be forgotten later.
- **OOD hallucinated-condiment diagnostic** (see ruling A): rate of emitted
  oil/dressing-type items on OOD images, tracked per arm.
- **Per-view Tier-1 reporting for NV-Real** (ruling G) — define now, compute
  in step 3; it's the same predictions aggregated differently.

---

## 4. Disagreements with the binding design

No structural disagreement — two amendments the design should absorb:

1. **§4.3 asserted "misses rare by construction" without an enforcement
   mechanism.** B1 is the enforcement debt coming due. Amend §8 with the
   resolution gate above.
2. **The strict-schema stance needs a controlled exception.** The design's
   hard-fail philosophy is right for structural violations, but rejecting an
   otherwise-valid completion whose shares sum to 95 turns a rounding slip
   into a user-visible failure. Amend to allow the flagged deterministic
   repair defined in ruling B.

Also, ruling A below resolves the "visible food" prompt tension in favor of
the design's source-label rule — the prompt stays frozen; do not reword it.

---

## Rulings on the nine questions

**A — Measured-but-invisible ingredients.** Keep every measured ingredient
that survives the 5% apportionment floor; drop nothing by hand. This is
objective, involves no manual labeling, and the floor already eliminates salt/
vinegar/spice-scale items by construction (verify and report how many N5K
completions retain oil-type items at ≥5%). The critical case is calorie-dense
fats: in factored assembly, dropping measured oil redistributes its mass onto
low-density items and produces a *systematic* kcal underestimate — the exact
class of quiet bias this architecture exists to kill. Inferring oil from a
fried or glossy appearance is legitimate visual inference, not hallucination;
"visible food" is satisfied in the way that matters. Track the OOD
hallucinated-condiment rate as the counterweight diagnostic.

**B — Share closure.** Strict rejection as the *only* behavior is wrong. A
sum of 95 is a rounding slip, not a hard mistake, and runtime rejection is a
user-visible failure with no upside. Rule: outputs that are otherwise
schema-valid with raw sum in [90, 110] get deterministic largest-remainder
renormalization to 100 (reuse the data-prep apportionment code), carry a
`repaired` flag, and are counted. Outside that window: reject. Repair is
never silent. Gates: repair rate ≤5% on test325 (higher means closure is
undertrained — retrain, don't paper over), post-repair rejection ≤1%. The
10-epoch run's 7/32 failures all fell in 85–115, so the window is empirically
in the right place; the 20-epoch result shows training, not repair, is still
the primary mechanism.

**C — Compute fairness.** Same-epochs is the correct primary design because
E3 is a *shipping decision between data policies*, not a causal-attribution
study; "each example seen twice" is the natural policy both arms share.
The 2.85× step confound only matters for attribution — handle per §2.5 above.

**D — Weak area shares.** Neither of the offered framings. Selection runs
through full factored assembly: Tier-1 first, then paired assembled kcal MAE
(intersection, in-dist within noise + OOD improved), share MAE diagnostic.
Assembly weighting is the only weighting of share error that matters —
a 10-point share error on lettuce is noise, on oil it's the meal. N5K share
degradation alone cannot block a mixed arm whose assembled numbers hold.

**E — Resolution rate.** 17.6% does **not** yet invalidate the architecture
claim, because E1 measured the wrong pair: v8's unconstrained vocabulary
against an unpopulated alias table. Both sides change: the factored candidate
emits (mostly) training-vocabulary names, and B1 populates the table those
names hit. But the concern becomes real if the candidate's rung-1/2 rate
stays low after B1 — then "looked up" is a fiction and the explicit-
uncertainty UI is carrying the whole trust story, which is not the design.
Hence the ≥85% gram-weighted rung-1/2 ship gate. Re-run E1 on candidate
emissions as the first post-training evaluation step.

**F — Yes.** Separate reviewed, frozen, versioned taxonomy (B2). The runtime
resolver must not be able to improve its own report card.

**G — OOD grouping.** Two-level aggregation (mean within dish, then across
225 dishes) is correct as the Tier-2 primary — it stops multi-view dishes
from dominating. For Tier-1 *event* metrics (HMR, refusal), pool per-view
with equal dish weights: every view is a photo a real user could take, and
within-dish averaging launders hard mistakes into fractions. Report per-view
P90 as a diagnostic; do not gate on worst-view (a max over ~4 samples is
noise-dominated).

**H — Resume.** Disable it for this track; add the B3 startup assertion.
Correct-invocation archaeology in mlx-vlm is not on the critical path.

**I — Conditional kcal eligibility.** The exclusion is the right
decontamination under two mandatory guards: (1) completeness (fully-resolved,
Tier-1-clean fraction) is a co-primary number printed adjacent to conditional
MAE, never separated; (2) all cross-model comparisons compute conditional MAE
on the intersection of dishes eligible in both models — the pair-by-dish
lesson applied to eligibility. A penalized composite (incomplete dishes
imputed at category-default error) may be reported as a tie-breaker but must
not be the headline; synthetic penalties invite constant-tuning games.

---

## Verdict on the planned next steps

Proceed with the plan as listed, with these insertions: B1 (alias promotion)
and B2 (taxonomy freeze) land before step 2's evaluation; B3 lands before
step 4's training run; the E3 decision procedure of §2.2 is written down
before either arm is scored; step 3 adds SCALE coverage on NV-Real and the
per-view Tier-1 aggregation. Steps 6–7 unchanged. v8 remains shipping;
nothing here touches the production factory.
