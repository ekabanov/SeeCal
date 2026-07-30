# Floor attribution — teacher ruling

**Date:** 2026-07-30
**Reviewing:** `docs/reviews/2026-07-30-regret-decomposition-teacher-ruling-response.md`
**Prior ruling:** `docs/reviews/2026-07-30-regret-decomposition-teacher-ruling.md`

## Approvals

The attribution is exactly what was asked for, the frozen counterfactual
order is properly documented, and the negative results are as valuable as
the positive ones: rung-3+ at 0.25% kills the category-expansion workstream,
and bucketing at 4.11% retires the share-granularity question (E4 can be
closed as answered-by-audit). The self-imposed firewall — frozen test may
identify the failure class but never choose the replacement profile — is
precisely the right discipline, stated before the teacher had to ask.
The IDENTIFY accountability tooling and the ordered work list are approved
as written.

## Prepared-variant pass — approved with one anti-Goodhart constraint

The four exposed aliases are textbook preparation-state errors (dry wheat at
351 kcal/100 g standing in for cooked at 91 is a 3.9× density error on an
exact alias hit). But the selection rule matters: if replacements are chosen
by *density proximity to Nutrition5K train measurements*, the resolver
becomes a lookup table of one cafeteria's kitchen — it will ace the rig and
mislead everywhere else. Constraint: a replacement must be justified by
**preparation-state semantics** (raw→cooked, dry→as-served, dressed→with
dressing); train/val measured density selects *among* semantically valid
candidates and breaks ties, never overrides them. Validation: the new alias
set must be scored on the NutritionVerse quality slice as well as N5K
validation — as-served logic should transfer across kitchens; N5K-quirk
fitting will not. (The frozen v3 taxonomy makes HMR immune to this pass by
construction — the B2 decision paying rent.)

## The exclusion residual: one number is missing before the policy fork

At 20.63 kcal (60.8% of floor), the visible-label residual is now the
largest identified error component in the whole system, and the response's
instinct — don't relabel it as resolver error, hidden-item recitation stays
rejected — is correct. But the fork between "structural uncertainty" and
"validated assembly policy" turns on a number the report doesn't give:
**the signed decomposition**. Report mean signed error and the
bias/variance split of the residual, on train/validation:

- If it is predominantly one-directional (systematic underestimation — the
  original BF1 prediction, since hidden mass is mostly fats and truth kcal
  includes them), then a *validated assembly policy* is available and is not
  a fudge: prefer **as-consumed FDC/FNDDS profiles** whose densities include
  typical preparation fats. Note the caesar-salad row already shows this —
  it is simultaneously a rung-1/2 mismatch and an exclusion case; the
  prepared-variant pass and the exclusion residual partially share one fix.
  Any residual scalar correction beyond that must be derived on train only,
  validated on val, and carried as provenance-visible ("estimated") — never
  silent.
- The variance component, whatever remains, is structural: it belongs in the
  interval. ASSEMBLE should carry a floor-derived uncertainty term (measured
  on train/val, per the report's own framing) so the displayed range honestly
  includes what visible-item assembly cannot know.

Also report the residual's distribution shape (median vs. mean, tail mass).
"Long tail" was asserted; if true, tail-risk features (item count,
sauce/dressing-prone dish classes) should eventually key the confirm UX
rather than paying the residual as uniform interval width on every scan.

## Standing state

1. Signed/bias-variance decomposition of the exclusion residual (train/val
   scoring only) — small, decides the §above fork.
2. Prepared-variant alias pass under the semantics-first constraint, validated
   on N5K val + NV quality slice.
3. IDENTIFY-v2 primary run → frozen v3-taxonomy eval + four model-vs-oracle
   gap slices — unchanged.
4. Probe E when Metal frees — unchanged.
5. Gate operating point before product wiring — unchanged.
6. E4 (share granularity) closed by audit; category-default expansion
   deprioritized by evidence.

v8 ships unchanged; everything above remains shadow. The system is now in
the state the design aimed for: every remaining kcal of error has a named
owner and a measured size.
