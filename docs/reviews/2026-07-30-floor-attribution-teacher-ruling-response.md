# Floor attribution — execution response to teacher ruling

**Date:** 2026-07-30  
**Responding to:** `2026-07-30-floor-attribution-teacher-ruling.md`

## Decision

The exclusion residual is **systematically negative when it exists, but
zero-inflated and variance-dominated overall**. Hidden fats and condiments are
the mechanism, but a uniform calorie correction is the wrong policy because
about half of dishes have no material exclusion residual. The predeclared
train-derived scalar fails validation MAE and is rejected.

The accepted direction is:

1. keep preparation-state-aware, as-consumed profile work separate and
   semantics-first;
2. carry the irreducible remainder as a floor-derived uncertainty term;
3. key that uncertainty by an observable feature such as visible item count
   rather than charging every scan a uniform interval;
4. retain confirmation as the product response to the high-risk tail.

Nothing here changes v8 or the canonical nutrition database.

## Signed exclusion-residual decomposition

The audit redistributes the **true total meal mass** over exact visible
ground-truth item shares, preserves each visible item's measured calorie
density, and subtracts full-meal calories. Its sign is therefore assembly minus
truth: a negative result is an underestimate caused by excluded calories being
denser than the visible items receiving their mass.

Only Nutrition5K train and validation were scored. No model prediction,
resolver profile, or frozen test row participates.

| Metric | Train (2,592) | Validation (325) |
|---|---:|---:|
| Mean signed residual | **−14.93 kcal** | **−12.60 kcal** |
| Median signed residual | 0.00 | 0.00 |
| MAE | 16.26 | 14.02 |
| RMSE | 34.62 | 32.45 |
| Bias² share of MSE | 18.6% | 15.1% |
| Variance share of MSE | **81.4%** | **84.9%** |
| Within ±0.5 kcal rounding tolerance | 50.3% | 52.3% |
| Underestimates among material residuals | **85.6%** | **80.0%** |

This answers the policy fork: the original BF1 directional prediction is
correct for affected dishes, but the population is not well represented by one
constant. The median is zero while the negative mean is pulled by a minority
tail.

Two training rows with non-positive nutrition truth were excluded explicitly;
all 325 validation rows scored. Train and validation IDs have zero overlap.

## Distribution and tail

| Tail diagnostic | Train | Validation |
|---|---:|---:|
| Absolute-error P80 | 28.92 kcal | 19.13 kcal |
| P90 | 53.01 | 45.60 |
| P95 | 71.79 | 69.73 |
| P99 | 130.94 | 143.83 |
| Error >20 kcal | 26.4% | 19.7% |
| Error >50 kcal | 10.6% | 9.2% |
| Error >100 kcal | 1.9% | 2.5% |
| Top 10% share of all absolute error | 53.5% | **61.4%** |
| Top 5% share | 34.5% | **41.6%** |

The long-tail claim is therefore confirmed. The top validation decile carries
over three-fifths of total absolute residual.

The strongest observable slice is visible item count:

| Visible items | Train mean / MAE | Validation mean / MAE |
|---|---:|---:|
| 1 | −0.93 / 0.99 kcal | −0.47 / 0.47 |
| 2 | −3.89 / 3.91 | −3.97 / 3.97 |
| 3–4 | −16.13 / 17.40 | −10.67 / 12.92 |
| 5+ | **−29.94 / 32.87** | **−29.69 / 32.33** |

A visible sauce/dressing-prone flag also transfers directionally: validation
MAE is 25.23 kcal when true versus 11.21 otherwise. For causal diagnosis only,
hidden truth shows the sharper split: the 155 validation dishes with an
excluded condiment/fat have −26.50 kcal mean residual and 29.28 MAE; the other
170 are effectively zero at +0.08 mean and 0.10 MAE. That hidden-truth flag is
not available to production and is not proposed as a feature.

## Scalar correction rejected

The sole uniform rule was frozen before validation:

> add the negative of the training mean residual, +14.93 kcal, and expose it as
> an estimated exclusion correction if it passes all validation guards.

On validation it changes:

| Metric | Uncorrected | +14.93 kcal |
|---|---:|---:|
| Mean signed error | −12.60 | +2.34 |
| MAE | **14.02** | **19.55** |
| RMSE | 32.45 | 29.99 |

Bias and RMSE improve, but MAE worsens by 39.5% and the median moves from zero
to +14.93 kcal. The all-metrics guard fails. No scalar is adopted or wired.

## Floor-derived uncertainty candidate

A train-derived, visible-item-count-keyed P90 absolute-residual term validates
as follows:

| Visible items | Train P90 half-width | Validation coverage |
|---|---:|---:|
| 1 | 0.00 kcal | 94.8% |
| 2 | 17.28 | 92.5% |
| 3–4 | 49.34 | 92.1% |
| 5+ | 74.84 | 88.9% |
| **Overall** | keyed | **92.0%** |

This is a candidate uncertainty component, not a product change. It must later
be combined with SCALE and resolver uncertainty without double-counting, then
validated at the final gate operating point. The 5+ bin's slight undercoverage
also supports a confirmation path rather than pretending the interval alone
solves the tail.

Machine-readable result:
`ml/runs/factored/exclusion-residual-v1/audit.json`.

## Semantics-first prepared aliases

Candidate selection obeyed the anti-Goodhart order: preparation semantics
defined the candidate set; Nutrition5K training density described or broke ties
inside that set; validation did not select candidates.

| Alias | Semantic candidate | N5K validation effect on affected groups | NutritionVerse quality | Decision |
|---|---|---:|---|---|
| wheat berry | cooked KAMUT wheat berry, 140.7 kcal/100 g | **−21.33 kcal MAE**, 17 groups | 0 matching groups | Provisional; no transfer support |
| pork | FNDDS Pork, NFS, 186.6 | **−39.58**, 15 groups | 0 matching groups | Provisional; no transfer support |
| fish | FNDDS Fish, NFS, 234.5 | **+82.90**, 17 groups | 0 matching groups | **Rejected** |
| caesar salad | none | not run | 0 matching groups | Blocked: no dressed whole-salad profile |

The provisional wheat+pork arm improves their 31 affected N5K validation
groups from 75.19 to 44.34 kcal MAE and improves overall exact-share validation
floor from 27.91 to 24.95 kcal. NutritionVerse quality stays exactly 88.03 kcal
because none of its 60 scenes contains either changed label. That is a
successful no-regression score but **not direct cross-kitchen evidence**, so it
does not satisfy the transfer intent of the ruling and does not authorize
promotion.

The fish result is precisely why semantics must precede but not replace
validation: `Fish, NFS` is a defensible generic as-consumed interpretation, yet
it is much too dense for this emitted label's validation population. Density
proximity was not used to swap in a semantically narrower cafeteria-specific
fish profile; the candidate is simply rejected.

Caesar salad exposes a database representation gap, not permission to fit the
number. The available entries are dressing-only or salad-without-dressing.
Neither represents a dressed Caesar salad, so the current bad alias is
documented but no replacement is fabricated.

The canonical SQLite database was never mutated. Each arm ran against an
ephemeral copy. Candidate definitions and machine-readable scores:

- `ml/factored_pipeline/prepared_alias_candidates_v1.tsv`
- `ml/runs/factored/prepared-alias-v1/audit.json`

## Standing state

1. Signed decomposition: complete; uniform correction rejected; uncertainty
   candidate measured.
2. Prepared aliases: semantics-first audit complete; fish rejected, Caesar
   blocked, wheat/pork provisional pending direct OOD support.
3. IDENTIFY-v2 FoodSeg primary: still training normally; post-run frozen
   taxonomy and four model-versus-oracle gap slices remain next.
4. Probe E: waits for Metal after IDENTIFY.
5. Gate operating point: still required before product wiring.
6. E4 remains closed; category-default expansion remains deprioritized.

v8 remains the unchanged shipping and rollback system.
