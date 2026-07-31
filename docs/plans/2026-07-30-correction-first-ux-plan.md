# Correction-first meal review plan

**Date:** 2026-07-30  
**Status:** first implementation slice complete; alternative prototype remains non-binding  
**Prototype:** `docs/design/prototype/seecal-correction-first-prototype.html`

## Product objective

SeeCal does not need to reconstruct a laboratory ingredient record before it
can help. It needs to turn a photo into an acceptable meal log faster than
manual entry.

The primary outcome is therefore:

> **Active time and interaction cost from the first visible draft to a
> user-accepted saved meal.**

Calorie, mass, and identification accuracy remain diagnostic inputs and
guardrails. They are not the product-level optimization target.

## Why the current editor is not enough

The shipped editing foundation is strong: the meal and every item are
editable, totals are derived, gram changes scale nutrition, items can be added
or removed with Undo, and nothing is logged without confirmation.

The costly path is correcting an ordinary recognition mistake:

1. tap the ingredient row;
2. enter an editor containing six fields;
3. focus the name field and summon the keyboard;
4. type a replacement;
5. understand whether nutrition still belongs to the original food;
6. optionally edit calories/macros by hand;
7. tap Done;
8. finally save the meal.

That is flexible but form-first. A predictable model error should normally cost
one or two taps and no nutrition knowledge.

## Proposed interaction contract

### 1. Open as a quick review, not an analysis report

- Keep the photo visible and label the result as a draft.
- Put the ingredient list before detailed macros; users verify recognizable
  foods, not a calorie headline.
- Keep one persistent **Save meal** action. Never require every uncertainty to
  be resolved.
- Show at most one high-impact **Quick check** at a time.

### 2. Replace foods instead of renaming text

Tapping an item name opens a ranked replacement tray:

- five likely foods;
- recent/frequent corrections when relevant;
- **Something else…** as the keyboard fallback.

Selecting a replacement:

- preserves the current estimated grams;
- resolves the new food's calories/macros from the nutrition database;
- records the item as corrected;
- updates meal totals immediately;
- takes one tap after opening the tray.

Literal renaming without changing nutrition remains an advanced action for
spelling or display-name changes. It must not be the default meaning of
changing “chicken” to “tofu.”

### 3. Adjust portions inline

- Keep minus, current grams, and plus on the item row.
- A tap on the amount opens coarse **Small / Medium / Large** choices plus
  exact grams.
- Preserve the corrected food density while changing its amount.
- Do not expose calories and all three macros during routine portion editing.

### 4. Treat hidden ingredients as optional shortcuts

For meals where they are plausible, show a compact **Maybe missing?** row:

- oil;
- dressing;
- sauce;
- butter.

These are suggestions, never silent predictions. One tap adds a conservative
database-backed amount; a second tap removes it. This converts the known
hidden-fat failure mode into a cheap user choice without forcing every meal to
pay a blanket calorie correction.

### 5. Make additions retrieval-first

**Add food** first shows:

- likely foods for the scene;
- recently used foods;
- favorites or repeated meal components.

Search/manual nutrition is the fallback. A common missing item should not begin
with an empty six-field form.

### 6. Ask only questions worth the interruption

A clarification earns screen space only when:

- alternatives are genuinely plausible;
- they differ materially in nutrition;
- the correction cannot be made cheaply after save; and
- the system can offer concrete choices.

“Chicken or tofu?” is useful. “We are uncertain” is not. Wide calorie intervals
alone do not justify another prompt.

## Success metrics

### Human acceptance metrics — primary

Measure from draft presentation until Save, excluding time when the app is
backgrounded:

| Metric | Proposed initial gate |
|---|---:|
| Median active review time | ≤10 seconds |
| P90 active review time | ≤25 seconds |
| Save without keyboard | ≥70% |
| Accept unchanged | report, do not optimize alone |
| Accept with at most 2 correction actions | ≥80% |
| Main-food replacement | ≤2 taps, no typing when in top five |
| Add a suggested hidden ingredient | 1 tap |
| Abandonment after a completed analysis | <5% |

These numbers are first usability hypotheses, not frozen release requirements.
A small observed pilot should calibrate them before they become gates.

### Offline correction-cost proxy — secondary

For each held-out draft, compute the cheapest supported edit path to a
human-accepted target:

| Operation | Cost |
|---|---:|
| Accept item | 0 |
| Choose visible replacement | 1 |
| Adjust to adjacent portion preset | 1 |
| Add/remove suggested item | 1 |
| Open tray then search | 2 + typed characters |
| Delete ordinary item | 1 |
| Manually add an unsuggested item | 3 + typed characters |
| Manually edit nutrition | high penalty |

Report median/P90 cost, keyboard rate, and cost by error type. Do not collapse
everything into one score without also showing the distribution.

### Accuracy guardrails

Keep a small set of conventional checks:

- no invalid or impossible output;
- no obviously non-food result treated as a meal;
- no missing dominant meal component without a cheap recovery path;
- no extreme calorie result silently saved;
- no regression in user-visible latency, memory, or battery behavior.

An accuracy improvement wins only when it reduces human correction cost or
protects a guardrail.

## Delivery plan

### Phase 0 — freeze the pivot

- Finish the already-running IDENTIFY training and evaluate it once.
- Do not start Probe E or another LoRA solely to improve an offline headline.
- Preserve v8 as the shipping baseline.

### Phase 1 — instrument the existing experience

- Add a local/test-only review session recorder:
  draft shown, item opened, replacement selected, amount changed, add/delete,
  keyboard search, save, discard, and elapsed active time.
- Store no photos or typed food names in production analytics by default.
- Add a deterministic edit-cost harness for saved evaluation fixtures.

### Phase 2 — implement the shortest correction paths

1. Add a `NutritionProfileCandidate` query over the existing local resolver.
2. Add a `replaceFood` draft operation that preserves grams and replaces
   density/provenance.
3. Build the five-choice replacement tray.
4. Add inline amount controls and portion presets.
5. Add retrieval-first **Add food**.
6. Add optional oil/dressing/sauce/butter shortcuts.
7. Retain the full field editor under **Edit details**.

The existing `MealEditDraft`, derived totals, persistence, Undo, and shared
fresh/edit sheet remain reusable.

### Phase 3 — validate with people, not only labels

- Run 10–15 representative meals through both the current and alternative
  flows with at least five people.
- Counterbalance the order of the two interfaces.
- Capture time, taps, keyboard use, abandonment, final accepted meal, and a
  one-question confidence rating.
- Include clean results, wrong main food, missing add-on, wrong portion, extra
  item, and a genuinely ambiguous dish.

### Phase 4 — let interaction evidence choose ML work

Aggregate the expensive correction paths:

- If correct foods are rarely in the five choices, improve candidate recall.
- If portions dominate effort, improve coarse ordering/presets before exact
  grams.
- If users repeatedly add the same hidden ingredients, improve contextual
  shortcuts or meal templates.
- Train another LoRA only when the observed interaction bottleneck is
  recognition and cannot be removed more cheaply in the interface.

## Proposed first implementation slice

Implement only:

1. replacement tray;
2. database-backed `replaceFood` preserving grams;
3. inline ±15 g controls;
4. local review-time/action instrumentation;
5. full editor retained under **Edit details**.

This slice addresses the most common correction path without changing model
training, persistence format, or the shipping inference runtime. Test it before
adding personalization or automatic clarification.

### Implementation update — 2026-07-30

The first slice is now implemented in the iOS app:

- tapping a food name opens up to five likely replacements;
- every choice comes from the existing local nutrition database;
- replacement preserves the current amount, swaps calorie/macro density, and
  keeps the original model estimate available for reset;
- inline minus/plus controls change the amount by 15 g;
- **Edit details** retains the complete field editor;
- review sessions locally record active time, action count, correction count,
  keyboard use, save/discard/dismiss, and pause while the app is inactive;
- events deliberately exclude photos, food names, and nutrition values;
- device builds bundle the generated nutrition database, while simulator
  development uses the existing local database.

The implementation is intentionally not declared the binding visual spec yet.
The next decision point is a small usability comparison against the current
form-first flow, using the human acceptance metrics above.

## Explicit non-goals

- Do not silently add oil, dressing, or other hidden ingredients.
- Do not require users to resolve every uncertainty.
- Do not expose raw model confidence as a percentage.
- Do not make exact grams the only portion control.
- Do not remove the precise editor; demote it to a fallback.
- Do not replace the binding app specification until the alternative wins a
  measured usability comparison.
