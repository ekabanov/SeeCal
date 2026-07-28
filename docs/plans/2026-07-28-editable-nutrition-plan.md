# Editable nutrition implementation plan

**Date:** 2026-07-28  
**Status:** Complete; implemented and verified 2026-07-28  
**Binding inputs:** `docs/specs/2026-07-26-app-spec.md` §2/§5 and
`docs/design/prototype/seecal-prototype.html`

## Outcome

Both a fresh scan and a previously logged meal use the same result sheet to edit
the meal name and every ingredient's name, grams, calories and macros; add and
remove ingredients; reset model-backed values; and keep meal totals equal to the
ingredient sum. A logged meal can also be deleted, with its photo, from that
sheet after confirmation.

## Settled product rules

- Nutrition corrections redefine the corrected field's per-gram density.
- Changing grams scales calories and all macros from their current densities.
- Calories and macros are independent; no 4/9/4 validation or derivation.
- The model's original item remains immutable and exists only for reset.
- Manual items start blank and have no reset-to-estimate controls.
- Ingredient deletion is immediate with a five-second Undo.
- Meal deletion is confirmed, not undoable, and available from the edit sheet.
- A non-empty trimmed meal name and at least one ingredient are required to
  log/save. Each ingredient requires a non-empty name and positive whole grams.
- Whole grams/calories and one-decimal macros are the UI precision. Stored
  densities retain `Double` precision.

## Proposed model shape

Keep `MealItemBase` as the value-at-a-reference-mass primitive so old JSON
continues to have an obvious migration path. Evolve `MealItem` to hold two
distinct bases:

```swift
MealItem {
    id: UUID
    name: String
    grams: Double
    nutritionBasis: MealItemBase       // mutable current density/basis
    modelEstimate: MealItemEstimate?   // immutable reset source; nil if manual
}

MealItemEstimate {
    name: String
    basis: MealItemBase
}
```

The exact Swift names may change during implementation, but the separation may
not: current corrected density and original estimate must not alias each other.
`nutritionBasis` can use any positive reference grams (100 g is convenient);
all calculations use ratios, so no persisted total fields or override flags are
needed.

### Decode compatibility

Implement custom `Codable` for `MealItem`:

1. New-schema JSON decodes `nutritionBasis` and optional `modelEstimate`.
2. Old-schema `{id,name,grams,base}` decodes with `nutritionBasis = base` and a
   model estimate of the old name/base.
3. The older whole-entry `scanResult` migration in `MealLogEntry` continues to
   synthesize one model-backed item.
4. Encoding writes only the new schema. Existing corruption backup behavior is
   unchanged.

This is an in-place lazy migration: no separate database migration job and no
rewrite on launch.

## Implementation sequence

### 1. Domain semantics and migration

**Files**

- `ios/SeeCal/Sources/SeeCalDomain/TrackingModels.swift`
- `ios/SeeCal/Sources/SeeCalPersistence/PersistenceModels.swift`
- `ios/SeeCal/Tests/SeeCalPersistenceTests/MealLogEntryMigrationTests.swift`
- new focused `MealItem` coding/scaling tests under `SeeCalDomainTests` or
  `SeeCalPersistenceTests`

**Work**

- Add the corrected nutrition basis and optional immutable model estimate.
- Map `ScanItem` into an unedited model-backed item.
- Add independent field setters that convert the entered current value to a
  density at current grams.
- Add reset-name, reset-grams, reset-nutrition-field and reset-whole-item
  operations with the exact §2 semantics.
- Expose `isManual` and `isEdited` as derived state. Avoid persisting status
  flags that can drift from the data.
- Preserve stable IDs and keep totals purely derived.
- Decode both existing persisted schemas as described above.

**Tests**

- Correct 200 kcal at 180 g, then change to 360 g → 400 kcal.
- Correct one macro without changing calories or the other macros.
- Reset one nutrition field at changed grams uses model density at current grams.
- Reset grams preserves corrected densities; whole reset restores every original.
- Manual items round-trip with `modelEstimate == nil`.
- Old `base` JSON and legacy whole-entry JSON retain their exact nutrition totals.

### 2. Draft mutation and validation

**Files**

- `ios/SeeCal/Sources/SeeCalApp/MealEditDraft.swift`
- `ios/SeeCal/Tests/SeeCalAppTests/MealEditDraftTests.swift`
- `ios/SeeCal/Tests/SeeCalAppTests/ScanFlowControllerTests.swift`

**Work**

- Replace the stepper-oriented mutation surface with focused editor operations:
  update name/grams/nutrition, add item, replace item, remove item, and restore a
  removed item at its original index.
- Keep item-editor changes in a temporary child draft so Cancel/back cannot
  mutate the meal draft.
- Add draft validity (`trimmed name`, `items not empty`) and ingredient validity
  (`trimmed name`, `whole grams > 0`, nonnegative numeric nutrition).
- Make `committedEntry()` reject an empty meal name as well as no items.
- Ensure edit-mode Cancel continues to discard all sheet changes while an
  interactively dismissed fresh result preserves its updated meal draft.

**Tests**

- Add, edit, delete and restore at original index.
- Deleting the final item is allowed in-memory but commit is rejected.
- Empty/whitespace meal and ingredient names are rejected.
- Blank manual nutrition commits as zero; negative/non-numeric UI input never
  reaches the draft.
- Fresh and existing-entry drafts produce identical item editing results.

### 3. Shared result-sheet UI

**Files**

- `ios/SeeCal/Sources/SeeCalApp/Scan/MealResultSheet.swift`
- optionally a small
  `ios/SeeCal/Sources/SeeCalApp/Scan/IngredientEditorView.swift`
- existing theme/design-system components only where they match the prototype

**Work**

- Rebuild the sheet from the updated prototype: editable title, derived summary,
  compact tappable ingredient rows, EDITED/ADDED chips and Add ingredient.
- Present the focused editor inside the same sheet, not as another modal.
- Use numeric keyboards and locale-safe parsing; normalize visible precision on
  commit without prematurely rounding stored density.
- Show model-backed reset affordances only when an estimate exists.
- Implement immediate ingredient deletion and a five-second in-sheet Undo bar.
  Cancel the timer on Undo, sheet close or a subsequent deletion.
- Disable Done/Log/Save from the validation rules and guard rapid double commits.
- Supply accessibility labels/hints for rows, reset controls and destructive
  actions; verify Dynamic Type, dark mode and Reduce Motion behavior.

**QA states**

- Unedited scan summary; edited item; manual item; empty meal; zero items.
- Model-backed item editor and manual item editor.
- Ingredient deleted with Undo visible.
- Logged-meal edit with Delete meal visible.

### 4. Logged-meal deletion wiring

**Files**

- `ios/SeeCal/Sources/SeeCalApp/RootView.swift`
- `ios/SeeCal/Sources/SeeCalApp/Scan/ScanFlowController.swift`
- `ios/SeeCal/Sources/SeeCalApp/DesignSystem/Components.swift`
- `ios/SeeCal/Sources/SeeCalApp/TodayScreen.swift`
- `ios/SeeCal/Sources/SeeCalApp/HistoryScreen.swift`
- `ios/SeeCal/Tests/SeeCalAppTests/AppViewModelEditFlowTests.swift`
- relevant scan-controller tests

**Work**

- Give `MealResultSheet` an edit-mode delete callback.
- Show the prototype-matching confirmation, then route confirmation through the
  existing `AppViewModel.deleteMeal(id:)` so store deletion and best-effort photo
  cleanup remain one path.
- Close edit state, return/select Today, recompute totals and show "Meal deleted"
  only after deletion succeeds. Preserve the sheet and surface the existing error
  alert on failure.
- Remove the meal-row context-menu delete affordance so deletion lives in the
  edit sheet as specified.

**Tests**

- Cancel confirmation changes nothing.
- Confirm removes the entry and photo, closes edit state and emits the toast.
- A store failure retains the entry/photo and leaves an actionable error.

### 5. Verification and documentation

Run, in order:

```bash
cd ios/SeeCal && swift test -Xcxx -DFMT_CONSTEVAL=
scripts/test.sh
```

The second command is required because macOS Swift tests do not typecheck the
iOS-only camera code. Then exercise the functional parity flow on simulator:

1. fresh scan → edit title/item → add item → delete/Undo → log;
2. reopen logged meal → edit and reset fields → save;
3. reopen → Delete meal → cancel, then confirm;
4. relaunch and verify corrected/manual items round-trip.

Delivered without a deliberate prototype behavior deviation. The implementation
keeps the existing `MealItemBase` name for the mutable current basis to minimize
source and persistence churn; its semantics are now documented explicitly.

## Definition of done

- Prototype, binding spec and Swift behavior agree in both modes.
- Existing meal JSON migrates without loss; new corrected/manual items survive
  relaunch.
- Totals can never diverge from ingredient values.
- Every reset path is deterministic and covered by tests.
- Meal deletion removes its photo and cannot occur accidentally from a row.
- All Swift/ML tests and the iOS build check pass.

## Delivery record

- `scripts/test.sh`: 41 ML tests passed.
- Swift package: 190 tests passed with one environment-gated adapter parity test
  skipped.
- Unsigned generic iOS Simulator build: succeeded.
- The unused "Use LiDAR depth" preference was removed during implementation by
  user request. Depth UI is now capability-driven by `supportsDepthCapture`;
  depth metadata and future capture work remain intact.
