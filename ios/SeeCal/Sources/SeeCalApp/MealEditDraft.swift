import Foundation
import SeeCalDomain
import SeeCalPersistence

public enum MealEditDraftError: Error, Equatable, CustomStringConvertible {
    case noItems
    case missingContext

    public var description: String {
        switch self {
        case .noItems:
            return "A meal must have at least one item"
        case .missingContext:
            return "Draft is missing the context needed to build a new entry"
        }
    }
}

/// One draft type serves both the new-scan flow (result sheet, spec §5) and the
/// edit-an-existing-entry flow (tapping a meal row, spec §4/§5): initialize from
/// either a fresh `FoodScanResult` or a previously-logged `MealLogEntry`, mutate
/// per-item gram steppers, then commit — the caller `save()`s (new scan) or
/// `update()`s (edit) the resulting `MealLogEntry` via `MealLogStore`.
public struct MealEditDraft: Equatable, Sendable {
    /// Context captured only in new-scan mode — everything needed to build a brand
    /// new `MealLogEntry` on commit that `existingEntry` doesn't already carry.
    public struct NewScanContext: Equatable, Sendable {
        public var imagePath: String
        public var mealType: MealType
        public var volumeMl: Double?
        public var maxHeightMm: Double?

        public init(imagePath: String, mealType: MealType, volumeMl: Double? = nil, maxHeightMm: Double? = nil) {
            self.imagePath = imagePath
            self.mealType = mealType
            self.volumeMl = volumeMl
            self.maxHeightMm = maxHeightMm
        }
    }

    public var name: String
    public var items: [MealItem]

    /// Set in edit mode (initialized from an existing entry); nil for a new scan.
    private var existingEntry: MealLogEntry?
    /// Set in new-scan mode; nil when editing an existing entry.
    private var newScanContext: NewScanContext?

    /// New-scan mode: build a draft straight from a fresh inference result (spec §0
    /// output schema — `items[{name, estimated_grams, calories, protein_g, fat_g,
    /// carbs_g}]`). Each item's base measurement *is* the model's own estimate.
    public init(
        scanResult: FoodScanResult,
        imagePath: String,
        mealType: MealType,
        volumeMl: Double? = nil,
        maxHeightMm: Double? = nil,
        name: String? = nil
    ) {
        let mappedItems = scanResult.items.map(MealItem.init(scanItem:))
        self.items = mappedItems
        self.name = name ?? MealLogEntry.defaultName(items: mappedItems, mealType: mealType)
        self.existingEntry = nil
        self.newScanContext = NewScanContext(
            imagePath: imagePath,
            mealType: mealType,
            volumeMl: volumeMl,
            maxHeightMm: maxHeightMm
        )
    }

    /// Edit mode: build a draft from an already-logged entry, ready to have its item
    /// grams adjusted and saved back over the same entry.
    public init(entry: MealLogEntry) {
        self.items = entry.items
        self.name = entry.name
        self.existingEntry = entry
        self.newScanContext = nil
    }

    /// True when this draft is editing a previously-logged entry (vs. a fresh scan).
    public var isEditingExisting: Bool {
        existingEntry != nil
    }

    /// The meal photo backing this draft, whichever mode it's in — the freshly
    /// captured file for a new scan, the stored entry's photo in edit mode. Drives
    /// the result sheet's header thumbnail (spec §5).
    public var imagePath: String? {
        existingEntry?.imagePath ?? newScanContext?.imagePath
    }

    /// Depth metadata (spec §5: "depth-assisted" chip + "~V ml · max height H mm"
    /// meta line, rendered ONLY when present). `nil` until D5 lands, in both modes.
    public var volumeMl: Double? {
        existingEntry != nil ? existingEntry?.volumeMl : newScanContext?.volumeMl
    }

    public var maxHeightMm: Double? {
        existingEntry != nil ? existingEntry?.maxHeightMm : newScanContext?.maxHeightMm
    }

    /// Derived totals: the sum of every item's current (grams-scaled) nutrition —
    /// never stored independently (spec §2).
    public var totals: NutritionTotals {
        items.reduce(NutritionTotals()) { $0 + $1.totals }
    }

    /// Steps one item's grams by `delta` (spec §2/§5: ±5 g, floor of 5 g), rescaling
    /// that item's (and so the draft's derived totals') nutrition proportionally.
    public mutating func stepGrams(itemID: MealItem.ID, by delta: Double) {
        guard let index = items.firstIndex(where: { $0.id == itemID }) else { return }
        items[index].stepGrams(by: delta)
    }

    /// Sets one item's grams directly, still floored at the 5 g minimum.
    public mutating func setGrams(itemID: MealItem.ID, to newValue: Double) {
        guard let index = items.firstIndex(where: { $0.id == itemID }) else { return }
        items[index].setGrams(newValue)
    }

    /// Commits the draft to a `MealLogEntry`. In edit mode this is the original
    /// entry with `name`/`items` replaced (pass to `MealLogStore.update`); in
    /// new-scan mode it's a brand new entry (pass to `MealLogStore.save`) — check
    /// `isEditingExisting` to know which.
    public func committedEntry() throws -> MealLogEntry {
        guard !items.isEmpty else {
            throw MealEditDraftError.noItems
        }

        if let existingEntry {
            var updated = existingEntry
            updated.name = name
            updated.items = items
            return updated
        }

        // newScanContext is expected non-nil here: every init sets exactly one of
        // existingEntry/newScanContext, and the existingEntry branch above returned.
        guard let context = newScanContext else {
            throw MealEditDraftError.missingContext
        }
        return MealLogEntry(
            name: name,
            mealType: context.mealType,
            imagePath: context.imagePath,
            items: items,
            volumeMl: context.volumeMl,
            maxHeightMm: context.maxHeightMm
        )
    }
}
