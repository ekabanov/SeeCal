import Foundation
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence

public enum MealEditDraftError: Error, Equatable, CustomStringConvertible {
    case noItems
    case emptyName
    case missingContext

    public var description: String {
        switch self {
        case .noItems:
            return "A meal must have at least one item"
        case .emptyName:
            return "A meal must have a name"
        case .missingContext:
            return "Draft is missing the context needed to build a new entry"
        }
    }
}

/// One draft type serves photo scans, manual/barcode entry, and editing an
/// edit-an-existing-entry flow (tapping a meal row, spec §4/§5): initialize from
/// either a fresh source or a previously-logged `MealLogEntry`, mutate
/// names and the ingredient list/nutrition, then commit — the caller `save()`s (new scan) or
/// `update()`s (edit) the resulting `MealLogEntry` via `MealLogStore`.
public struct MealEditDraft: Equatable, Sendable {
    /// Context captured only in new-scan mode — everything needed to build a brand
    /// new `MealLogEntry` on commit that `existingEntry` doesn't already carry.
    public struct NewMealContext: Equatable, Sendable {
        public var imagePath: String?
        public var mealType: MealType
        public var origin: MealOrigin
        public var barcodeSource: BarcodeSourceMetadata?
        public var volumeMl: Double?
        public var maxHeightMm: Double?

        public init(
            imagePath: String? = nil,
            mealType: MealType,
            origin: MealOrigin,
            barcodeSource: BarcodeSourceMetadata? = nil,
            volumeMl: Double? = nil,
            maxHeightMm: Double? = nil
        ) {
            self.imagePath = imagePath
            self.mealType = mealType
            self.origin = origin
            self.barcodeSource = barcodeSource
            self.volumeMl = volumeMl
            self.maxHeightMm = maxHeightMm
        }
    }

    public var name: String
    public var items: [MealItem]

    /// Set in edit mode (initialized from an existing entry); nil for a new scan.
    private var existingEntry: MealLogEntry?
    /// Set in new-scan mode; nil when editing an existing entry.
    private var newMealContext: NewMealContext?

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
        self.newMealContext = NewMealContext(
            imagePath: imagePath,
            mealType: mealType,
            origin: .photo,
            volumeMl: volumeMl,
            maxHeightMm: maxHeightMm
        )
    }

    /// Manual-new-meal mode. It intentionally starts empty; the sheet opens its
    /// blank ingredient editor immediately and keeps Log disabled until a valid
    /// item is added.
    public init(manualMealType mealType: MealType, name: String = "Manual meal") {
        self.items = []
        self.name = name
        self.existingEntry = nil
        self.newMealContext = NewMealContext(mealType: mealType, origin: .manual)
    }

    /// Barcode-new-meal mode. The item already contains the consumed amount and
    /// its immutable package-label reference.
    public init(
        barcodeItem item: MealItem,
        name: String,
        mealType: MealType,
        source: BarcodeSourceMetadata
    ) {
        self.items = [item]
        self.name = name
        self.existingEntry = nil
        self.newMealContext = NewMealContext(
            mealType: mealType,
            origin: .barcode,
            barcodeSource: source
        )
    }

    /// Edit mode: build a draft from an already-logged entry, ready to be edited
    /// and saved back over the same entry.
    public init(entry: MealLogEntry) {
        self.items = entry.items
        self.name = entry.name
        self.existingEntry = entry
        self.newMealContext = nil
    }

    /// True when this draft is editing a previously-logged entry (vs. a fresh scan).
    public var isEditingExisting: Bool {
        existingEntry != nil
    }

    public var existingEntryID: UUID? {
        existingEntry?.id
    }

    public var isValid: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !items.isEmpty
    }

    /// The meal photo backing this draft, whichever mode it's in — the freshly
    /// captured file for a new scan, the stored entry's photo in edit mode. Drives
    /// the result sheet's header thumbnail (spec §5).
    public var imagePath: String? {
        existingEntry?.imagePath ?? newMealContext?.imagePath
    }

    public var origin: MealOrigin {
        existingEntry?.origin ?? newMealContext?.origin ?? .manual
    }

    public var barcodeSource: BarcodeSourceMetadata? {
        existingEntry?.barcodeSource ?? newMealContext?.barcodeSource
    }

    /// Depth metadata (spec §5: "depth-assisted" chip + "~V ml · max height H mm"
    /// meta line, rendered ONLY when present). `nil` until D5 lands, in both modes.
    public var volumeMl: Double? {
        existingEntry != nil ? existingEntry?.volumeMl : newMealContext?.volumeMl
    }

    public var maxHeightMm: Double? {
        existingEntry != nil ? existingEntry?.maxHeightMm : newMealContext?.maxHeightMm
    }

    /// Derived totals: the sum of every item's current (grams-scaled) nutrition —
    /// never stored independently (spec §2).
    public var totals: NutritionTotals {
        items.reduce(NutritionTotals()) { $0 + $1.totals }
    }

    /// Compatibility mutation for non-editor clients. The focused editor accepts
    /// positive whole grams directly; either path rescales nutrition proportionally.
    public mutating func stepGrams(itemID: MealItem.ID, by delta: Double) {
        guard let index = items.firstIndex(where: { $0.id == itemID }) else { return }
        items[index].stepGrams(by: delta)
    }

    /// Sets one item's grams directly, floored at one positive gram.
    public mutating func setGrams(itemID: MealItem.ID, to newValue: Double) {
        guard let index = items.firstIndex(where: { $0.id == itemID }) else { return }
        items[index].setGrams(newValue)
    }

    public mutating func replaceItem(_ item: MealItem) {
        guard let index = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[index] = item
    }

    /// Replaces the food identity and nutrition density while keeping the
    /// model's current portion estimate and immutable reset reference.
    public mutating func replaceFood(
        itemID: MealItem.ID,
        with candidate: NutritionProfileCandidate
    ) {
        guard let index = items.firstIndex(where: { $0.id == itemID }) else { return }
        let current = items[index]
        let profile = candidate.profile
        items[index] = MealItem(
            id: current.id,
            name: candidate.displayName,
            grams: current.grams,
            amountUnit: current.amountUnit,
            base: MealItemBase(
                grams: 100,
                kcal: profile.kcalPer100g,
                protein: profile.proteinPer100g,
                fat: profile.fatPer100g,
                carbs: profile.carbsPer100g
            ),
            sourceReference: current.sourceReference
        )
    }

    public mutating func addItem(_ item: MealItem) {
        items.append(item)
    }

    @discardableResult
    public mutating func removeItem(id: MealItem.ID) -> (item: MealItem, index: Int)? {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return nil }
        return (items.remove(at: index), index)
    }

    public mutating func restoreItem(_ item: MealItem, at index: Int) {
        guard !items.contains(where: { $0.id == item.id }) else { return }
        items.insert(item, at: min(max(0, index), items.count))
    }

    /// Commits the draft to a `MealLogEntry`. In edit mode this is the original
    /// entry with `name`/`items` replaced (pass to `MealLogStore.update`); in
    /// new-scan mode it's a brand new entry (pass to `MealLogStore.save`) — check
    /// `isEditingExisting` to know which.
    public func committedEntry() throws -> MealLogEntry {
        guard !items.isEmpty else {
            throw MealEditDraftError.noItems
        }
        let committedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !committedName.isEmpty else {
            throw MealEditDraftError.emptyName
        }

        if let existingEntry {
            var updated = existingEntry
            updated.name = committedName
            updated.items = items
            return updated
        }

        guard let context = newMealContext else {
            throw MealEditDraftError.missingContext
        }
        return MealLogEntry(
            name: committedName,
            mealType: context.mealType,
            imagePath: context.imagePath,
            origin: context.origin,
            barcodeSource: context.barcodeSource,
            items: items,
            volumeMl: context.volumeMl,
            maxHeightMm: context.maxHeightMm
        )
    }
}
