import Foundation

/// Nutrition values measured at a reference mass. Ratios are the source of truth:
/// `value(at: grams) = storedValue * grams / self.grams`.
public struct MealItemBase: Codable, Equatable, Sendable {
    public var grams: Double
    public var kcal: Double
    public var protein: Double
    public var fat: Double
    public var carbs: Double

    public init(grams: Double, kcal: Double, protein: Double, fat: Double, carbs: Double) {
        self.grams = grams
        self.kcal = kcal
        self.protein = protein
        self.fat = fat
        self.carbs = carbs
    }
}

/// The immutable model-produced value used by reset actions. Manual ingredients
/// have no estimate.
public struct MealItemEstimate: Codable, Equatable, Sendable {
    public let name: String
    public let base: MealItemBase

    public init(name: String, base: MealItemBase) {
        self.name = name
        self.base = base
    }
}

public enum MealNutritionField: String, Codable, CaseIterable, Sendable {
    case kcal
    case protein
    case fat
    case carbs
}

/// One ingredient within a logged (or being-logged) meal. `base` is the CURRENT
/// nutrition basis, not necessarily the model value: correcting a field rewrites
/// that field's density. `modelEstimate` remains immutable for reset.
public struct MealItem: Codable, Equatable, Sendable, Identifiable {
    /// Kept for non-editor clients; the focused editor accepts any positive whole gram.
    public static let gramStep: Double = 5
    public static let minimumGrams: Double = 1

    public var id: UUID
    public var name: String
    public var grams: Double
    public var base: MealItemBase
    public let modelEstimate: MealItemEstimate?

    /// Model-backed initializer retained for source compatibility. The supplied
    /// name/base become both the current basis and immutable reset estimate.
    public init(id: UUID = UUID(), name: String, grams: Double, base: MealItemBase) {
        self.id = id
        self.name = name
        self.grams = grams
        self.base = base
        self.modelEstimate = MealItemEstimate(name: name, base: base)
    }

    /// Explicit initializer used when preserving an existing estimate or creating
    /// a manual item (`modelEstimate: nil`).
    public init(
        id: UUID = UUID(),
        name: String,
        grams: Double,
        base: MealItemBase,
        modelEstimate: MealItemEstimate?
    ) {
        self.id = id
        self.name = name
        self.grams = grams
        self.base = base
        self.modelEstimate = modelEstimate
    }

    /// Linear rescale factor relative to the base measurement (spec §2).
    public var scaleFactor: Double {
        guard base.grams != 0 else { return 0 }
        return grams / base.grams
    }

    public var kcal: Double { base.kcal * scaleFactor }
    public var protein: Double { base.protein * scaleFactor }
    public var fat: Double { base.fat * scaleFactor }
    public var carbs: Double { base.carbs * scaleFactor }

    public var totals: NutritionTotals {
        NutritionTotals(calories: kcal, proteinGrams: protein, fatGrams: fat, carbsGrams: carbs)
    }

    public var isManual: Bool {
        modelEstimate == nil
    }

    public var isEdited: Bool {
        guard let estimate = modelEstimate else { return true }
        return name != estimate.name
            || !Self.nearlyEqual(grams, estimate.base.grams)
            || !Self.sameDensity(base, estimate.base, field: .kcal)
            || !Self.sameDensity(base, estimate.base, field: .protein)
            || !Self.sameDensity(base, estimate.base, field: .fat)
            || !Self.sameDensity(base, estimate.base, field: .carbs)
    }

    public static func clampedGrams(_ value: Double) -> Double {
        max(minimumGrams, value)
    }

    public mutating func setGrams(_ newValue: Double) {
        grams = Self.clampedGrams(newValue)
    }

    public mutating func stepGrams(by delta: Double) {
        setGrams(grams + delta)
    }

    public mutating func incrementGrams() {
        stepGrams(by: Self.gramStep)
    }

    public mutating func decrementGrams() {
        stepGrams(by: -Self.gramStep)
    }

    /// Replaces one CURRENT value at the item's current grams. The other fields'
    /// densities are preserved exactly.
    public mutating func setCurrentValue(_ value: Double, for field: MealNutritionField) {
        let current = totals
        let nonnegativeValue = max(0, value)
        base = MealItemBase(
            grams: grams,
            kcal: field == .kcal ? nonnegativeValue : current.calories,
            protein: field == .protein ? nonnegativeValue : current.proteinGrams,
            fat: field == .fat ? nonnegativeValue : current.fatGrams,
            carbs: field == .carbs ? nonnegativeValue : current.carbsGrams
        )
    }

    public mutating func resetNameToEstimate() {
        guard let estimate = modelEstimate else { return }
        name = estimate.name
    }

    /// Resetting grams preserves all current densities, including corrections.
    public mutating func resetGramsToEstimate() {
        guard let estimate = modelEstimate else { return }
        grams = Self.clampedGrams(estimate.base.grams)
    }

    /// Restores one model density evaluated at the current grams.
    public mutating func resetNutritionToEstimate(_ field: MealNutritionField) {
        guard let estimate = modelEstimate else { return }
        let modelValue = Self.value(in: estimate.base, field: field)
        let modelDensity = estimate.base.grams == 0 ? 0 : modelValue / estimate.base.grams
        setCurrentValue(modelDensity * grams, for: field)
    }

    public mutating func resetToEstimate() {
        guard let estimate = modelEstimate else { return }
        name = estimate.name
        grams = Self.clampedGrams(estimate.base.grams)
        base = estimate.base
    }

    private static func value(in base: MealItemBase, field: MealNutritionField) -> Double {
        switch field {
        case .kcal: return base.kcal
        case .protein: return base.protein
        case .fat: return base.fat
        case .carbs: return base.carbs
        }
    }

    private static func sameDensity(
        _ lhs: MealItemBase,
        _ rhs: MealItemBase,
        field: MealNutritionField
    ) -> Bool {
        let lhsDensity = lhs.grams == 0 ? 0 : value(in: lhs, field: field) / lhs.grams
        let rhsDensity = rhs.grams == 0 ? 0 : value(in: rhs, field: field) / rhs.grams
        return nearlyEqual(lhsDensity, rhsDensity)
    }

    private static func nearlyEqual(_ lhs: Double, _ rhs: Double) -> Bool {
        abs(lhs - rhs) < 0.000_001
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, grams, base, modelEstimate
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        name = try container.decode(String.self, forKey: .name)
        grams = try container.decode(Double.self, forKey: .grams)
        base = try container.decode(MealItemBase.self, forKey: .base)

        if container.contains(.modelEstimate) {
            modelEstimate = try container.decodeIfPresent(MealItemEstimate.self, forKey: .modelEstimate)
        } else {
            // Pre-editable schema: `base` was the immutable model estimate.
            modelEstimate = MealItemEstimate(name: name, base: base)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(grams, forKey: .grams)
        try container.encode(base, forKey: .base)
        // Encode explicit null for manual items so decoding can distinguish them
        // from old JSON where the key did not exist.
        try container.encode(modelEstimate, forKey: .modelEstimate)
    }
}

public extension MealItem {
    /// Maps a model-output item (`{name, estimated_grams, calories, protein_g, fat_g,
    /// carbs_g}`, spec §0) into a `MealItem` whose base measurement *is* the model's
    /// estimate — current grams start out equal to the base, so initial scaling is 1:1.
    init(scanItem: ScanItem) {
        self.init(
            name: scanItem.name,
            grams: scanItem.estimatedGrams,
            base: MealItemBase(
                grams: scanItem.estimatedGrams,
                kcal: scanItem.calories,
                protein: scanItem.proteinGrams,
                fat: scanItem.fatGrams,
                carbs: scanItem.carbsGrams
            )
        )
    }
}

public struct DailyNutritionTarget: Codable, Equatable, Sendable {
    public var calories: Double
    public var proteinGrams: Double
    public var fatGrams: Double
    public var carbsGrams: Double

    public init(calories: Double, proteinGrams: Double, fatGrams: Double, carbsGrams: Double) {
        self.calories = calories
        self.proteinGrams = proteinGrams
        self.fatGrams = fatGrams
        self.carbsGrams = carbsGrams
    }

    public static let defaultTarget = DailyNutritionTarget(
        calories: 2200,
        proteinGrams: 150,
        fatGrams: 70,
        carbsGrams: 220
    )
}

public struct NutritionTotals: Codable, Equatable, Sendable {
    public var calories: Double
    public var proteinGrams: Double
    public var fatGrams: Double
    public var carbsGrams: Double

    public init(calories: Double = 0, proteinGrams: Double = 0, fatGrams: Double = 0, carbsGrams: Double = 0) {
        self.calories = calories
        self.proteinGrams = proteinGrams
        self.fatGrams = fatGrams
        self.carbsGrams = carbsGrams
    }

    public static func + (lhs: NutritionTotals, rhs: NutritionTotals) -> NutritionTotals {
        NutritionTotals(
            calories: lhs.calories + rhs.calories,
            proteinGrams: lhs.proteinGrams + rhs.proteinGrams,
            fatGrams: lhs.fatGrams + rhs.fatGrams,
            carbsGrams: lhs.carbsGrams + rhs.carbsGrams
        )
    }
}

public struct NutritionRemaining: Codable, Equatable, Sendable {
    public var calories: Double
    public var proteinGrams: Double
    public var fatGrams: Double
    public var carbsGrams: Double

    public init(calories: Double, proteinGrams: Double, fatGrams: Double, carbsGrams: Double) {
        self.calories = calories
        self.proteinGrams = proteinGrams
        self.fatGrams = fatGrams
        self.carbsGrams = carbsGrams
    }
}

public enum NutritionTracker {
    public static func totals(from results: [FoodScanResult]) -> NutritionTotals {
        results.reduce(NutritionTotals()) { partial, result in
            partial + NutritionTotals(
                calories: result.totalCalories,
                proteinGrams: result.proteinGrams,
                fatGrams: result.fatGrams,
                carbsGrams: result.carbsGrams
            )
        }
    }

    public static func remaining(target: DailyNutritionTarget, consumed: NutritionTotals) -> NutritionRemaining {
        NutritionRemaining(
            calories: target.calories - consumed.calories,
            proteinGrams: target.proteinGrams - consumed.proteinGrams,
            fatGrams: target.fatGrams - consumed.fatGrams,
            carbsGrams: target.carbsGrams - consumed.carbsGrams
        )
    }
}
