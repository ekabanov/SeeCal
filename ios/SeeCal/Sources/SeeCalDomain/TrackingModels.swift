import Foundation

/// The model's per-item measurement at the moment it was captured: the grams the
/// model estimated for this ingredient and the nutrition values that go with that
/// specific mass. `MealItem.grams` scales linearly off this fixed reference (spec §2).
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

/// One ingredient within a logged (or being-logged) meal. `grams` is the current,
/// user-adjustable portion size; `base` is the model's original measurement.
/// Current nutrition is always `base.value * grams / base.grams` (spec §2) — never
/// stored independently, so the two can never drift out of sync.
public struct MealItem: Codable, Equatable, Sendable, Identifiable {
    /// Gram stepper granularity (spec §2/§5): "Grams stepping: ±5 g, minimum 5 g."
    public static let gramStep: Double = 5
    public static let minimumGrams: Double = 5

    public var id: UUID
    public var name: String
    public var grams: Double
    public var base: MealItemBase

    public init(id: UUID = UUID(), name: String, grams: Double, base: MealItemBase) {
        self.id = id
        self.name = name
        self.grams = grams
        self.base = base
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

    /// Clamps a candidate gram value to the spec's 5 g floor.
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
