import Foundation

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
