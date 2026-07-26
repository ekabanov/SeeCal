import Foundation
import SeeCalDomain

public struct GoalEditDraft: Equatable, Sendable {
    public var caloriesText: String
    public var proteinText: String
    public var fatText: String
    public var carbsText: String

    public init(target: DailyNutritionTarget) {
        caloriesText = String(Int(target.calories.rounded()))
        proteinText = String(Int(target.proteinGrams.rounded()))
        fatText = String(Int(target.fatGrams.rounded()))
        carbsText = String(Int(target.carbsGrams.rounded()))
    }

    public func toDailyTarget() throws -> DailyNutritionTarget {
        DailyNutritionTarget(
            calories: try Self.parsePositive(caloriesText, field: "calories"),
            proteinGrams: try Self.parsePositive(proteinText, field: "protein"),
            fatGrams: try Self.parsePositive(fatText, field: "fat"),
            carbsGrams: try Self.parsePositive(carbsText, field: "carbs")
        )
    }

    private static func parsePositive(_ text: String, field: String) throws -> Double {
        guard let value = Double(text), value.isFinite, value > 0 else {
            throw NSError(domain: "GoalEditDraft", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid \(field) target"])
        }
        return value
    }
}
