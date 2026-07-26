import Foundation
import SeeCalDomain
import SeeCalPersistence

public struct MealEditDraft: Equatable, Sendable {
    public var caloriesText: String
    public var proteinText: String
    public var fatText: String
    public var carbsText: String

    public init(entry: MealLogEntry) {
        caloriesText = String(Int(entry.scanResult.totalCalories.rounded()))
        proteinText = String(Int(entry.scanResult.proteinGrams.rounded()))
        fatText = String(Int(entry.scanResult.fatGrams.rounded()))
        carbsText = String(Int(entry.scanResult.carbsGrams.rounded()))
    }

    public func toFoodScanResult(basedOn original: FoodScanResult) throws -> FoodScanResult {
        let calories = try Self.parseNonNegative(caloriesText, field: "calories")
        let protein = try Self.parseNonNegative(proteinText, field: "protein")
        let fat = try Self.parseNonNegative(fatText, field: "fat")
        let carbs = try Self.parseNonNegative(carbsText, field: "carbs")

        let itemName = original.items.first?.name ?? "meal"
        let itemGrams = max(original.items.first?.estimatedGrams ?? 100, 1)

        return FoodScanResult(
            totalCalories: calories,
            proteinGrams: protein,
            fatGrams: fat,
            carbsGrams: carbs,
            confidence: original.confidence.map { min($0, 0.99) },
            items: [
                ScanItem(
                    name: itemName,
                    estimatedGrams: itemGrams,
                    calories: calories,
                    proteinGrams: protein,
                    fatGrams: fat,
                    carbsGrams: carbs
                )
            ],
            uncertaintyFlags: original.uncertaintyFlags
        )
    }

    private static func parseNonNegative(_ text: String, field: String) throws -> Double {
        guard let value = Double(text), value.isFinite, value >= 0 else {
            throw NSError(domain: "MealEditDraft", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid \(field) value"])
        }
        return value
    }
}
