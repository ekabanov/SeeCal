import XCTest
import SeeCalDomain
import SeeCalPersistence
@testable import SeeCalApp

final class MealEditDraftTests: XCTestCase {
    func testDraftConvertsToUpdatedResult() throws {
        let entry = MealLogEntry(
            mealType: .lunch,
            imagePath: "/tmp/a.jpg",
            scanResult: FoodScanResult(
                totalCalories: 500,
                proteinGrams: 30,
                fatGrams: 10,
                carbsGrams: 60,
                confidence: 0.9,
                items: [ScanItem(name: "meal", estimatedGrams: 200, calories: 500, proteinGrams: 30, fatGrams: 10, carbsGrams: 60)],
                uncertaintyFlags: []
            )
        )

        var draft = MealEditDraft(entry: entry)
        draft.caloriesText = "650"
        draft.proteinText = "42"
        draft.fatText = "20"
        draft.carbsText = "55"

        let result = try draft.toFoodScanResult(basedOn: entry.scanResult)
        XCTAssertEqual(result.totalCalories, 650)
        XCTAssertEqual(result.proteinGrams, 42)
    }
}
