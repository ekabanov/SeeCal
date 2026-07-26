import XCTest
@testable import SeeCalDomain

final class NutritionTrackerTests: XCTestCase {
    func testTotalsAndRemaining() {
        let one = FoodScanResult(
            totalCalories: 500,
            proteinGrams: 30,
            fatGrams: 18,
            carbsGrams: 48,
            confidence: 0.9,
            items: [
                ScanItem(name: "meal", estimatedGrams: 250, calories: 500, proteinGrams: 30, fatGrams: 18, carbsGrams: 48)
            ],
            uncertaintyFlags: []
        )

        let two = FoodScanResult(
            totalCalories: 650,
            proteinGrams: 42,
            fatGrams: 20,
            carbsGrams: 70,
            confidence: 0.85,
            items: [
                ScanItem(name: "meal2", estimatedGrams: 350, calories: 650, proteinGrams: 42, fatGrams: 20, carbsGrams: 70)
            ],
            uncertaintyFlags: []
        )

        let consumed = NutritionTracker.totals(from: [one, two])
        XCTAssertEqual(consumed.calories, 1150)
        XCTAssertEqual(consumed.proteinGrams, 72)

        let target = DailyNutritionTarget(calories: 2200, proteinGrams: 150, fatGrams: 70, carbsGrams: 220)
        let remaining = NutritionTracker.remaining(target: target, consumed: consumed)

        XCTAssertEqual(remaining.calories, 1050)
        XCTAssertEqual(remaining.proteinGrams, 78)
        XCTAssertEqual(remaining.fatGrams, 32)
        XCTAssertEqual(remaining.carbsGrams, 102)
    }
}
