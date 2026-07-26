import XCTest
@testable import SeeCalDomain

final class ProgressAggregatorTests: XCTestCase {
    func testBuildsFixedLengthSeries() {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        let today = calendar.startOfDay(for: now)
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today)!

        let result = FoodScanResult(
            totalCalories: 500,
            proteinGrams: 20,
            fatGrams: 10,
            carbsGrams: 55,
            confidence: 0.9,
            items: [ScanItem(name: "meal", estimatedGrams: 200, calories: 500, proteinGrams: 20, fatGrams: 10, carbsGrams: 55)],
            uncertaintyFlags: []
        )

        let entries = [
            AnyMealLogEntry(createdAt: today, scanResult: result),
            AnyMealLogEntry(createdAt: yesterday, scanResult: result)
        ]

        let points = ProgressAggregator.dailyPoints(from: entries, calendar: calendar, days: 7)
        XCTAssertEqual(points.count, 7)
        XCTAssertEqual(points.last?.calories, 500)
    }
}
