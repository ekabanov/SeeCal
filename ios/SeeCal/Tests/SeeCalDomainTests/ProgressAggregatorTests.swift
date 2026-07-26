import XCTest
@testable import SeeCalDomain

final class ProgressAggregatorTests: XCTestCase {
    func testBuildsFixedLengthSeries() {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        let today = calendar.startOfDay(for: now)
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today)!

        let totals = NutritionTotals(calories: 500, proteinGrams: 20, fatGrams: 10, carbsGrams: 55)

        let entries = [
            AnyMealLogEntry(createdAt: today, totals: totals),
            AnyMealLogEntry(createdAt: yesterday, totals: totals)
        ]

        let points = ProgressAggregator.dailyPoints(from: entries, calendar: calendar, days: 7)
        XCTAssertEqual(points.count, 7)
        XCTAssertEqual(points.last?.calories, 500)
    }
}
