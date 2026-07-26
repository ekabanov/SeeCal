import XCTest
@testable import SeeCalDomain

final class WeightTrendTests: XCTestCase {
    func testWeeklyAveragesReturnsFixedWindow() {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        let entries = [
            AnyWeightEntry(date: now, weightKg: 80),
            AnyWeightEntry(date: now, weightKg: 82)
        ]

        let points = WeightTrend.weeklyAveragePoints(weights: entries, calendar: calendar, weeks: 8)
        XCTAssertEqual(points.count, 8)
        XCTAssertEqual(points.last?.averageWeightKg, 81)
    }
}
