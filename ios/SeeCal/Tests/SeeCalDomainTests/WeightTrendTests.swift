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

    // MARK: - monthChangeKg (spec §7: Profile weight trend note)

    func testMonthChangeUsesOnlyTrailing30DayWindow() {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        let entries = [
            AnyWeightEntry(date: calendar.date(byAdding: .day, value: -40, to: now)!, weightKg: 85.0), // outside window
            AnyWeightEntry(date: calendar.date(byAdding: .day, value: -20, to: now)!, weightKg: 79.6),
            AnyWeightEntry(date: now, weightKg: 78.4)
        ]

        let change = WeightTrend.monthChangeKg(weights: entries, now: now, calendar: calendar)
        XCTAssertEqual(change!, -1.2, accuracy: 0.0001)
    }

    func testMonthChangeNilWhenFewerThanTwoEntriesInWindow() {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()

        XCTAssertNil(WeightTrend.monthChangeKg(weights: [], now: now, calendar: calendar))
        XCTAssertNil(WeightTrend.monthChangeKg(
            weights: [AnyWeightEntry(date: now, weightKg: 78.4)],
            now: now,
            calendar: calendar
        ))
        // Two entries, but only one inside the trailing 30 days.
        XCTAssertNil(WeightTrend.monthChangeKg(
            weights: [
                AnyWeightEntry(date: calendar.date(byAdding: .day, value: -45, to: now)!, weightKg: 80),
                AnyWeightEntry(date: now, weightKg: 78.4)
            ],
            now: now,
            calendar: calendar
        ))
    }

    func testMonthChangePositiveForGain() {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        let entries = [
            AnyWeightEntry(date: calendar.date(byAdding: .day, value: -10, to: now)!, weightKg: 78.0),
            AnyWeightEntry(date: now, weightKg: 78.4)
        ]

        let change = WeightTrend.monthChangeKg(weights: entries, now: now, calendar: calendar)
        XCTAssertEqual(change!, 0.4, accuracy: 0.0001)
    }
}
