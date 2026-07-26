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

    // MARK: - Fixtures

    private static let utcCalendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }()

    /// Sunday 2026-07-26 — a known weekday, so the trailing-7-day week window
    /// (Mon 07-20 … Sun 07-26) has an independently-verifiable weekday
    /// sequence.
    private static let referenceDate: Date = {
        var components = DateComponents()
        components.year = 2026
        components.month = 7
        components.day = 26
        return utcCalendar.date(from: components)!
    }()

    private func day(offsetFromReference offset: Int) -> Date {
        Self.utcCalendar.date(byAdding: .day, value: -offset, to: Self.referenceDate)!
    }

    private func entry(offset: Int, calories: Double, protein: Double = 0) -> AnyMealLogEntry {
        AnyMealLogEntry(
            createdAt: day(offsetFromReference: offset),
            totals: NutritionTotals(calories: calories, proteinGrams: protein, fatGrams: 0, carbsGrams: 0)
        )
    }

    // MARK: - Color-class thresholds (exact boundary behavior)

    func testColorClassIsNoneForUnloggedOrNonPositiveValues() {
        XCTAssertEqual(ProgressAggregator.colorClass(value: 0, goalCalories: 2000), .none)
        XCTAssertEqual(ProgressAggregator.colorClass(value: -100, goalCalories: 2000), .none)
        // A non-positive goal can't be evaluated as a ratio; treat as none rather
        // than dividing by zero/negative.
        XCTAssertEqual(ProgressAggregator.colorClass(value: 500, goalCalories: 0), .none)
    }

    func testColorClassOkIncludesExactly105xGoal() {
        // goal 2000 → 1.05x == 2100 exactly must still be "ok" (`r <= 1.05`).
        XCTAssertEqual(ProgressAggregator.colorClass(value: 2100, goalCalories: 2000), .ok)
        XCTAssertEqual(ProgressAggregator.colorClass(value: 2000, goalCalories: 2000), .ok)
        XCTAssertEqual(ProgressAggregator.colorClass(value: 1, goalCalories: 2000), .ok)
    }

    func testColorClassWarnJustAbove105xGoal() {
        XCTAssertEqual(ProgressAggregator.colorClass(value: 2100.01, goalCalories: 2000), .warn)
    }

    func testColorClassWarnIncludesExactly125xGoal() {
        // goal 2000 → 1.25x == 2500 exactly must still be "warn" (`r <= 1.25`).
        XCTAssertEqual(ProgressAggregator.colorClass(value: 2500, goalCalories: 2000), .warn)
    }

    func testColorClassOverJustAbove125xGoal() {
        XCTAssertEqual(ProgressAggregator.colorClass(value: 2500.01, goalCalories: 2000), .over)
        XCTAssertEqual(ProgressAggregator.colorClass(value: 10_000, goalCalories: 2000), .over)
    }

    // MARK: - Week (7 daily bars, weekday labels, today flag)

    func testWeekChartDataLabelsAndColorsAndTodayFlag() {
        let entries = [
            entry(offset: 0, calories: 2100),  // today, Sun — ok (exactly 1.05x of 2000)
            entry(offset: 1, calories: 2600),  // Sat — over
            entry(offset: 3, calories: 1900)   // Thu — ok
            // offsets 2, 4, 5, 6 unlogged
        ]

        let data = ProgressAggregator.historyChartData(
            from: entries,
            goalCalories: 2000,
            range: .week,
            calendar: Self.utcCalendar,
            referenceDate: Self.referenceDate
        )

        XCTAssertEqual(data.range, .week)
        XCTAssertEqual(data.subtitle, "Last 7 days")
        XCTAssertEqual(data.bars.count, 7)
        XCTAssertEqual(data.daysInRange, 7)

        // Mon(-6) Tue(-5) Wed(-4) Thu(-3) Fri(-2) Sat(-1) Sun(today), oldest first.
        XCTAssertEqual(data.bars.map(\.label), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        // Every bar in "week" mode carries a label — none are density-suppressed.
        XCTAssertTrue(data.bars.allSatisfy { $0.label != nil && !$0.label!.isEmpty })

        // isToday is true ONLY on the last (rightmost) bar.
        XCTAssertEqual(data.bars.map(\.isToday), [false, false, false, false, false, false, true])

        // Thu (offset 3, index 3): 1900 kcal, ok. Sat (offset 1, index 5): 2600, over.
        // Sun/today (offset 0, index 6): 2100 == 1.05x goal exactly, ok.
        XCTAssertEqual(data.bars[3].value, 1900)
        XCTAssertEqual(data.bars[3].colorClass, .ok)
        XCTAssertEqual(data.bars[5].value, 2600)
        XCTAssertEqual(data.bars[5].colorClass, .over)
        XCTAssertEqual(data.bars[6].value, 2100)
        XCTAssertEqual(data.bars[6].colorClass, .ok)

        // Unlogged days are zero-valued and colorClass .none.
        XCTAssertEqual(data.bars[0].value, 0)
        XCTAssertEqual(data.bars[0].colorClass, .none)

        XCTAssertEqual(data.daysLogged, 3)
        // avg kcal/day over LOGGED days only: (2100+2600+1900)/3.
        XCTAssertEqual(data.averageCaloriesPerLoggedDay ?? -1, (2100.0 + 2600 + 1900) / 3, accuracy: 0.001)
    }

    // MARK: - Month (30 dense daily bars, every-7th-day labels)

    func testMonthChartDataLabelDensity() {
        let data = ProgressAggregator.historyChartData(
            from: [],
            goalCalories: 2000,
            range: .month,
            calendar: Self.utcCalendar,
            referenceDate: Self.referenceDate
        )

        XCTAssertEqual(data.subtitle, "Last 30 days")
        XCTAssertEqual(data.bars.count, 30)
        XCTAssertEqual(data.daysInRange, 30)

        // Oldest bar (index 0, "ago" = 29) is NOT a multiple of 7 → no label,
        // matching the prototype's `"−28d…now"` range (the true oldest bar has
        // no label; −28d is the oldest LABELED bar).
        XCTAssertNil(data.bars[0].label)
        // ago=28 → index 1 → "−28d".
        XCTAssertEqual(data.bars[1].label, "\u{2212}28d")
        // ago=21 → index 8 → "−21d".
        XCTAssertEqual(data.bars[8].label, "\u{2212}21d")
        // ago=14 → index 15 → "−14d".
        XCTAssertEqual(data.bars[15].label, "\u{2212}14d")
        // ago=7 → index 22 → "−7d".
        XCTAssertEqual(data.bars[22].label, "\u{2212}7d")
        // ago=0 (today) → index 29 → "now".
        XCTAssertEqual(data.bars[29].label, "now")
        XCTAssertTrue(data.bars[29].isToday)

        // A handful of non-multiple-of-7 positions carry no label.
        XCTAssertNil(data.bars[2].label)
        XCTAssertNil(data.bars[10].label)
        XCTAssertNil(data.bars[28].label)

        // Only the last bar is flagged "today".
        XCTAssertEqual(data.bars.filter(\.isToday).count, 1)
    }

    // MARK: - 6 months (26 weekly-average bars)

    func testHalfYearWeeklyAverageOverPartialWeekOnly() {
        // The week ending 7 days ago (offsets 7...13) is the second-to-last
        // week bar. Log 3 of its 7 days; the other 4 stay unlogged and must
        // NOT be averaged in as zeros.
        let entries = [
            entry(offset: 7, calories: 1000),
            entry(offset: 9, calories: 3000),
            entry(offset: 11, calories: 2000)
            // offsets 8, 10, 12, 13 unlogged
        ]

        let data = ProgressAggregator.historyChartData(
            from: entries,
            goalCalories: 2000,
            range: .halfYear,
            calendar: Self.utcCalendar,
            referenceDate: Self.referenceDate
        )

        XCTAssertEqual(data.subtitle, "Last 6 months · weekly average")
        XCTAssertEqual(data.bars.count, 26)
        XCTAssertEqual(data.daysInRange, 26 * 7)

        // The current (most recent) week is the last bar; the week starting
        // 7 days before that is the second-to-last.
        let partialWeekBar = data.bars[24]
        XCTAssertEqual(partialWeekBar.value, (1000.0 + 3000 + 2000) / 3, accuracy: 0.001)
        XCTAssertEqual(partialWeekBar.colorClass, ProgressAggregator.colorClass(value: (1000.0 + 3000 + 2000) / 3, goalCalories: 2000))
        XCTAssertFalse(partialWeekBar.isToday)
    }

    func testHalfYearEmptyWeekIsUnlogged() {
        // No entries at all anywhere in the 6-month window: every week is
        // empty → value 0, colorClass .none (never a false-zero average).
        let data = ProgressAggregator.historyChartData(
            from: [],
            goalCalories: 2000,
            range: .halfYear,
            calendar: Self.utcCalendar,
            referenceDate: Self.referenceDate
        )

        XCTAssertTrue(data.bars.allSatisfy { $0.value == 0 })
        XCTAssertTrue(data.bars.allSatisfy { $0.colorClass == .none })
        XCTAssertEqual(data.daysLogged, 0)
        XCTAssertNil(data.averageCaloriesPerLoggedDay)
        XCTAssertNil(data.averageProteinPerLoggedDay)
    }

    func testHalfYearTodayFlagOnlyOnLastBar() {
        let data = ProgressAggregator.historyChartData(
            from: [],
            goalCalories: 2000,
            range: .halfYear,
            calendar: Self.utcCalendar,
            referenceDate: Self.referenceDate
        )
        XCTAssertEqual(data.bars.filter(\.isToday).count, 1)
        XCTAssertTrue(data.bars.last!.isToday)
        XCTAssertEqual(data.bars.last?.label, "now")
    }

    func testHalfYearMonthLabelsEveryFourWeeks() {
        let data = ProgressAggregator.historyChartData(
            from: [],
            goalCalories: 2000,
            range: .halfYear,
            calendar: Self.utcCalendar,
            referenceDate: Self.referenceDate
        )

        // ago = 25 - weekIndex. Labeled weeks are ago in {0,4,8,12,16,20,24}.
        // ago=0 → "now" (already covered above); the rest carry the actual
        // month abbreviation of that week's start date (independently computed
        // against the fixed 2026-07-26 reference date).
        XCTAssertEqual(data.bars[21].label, "Jun")  // ago=4
        XCTAssertEqual(data.bars[17].label, "May")  // ago=8
        XCTAssertEqual(data.bars[13].label, "Apr")  // ago=12
        XCTAssertEqual(data.bars[9].label, "Mar")   // ago=16
        XCTAssertEqual(data.bars[5].label, "Mar")   // ago=20
        XCTAssertEqual(data.bars[1].label, "Feb")   // ago=24

        // Weeks that are neither "now" nor a multiple of 4 weeks carry no label.
        XCTAssertNil(data.bars[0].label)  // ago=25
        XCTAssertNil(data.bars[22].label) // ago=3
        XCTAssertNil(data.bars[20].label) // ago=5
    }

    // MARK: - Empty range (no entries anywhere)

    func testEmptyEntriesProduceAllUnloggedBarsForEveryRange() {
        for range in HistoryRange.allCases {
            let data = ProgressAggregator.historyChartData(
                from: [],
                goalCalories: 2000,
                range: range,
                calendar: Self.utcCalendar,
                referenceDate: Self.referenceDate
            )
            XCTAssertTrue(data.bars.allSatisfy { $0.value == 0 && $0.colorClass == .none }, "range \(range)")
            XCTAssertEqual(data.daysLogged, 0, "range \(range)")
            XCTAssertNil(data.averageCaloriesPerLoggedDay, "range \(range)")
            XCTAssertNil(data.averageProteinPerLoggedDay, "range \(range)")
            // Axis still scales to at least the goal even with no data plotted.
            XCTAssertEqual(data.axisMax, 2000 * 1.08, accuracy: 0.001, "range \(range)")
        }
    }

    // MARK: - Stats row averages (daily granularity, independent of bar bucketing)

    func testAverageProteinPerLoggedDayOnlyCountsLoggedDays() {
        let entries = [
            entry(offset: 0, calories: 2000, protein: 150),
            entry(offset: 1, calories: 1800, protein: 100)
            // offsets 2-6 unlogged.
        ]
        let data = ProgressAggregator.historyChartData(
            from: entries,
            goalCalories: 2000,
            range: .week,
            calendar: Self.utcCalendar,
            referenceDate: Self.referenceDate
        )
        XCTAssertEqual(data.daysLogged, 2)
        XCTAssertEqual(data.averageProteinPerLoggedDay ?? -1, 125, accuracy: 0.001)
    }
}
