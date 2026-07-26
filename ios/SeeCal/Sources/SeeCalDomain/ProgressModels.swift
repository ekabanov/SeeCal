import Foundation

public struct DailyProgressPoint: Codable, Equatable, Sendable {
    public var dayStart: Date
    public var calories: Double
    public var proteinGrams: Double
    public var fatGrams: Double
    public var carbsGrams: Double

    public init(dayStart: Date, calories: Double, proteinGrams: Double, fatGrams: Double, carbsGrams: Double) {
        self.dayStart = dayStart
        self.calories = calories
        self.proteinGrams = proteinGrams
        self.fatGrams = fatGrams
        self.carbsGrams = carbsGrams
    }
}

// MARK: - History chart (spec §6)

/// The three selectable ranges on the History screen's `.seg` segmented control.
public enum HistoryRange: String, CaseIterable, Equatable, Sendable {
    case week
    case month
    case halfYear
}

/// A bar's adherence-to-goal color, matching the prototype's `goalClass(v)`
/// exactly: `!v` (unlogged/zero) → `.none` (no color class in the prototype —
/// the bar renders in the neutral `--app-line` gray); otherwise the ratio
/// `value/goal` is compared against the two thresholds.
public enum ChartColorClass: String, Codable, Equatable, Sendable {
    /// `r <= 1.05` — `.cbar.ok` (brand basil green).
    case ok
    /// `1.05 < r <= 1.25` — `.cbar.warn` (fat/amber).
    case warn
    /// `r > 1.25` — `.cbar.over` (danger/red).
    case over
    /// Unlogged day/week (value is zero) — neutral `--app-line` gray, no legend
    /// swatch. Named `none` to mirror the prototype's `goalClass` returning `""`.
    case none
}

/// One bar in the history chart: a value (0 = unlogged), an optional axis-zone
/// label (density rules below decide which bars get one), a color class, and
/// whether this is the rightmost/most-recent bar (today, or the current
/// in-progress week for the 6-month range) — the prototype applies the `.today`
/// outline to the last bar unconditionally, regardless of granularity.
public struct HistoryBar: Equatable, Sendable {
    public var value: Double
    public var label: String?
    public var colorClass: ChartColorClass
    public var isToday: Bool

    public init(value: Double, label: String?, colorClass: ChartColorClass, isToday: Bool) {
        self.value = value
        self.label = label
        self.colorClass = colorClass
        self.isToday = isToday
    }
}

/// Everything `HistoryChartView` + the stats row need to render one range,
/// pre-computed so the view layer stays pure presentation.
public struct HistoryChartData: Equatable, Sendable {
    public var range: HistoryRange
    public var bars: [HistoryBar]
    public var goalCalories: Double
    /// Axis scale ceiling: `max(goal, max(bar values)) * 1.08`, matching the
    /// prototype's `renderHistory()` (`var max = Math.max(state.goal, ...)*1.08`).
    public var axisMax: Double
    /// `histSub` text: "Last 7 days" / "Last 30 days" / "Last 6 months · weekly average".
    public var subtitle: String
    /// Whole calendar days spanned by the range (7 / 30 / 182), used for the
    /// "N of M days logged" stat regardless of the bar granularity (so 6-month
    /// logging adherence reads as actual days, not weeks).
    public var daysInRange: Int
    public var daysLogged: Int
    /// nil when zero days were logged (prototype: `fmt(Math.round(sum/logged.length))`
    /// on an empty `logged` array is `NaN`; the UI renders "—" instead).
    public var averageCaloriesPerLoggedDay: Double?
    public var averageProteinPerLoggedDay: Double?

    public init(
        range: HistoryRange,
        bars: [HistoryBar],
        goalCalories: Double,
        axisMax: Double,
        subtitle: String,
        daysInRange: Int,
        daysLogged: Int,
        averageCaloriesPerLoggedDay: Double?,
        averageProteinPerLoggedDay: Double?
    ) {
        self.range = range
        self.bars = bars
        self.goalCalories = goalCalories
        self.axisMax = axisMax
        self.subtitle = subtitle
        self.daysInRange = daysInRange
        self.daysLogged = daysLogged
        self.averageCaloriesPerLoggedDay = averageCaloriesPerLoggedDay
        self.averageProteinPerLoggedDay = averageProteinPerLoggedDay
    }
}

public enum ProgressAggregator {
    public static func dailyPoints(
        from entries: [AnyMealLogEntry],
        calendar: Calendar = .current,
        days: Int,
        referenceDate: Date = Date()
    ) -> [DailyProgressPoint] {
        dailyTotalsSeries(from: entries, calendar: calendar, referenceDate: referenceDate, days: days).map {
            DailyProgressPoint(
                dayStart: $0.dayStart,
                calories: $0.totals.calories,
                proteinGrams: $0.totals.proteinGrams,
                fatGrams: $0.totals.fatGrams,
                carbsGrams: $0.totals.carbsGrams
            )
        }
    }

    // MARK: - History chart

    /// Adherence color class for one value against the daily goal — exactly the
    /// prototype's `goalClass(v)`: value `<= 0` (or a non-positive goal, which
    /// can't be evaluated) is `.none`; otherwise `value/goalCalories <= 1.05` is
    /// `.ok`, `<= 1.25` is `.warn`, else `.over`. Both boundary comparisons are
    /// inclusive on the lower side (a value at exactly `goal*1.05` is still
    /// `.ok`; exactly `goal*1.25` is still `.warn`).
    public static func colorClass(value: Double, goalCalories: Double) -> ChartColorClass {
        guard value > 0, goalCalories > 0 else { return .none }
        let ratio = value / goalCalories
        if ratio <= 1.05 { return .ok }
        if ratio <= 1.25 { return .warn }
        return .over
    }

    /// Builds the full chart dataset for one range. `referenceDate` is "today"
    /// (injectable for deterministic tests); `entries` need only cover the
    /// window the range requires (a week/month/6-months back from
    /// `referenceDate` — anything older is ignored).
    public static func historyChartData(
        from entries: [AnyMealLogEntry],
        goalCalories: Double,
        range: HistoryRange,
        calendar: Calendar = .current,
        referenceDate: Date = Date()
    ) -> HistoryChartData {
        switch range {
        case .week:
            return weekChartData(from: entries, goalCalories: goalCalories, calendar: calendar, referenceDate: referenceDate)
        case .month:
            return monthChartData(from: entries, goalCalories: goalCalories, calendar: calendar, referenceDate: referenceDate)
        case .halfYear:
            return halfYearChartData(from: entries, goalCalories: goalCalories, calendar: calendar, referenceDate: referenceDate)
        }
    }

    // MARK: Week (7 daily bars, Sun..Sat labels)

    private static func weekChartData(
        from entries: [AnyMealLogEntry],
        goalCalories: Double,
        calendar: Calendar,
        referenceDate: Date
    ) -> HistoryChartData {
        let days = 7
        let daily = dailyTotalsSeries(from: entries, calendar: calendar, referenceDate: referenceDate, days: days)
        let bars = daily.enumerated().map { index, point -> HistoryBar in
            HistoryBar(
                value: point.totals.calories,
                label: weekdayAbbreviation(for: point.dayStart, calendar: calendar),
                colorClass: colorClass(value: point.totals.calories, goalCalories: goalCalories),
                isToday: index == daily.count - 1
            )
        }
        return makeChartData(
            range: .week,
            bars: bars,
            goalCalories: goalCalories,
            subtitle: "Last 7 days",
            daysInRange: days,
            dailySeries: daily
        )
    }

    // MARK: Month (30 daily slim bars, "−28d…now" every 7th)

    private static func monthChartData(
        from entries: [AnyMealLogEntry],
        goalCalories: Double,
        calendar: Calendar,
        referenceDate: Date
    ) -> HistoryChartData {
        let days = 30
        let daily = dailyTotalsSeries(from: entries, calendar: calendar, referenceDate: referenceDate, days: days)
        let bars = daily.enumerated().map { index, point -> HistoryBar in
            let ago = daily.count - 1 - index
            let label: String?
            if ago == 0 {
                label = "now"
            } else if ago % 7 == 0 {
                label = "\u{2212}\(ago)d"
            } else {
                label = nil
            }
            return HistoryBar(
                value: point.totals.calories,
                label: label,
                colorClass: colorClass(value: point.totals.calories, goalCalories: goalCalories),
                isToday: ago == 0
            )
        }
        return makeChartData(
            range: .month,
            bars: bars,
            goalCalories: goalCalories,
            subtitle: "Last 30 days",
            daysInRange: days,
            dailySeries: daily
        )
    }

    // MARK: 6 months (26 weekly-average bars, month labels every 4 weeks + "now")

    private static func halfYearChartData(
        from entries: [AnyMealLogEntry],
        goalCalories: Double,
        calendar: Calendar,
        referenceDate: Date
    ) -> HistoryChartData {
        let weeks = 26
        let totalDays = weeks * 7
        let daily = dailyTotalsSeries(from: entries, calendar: calendar, referenceDate: referenceDate, days: totalDays)

        var bars: [HistoryBar] = []
        bars.reserveCapacity(weeks)
        for weekIndex in 0..<weeks {
            let ago = weeks - 1 - weekIndex // 0 = the current (most recent) week
            let endIndex = totalDays - 1 - (ago * 7)
            let startIndex = endIndex - 6
            let window = daily[startIndex...endIndex]
            // "average over LOGGED days in each week, empty week = unlogged":
            // a day with calories <= 0 is treated as not logged (same convention
            // as the prototype's `if (v)` truthiness check), so a week where
            // every day is unlogged averages to 0/`.none` rather than a false 0.
            let loggedCalories = window.map(\.totals.calories).filter { $0 > 0 }
            let value = loggedCalories.isEmpty ? 0 : loggedCalories.reduce(0, +) / Double(loggedCalories.count)

            let label: String?
            if ago == 0 {
                label = "now"
            } else if ago % 4 == 0 {
                label = monthAbbreviation(for: window.first!.dayStart, calendar: calendar)
            } else {
                label = nil
            }

            bars.append(HistoryBar(
                value: value,
                label: label,
                colorClass: colorClass(value: value, goalCalories: goalCalories),
                isToday: ago == 0
            ))
        }

        return makeChartData(
            range: .halfYear,
            bars: bars,
            goalCalories: goalCalories,
            subtitle: "Last 6 months · weekly average",
            daysInRange: totalDays,
            dailySeries: daily
        )
    }

    // MARK: Shared assembly

    /// Stats (days logged, averages) are always computed at DAILY granularity
    /// from `dailySeries` — even for the 6-month range, where the bars
    /// themselves are weekly averages — so "N of M days logged" and "avg
    /// kcal/day" stay meaningful, stable metrics independent of how the chart
    /// buckets its bars.
    private static func makeChartData(
        range: HistoryRange,
        bars: [HistoryBar],
        goalCalories: Double,
        subtitle: String,
        daysInRange: Int,
        dailySeries: [(dayStart: Date, totals: NutritionTotals)]
    ) -> HistoryChartData {
        let maxBarValue = bars.map(\.value).max() ?? 0
        let axisMax = max(goalCalories, maxBarValue) * 1.08

        let loggedDays = dailySeries.filter { $0.totals.calories > 0 }
        let avgCalories = loggedDays.isEmpty ? nil : loggedDays.reduce(0.0) { $0 + $1.totals.calories } / Double(loggedDays.count)
        let avgProtein = loggedDays.isEmpty ? nil : loggedDays.reduce(0.0) { $0 + $1.totals.proteinGrams } / Double(loggedDays.count)

        return HistoryChartData(
            range: range,
            bars: bars,
            goalCalories: goalCalories,
            axisMax: axisMax,
            subtitle: subtitle,
            daysInRange: daysInRange,
            daysLogged: loggedDays.count,
            averageCaloriesPerLoggedDay: avgCalories,
            averageProteinPerLoggedDay: avgProtein
        )
    }

    // MARK: - Shared daily bucketing

    private static func dailyTotalsSeries(
        from entries: [AnyMealLogEntry],
        calendar: Calendar,
        referenceDate: Date,
        days: Int
    ) -> [(dayStart: Date, totals: NutritionTotals)] {
        guard days > 0 else { return [] }
        let todayStart = calendar.startOfDay(for: referenceDate)

        var buckets: [Date: NutritionTotals] = [:]
        for entry in entries {
            let day = calendar.startOfDay(for: entry.createdAt)
            buckets[day] = (buckets[day] ?? NutritionTotals()) + entry.totals
        }

        return (0..<days).reversed().compactMap { offset in
            guard let day = calendar.date(byAdding: .day, value: -offset, to: todayStart) else {
                return nil
            }
            return (day, buckets[day] ?? NutritionTotals())
        }
    }

    /// "Sun"/"Mon"/.../"Sat" for the given date, matching the prototype's fixed
    /// `["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]` label set exactly (each bar
    /// labeled by its OWN actual weekday, not a hardcoded Sunday-first sequence —
    /// the prototype's demo data is synthetic and always ends on a Saturday,
    /// which a real trailing-7-day window ending "today" cannot assume).
    private static func weekdayAbbreviation(for date: Date, calendar: Calendar) -> String {
        var fixedCalendar = calendar
        fixedCalendar.locale = Locale(identifier: "en_US_POSIX")
        let formatter = DateFormatter()
        formatter.calendar = fixedCalendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "EEE"
        return formatter.string(from: date)
    }

    /// "Feb"/"Mar"/... for the given date (used by the 6-month range's every-
    /// 4-weeks month labels).
    private static func monthAbbreviation(for date: Date, calendar: Calendar) -> String {
        var fixedCalendar = calendar
        fixedCalendar.locale = Locale(identifier: "en_US_POSIX")
        let formatter = DateFormatter()
        formatter.calendar = fixedCalendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "MMM"
        return formatter.string(from: date)
    }
}

/// A type-erased view of a persisted meal log entry, carrying only what aggregation
/// needs: when it happened and its (already-scaled-items-derived) totals. Kept
/// independent of `SeeCalPersistence.MealLogEntry` so this module has no dependency
/// on the persistence layer.
public struct AnyMealLogEntry: Sendable {
    public var createdAt: Date
    public var totals: NutritionTotals

    public init(createdAt: Date, totals: NutritionTotals) {
        self.createdAt = createdAt
        self.totals = totals
    }
}
