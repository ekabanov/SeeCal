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

public enum ProgressAggregator {
    public static func dailyPoints(
        from entries: [AnyMealLogEntry],
        calendar: Calendar = .current,
        days: Int
    ) -> [DailyProgressPoint] {
        guard days > 0 else { return [] }
        let todayStart = calendar.startOfDay(for: Date())

        var buckets: [Date: NutritionTotals] = [:]
        for entry in entries {
            let day = calendar.startOfDay(for: entry.createdAt)
            let current = buckets[day] ?? NutritionTotals()
            let added = NutritionTotals(
                calories: entry.scanResult.totalCalories,
                proteinGrams: entry.scanResult.proteinGrams,
                fatGrams: entry.scanResult.fatGrams,
                carbsGrams: entry.scanResult.carbsGrams
            )
            buckets[day] = current + added
        }

        return (0..<days).reversed().compactMap { offset in
            guard let day = calendar.date(byAdding: .day, value: -offset, to: todayStart) else {
                return nil
            }
            let total = buckets[day] ?? NutritionTotals()
            return DailyProgressPoint(
                dayStart: day,
                calories: total.calories,
                proteinGrams: total.proteinGrams,
                fatGrams: total.fatGrams,
                carbsGrams: total.carbsGrams
            )
        }
    }
}

public struct AnyMealLogEntry: Sendable {
    public var createdAt: Date
    public var scanResult: FoodScanResult

    public init(createdAt: Date, scanResult: FoodScanResult) {
        self.createdAt = createdAt
        self.scanResult = scanResult
    }
}
