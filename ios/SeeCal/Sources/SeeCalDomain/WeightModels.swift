import Foundation

public struct WeeklyWeightPoint: Codable, Equatable, Sendable {
    public var weekStart: Date
    public var averageWeightKg: Double

    public init(weekStart: Date, averageWeightKg: Double) {
        self.weekStart = weekStart
        self.averageWeightKg = averageWeightKg
    }
}

public enum WeightTrend {
    public static func weeklyAveragePoints(
        weights: [AnyWeightEntry],
        calendar: Calendar = .current,
        weeks: Int = 8
    ) -> [WeeklyWeightPoint] {
        guard weeks > 0 else { return [] }

        let today = Date()
        let startOfThisWeek = calendar.dateInterval(of: .weekOfYear, for: today)?.start ?? calendar.startOfDay(for: today)
        var buckets: [Date: [Double]] = [:]

        for entry in weights {
            guard let weekStart = calendar.dateInterval(of: .weekOfYear, for: entry.date)?.start else {
                continue
            }
            buckets[weekStart, default: []].append(entry.weightKg)
        }

        return (0..<weeks).reversed().compactMap { offset in
            guard let week = calendar.date(byAdding: .weekOfYear, value: -offset, to: startOfThisWeek) else {
                return nil
            }
            let values = buckets[week] ?? []
            let avg = values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
            return WeeklyWeightPoint(weekStart: week, averageWeightKg: avg)
        }
    }
}

public struct AnyWeightEntry: Sendable {
    public var date: Date
    public var weightKg: Double

    public init(date: Date, weightKg: Double) {
        self.date = date
        self.weightKg = weightKg
    }
}
