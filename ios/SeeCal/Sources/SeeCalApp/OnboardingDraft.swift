import Foundation
import SeeCalDomain

public struct OnboardingDraft: Equatable, Sendable {
    public var sex: BiologicalSex
    public var dateOfBirth: Date
    public var heightCmText: String
    public var weightKgText: String
    public var activity: ActivityLevel
    public var weeklyRateKg: Double

    public init(
        sex: BiologicalSex = .male,
        dateOfBirth: Date = OnboardingDraft.defaultDateOfBirth,
        heightCmText: String = "175",
        weightKgText: String = "80",
        activity: ActivityLevel = .moderate,
        weeklyRateKg: Double = UserProfile.defaultWeeklyRateKg
    ) {
        self.sex = sex
        self.dateOfBirth = dateOfBirth
        self.heightCmText = heightCmText
        self.weightKgText = weightKgText
        self.activity = activity
        self.weeklyRateKg = UserProfile.clampWeeklyRate(weeklyRateKg)
    }

    public init(profile: UserProfile) {
        sex = profile.sex
        dateOfBirth = profile.dateOfBirth
        heightCmText = String(profile.heightCm)
        weightKgText = String(format: "%.1f", profile.weightKg)
        activity = profile.activity
        weeklyRateKg = profile.weeklyRateKg
    }

    public func toUserProfile() throws -> UserProfile {
        let height = try Self.parseInteger(heightCmText, field: "height", min: 120, max: 230)
        let weight = try Self.parseDecimal(weightKgText, field: "weight", min: 35, max: 250)

        return UserProfile(
            sex: sex,
            dateOfBirth: dateOfBirth,
            heightCm: height,
            weightKg: weight,
            activity: activity,
            weeklyRateKg: weeklyRateKg
        )
    }

    public static var defaultDateOfBirth: Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .current
        return calendar.date(byAdding: .year, value: -30, to: Date()) ?? Date()
    }

    private static func parseInteger(_ text: String, field: String, min: Int, max: Int) throws -> Int {
        guard let value = Int(text), value >= min, value <= max else {
            throw NSError(domain: "OnboardingDraft", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid \(field)"])
        }
        return value
    }

    private static func parseDecimal(_ text: String, field: String, min: Double, max: Double) throws -> Double {
        guard let value = Double(text), value.isFinite, value >= min, value <= max else {
            throw NSError(domain: "OnboardingDraft", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid \(field)"])
        }
        return value
    }
}
