import Foundation
import SeeCalDomain

public struct OnboardingDraft: Equatable, Sendable {
    public var biologicalSex: BiologicalSex
    public var ageYearsText: String
    public var heightCmText: String
    public var weightKgText: String
    public var activityLevel: ActivityLevel
    public var goalPace: GoalPace

    public init(
        biologicalSex: BiologicalSex = .male,
        ageYearsText: String = "30",
        heightCmText: String = "175",
        weightKgText: String = "80",
        activityLevel: ActivityLevel = .moderatelyActive,
        goalPace: GoalPace = .maintain
    ) {
        self.biologicalSex = biologicalSex
        self.ageYearsText = ageYearsText
        self.heightCmText = heightCmText
        self.weightKgText = weightKgText
        self.activityLevel = activityLevel
        self.goalPace = goalPace
    }

    public init(profile: UserProfile) {
        biologicalSex = profile.biologicalSex
        ageYearsText = String(profile.ageYears)
        heightCmText = String(Int(profile.heightCm.rounded()))
        weightKgText = String(format: "%.1f", profile.weightKg)
        activityLevel = profile.activityLevel
        goalPace = profile.goalPace
    }

    public func toUserProfile() throws -> UserProfile {
        let age = try Self.parseInteger(ageYearsText, field: "age", min: 14, max: 100)
        let height = try Self.parseDecimal(heightCmText, field: "height", min: 120, max: 230)
        let weight = try Self.parseDecimal(weightKgText, field: "weight", min: 35, max: 250)

        return UserProfile(
            biologicalSex: biologicalSex,
            ageYears: age,
            heightCm: height,
            weightKg: weight,
            activityLevel: activityLevel,
            goalPace: goalPace
        )
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
