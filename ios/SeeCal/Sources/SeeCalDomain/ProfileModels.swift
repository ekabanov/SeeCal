import Foundation

public enum BiologicalSex: String, Codable, CaseIterable, Sendable {
    case male
    case female
}

public enum ActivityLevel: String, Codable, CaseIterable, Sendable {
    case sedentary
    case lightlyActive
    case moderatelyActive
    case veryActive

    public var multiplier: Double {
        switch self {
        case .sedentary:
            return 1.2
        case .lightlyActive:
            return 1.375
        case .moderatelyActive:
            return 1.55
        case .veryActive:
            return 1.725
        }
    }
}

public enum GoalPace: String, Codable, CaseIterable, Sendable {
    case loseSlow
    case loseModerate
    case maintain
    case gainSlow

    public var weeklyKgDelta: Double {
        switch self {
        case .loseSlow:
            return -0.25
        case .loseModerate:
            return -0.5
        case .maintain:
            return 0
        case .gainSlow:
            return 0.25
        }
    }
}

public struct UserProfile: Codable, Equatable, Sendable {
    public var biologicalSex: BiologicalSex
    public var ageYears: Int
    public var heightCm: Double
    public var weightKg: Double
    public var activityLevel: ActivityLevel
    public var goalPace: GoalPace

    public init(
        biologicalSex: BiologicalSex,
        ageYears: Int,
        heightCm: Double,
        weightKg: Double,
        activityLevel: ActivityLevel,
        goalPace: GoalPace
    ) {
        self.biologicalSex = biologicalSex
        self.ageYears = ageYears
        self.heightCm = heightCm
        self.weightKg = weightKg
        self.activityLevel = activityLevel
        self.goalPace = goalPace
    }
}

public enum DailyNutritionTargetPlanner {
    public static func deriveTarget(from profile: UserProfile) -> DailyNutritionTarget {
        let bmr = miffinStJeor(profile: profile)
        let tdee = bmr * profile.activityLevel.multiplier

        // Approx 7700 kcal per kg change; convert weekly pace to daily.
        let dailyGoalAdjustment = profile.goalPace.weeklyKgDelta * 7700 / 7
        let targetCalories = max(1200, tdee + dailyGoalAdjustment)

        let proteinPerKg = profile.goalPace == .gainSlow ? 1.6 : 1.8
        let proteinGrams = max(80, profile.weightKg * proteinPerKg)
        let fatGrams = max(45, profile.weightKg * 0.8)
        let carbsCalories = max(0, targetCalories - (proteinGrams * 4 + fatGrams * 9))
        let carbsGrams = carbsCalories / 4

        return DailyNutritionTarget(
            calories: round(targetCalories),
            proteinGrams: round(proteinGrams),
            fatGrams: round(fatGrams),
            carbsGrams: round(carbsGrams)
        )
    }

    private static func miffinStJeor(profile: UserProfile) -> Double {
        let base = 10 * profile.weightKg + 6.25 * profile.heightCm - 5 * Double(profile.ageYears)
        switch profile.biologicalSex {
        case .male:
            return base + 5
        case .female:
            return base - 161
        }
    }
}
