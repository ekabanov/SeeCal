import Foundation

/// Pure goal-calculation engine per spec §2. Replaces the old `GoalEditDraft`
/// manual-goal-editing flow: the daily target is always derived from the user's
/// profile, never independently stored/edited.
///
/// Formulas (spec §2):
/// ```
/// BMR  = 10*kg + 6.25*cm - 5*age(years) + (male ? +5 : -161)   // Mifflin-St Jeor
/// TDEE = BMR * factor   // sedentary 1.2, light 1.375, moderate 1.55, active 1.725
/// goal = max(1200, round10(TDEE + weeklyRateKg * 7700 / 7))
/// ```
/// Reference vector: male, dob 1988-03-14, now 2026-07-26 (age 38), 183 cm, 78.4 kg,
/// moderate, -0.5 kg/wk -> BMR 1743, TDEE 2701, goal 2150.
public enum GoalCalculator {
    /// A fixed-UTC Gregorian calendar, used as the default for all calculations so
    /// results are deterministic regardless of the device's local time zone.
    public static let defaultCalendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .current
        return calendar
    }()

    /// Whole years elapsed between `dateOfBirth` and `now`.
    public static func age(
        dateOfBirth: Date,
        now: Date = Date(),
        calendar: Calendar = GoalCalculator.defaultCalendar
    ) -> Int {
        calendar.dateComponents([.year], from: dateOfBirth, to: now).year ?? 0
    }

    /// Mifflin-St Jeor basal metabolic rate, in kcal/day. Not rounded — callers that
    /// display this value round it themselves (see reference vector: BMR 1742.75
    /// displays as 1743, but the unrounded value feeds into `tdee`/`goalCalories`).
    public static func bmr(
        for profile: UserProfile,
        now: Date = Date(),
        calendar: Calendar = GoalCalculator.defaultCalendar
    ) -> Double {
        let years = Double(age(dateOfBirth: profile.dateOfBirth, now: now, calendar: calendar))
        let base = 10 * profile.weightKg + 6.25 * Double(profile.heightCm) - 5 * years
        switch profile.sex {
        case .male:
            return base + 5
        case .female:
            return base - 161
        }
    }

    /// Total daily energy expenditure: BMR * activity multiplier. Not rounded (see
    /// `bmr`).
    public static func tdee(
        for profile: UserProfile,
        now: Date = Date(),
        calendar: Calendar = GoalCalculator.defaultCalendar
    ) -> Double {
        bmr(for: profile, now: now, calendar: calendar) * profile.activity.multiplier
    }

    /// Daily calorie goal: TDEE adjusted by the weekly rate (7700 kcal ~= 1 kg),
    /// floored at 1200 kcal, rounded to the nearest 10.
    public static func goalCalories(
        for profile: UserProfile,
        now: Date = Date(),
        calendar: Calendar = GoalCalculator.defaultCalendar
    ) -> Int {
        let dailyAdjustment = profile.weeklyRateKg * 7700 / 7
        let adjustedCalories = tdee(for: profile, now: now, calendar: calendar) + dailyAdjustment
        return roundToNearestTen(max(1200, adjustedCalories))
    }

    /// Fixed 30/25/45% macro split of the goal calories, converted to grams via
    /// 4/9/4 kcal per gram (protein/fat/carbs).
    public static func macroTargets(forGoalCalories goalCalories: Int) -> DailyNutritionTarget {
        let kcal = Double(goalCalories)
        return DailyNutritionTarget(
            calories: kcal,
            proteinGrams: (kcal * 0.30) / 4,
            fatGrams: (kcal * 0.25) / 9,
            carbsGrams: (kcal * 0.45) / 4
        )
    }

    /// Convenience: goal calories + macro targets in one call.
    public static func dailyTarget(
        for profile: UserProfile,
        now: Date = Date(),
        calendar: Calendar = GoalCalculator.defaultCalendar
    ) -> DailyNutritionTarget {
        macroTargets(forGoalCalories: goalCalories(for: profile, now: now, calendar: calendar))
    }

    /// Classifies a weekly-rate value against the spec §2 recommended band.
    public static func weeklyRateBand(for weeklyRateKg: Double) -> WeeklyRateBand {
        if weeklyRateKg < -0.75 {
            return .aggressiveLoss
        }
        if weeklyRateKg > 0.25 {
            return .aggressiveGain
        }
        if weeklyRateKg >= -0.75 && weeklyRateKg <= -0.25 {
            return .recommended
        }
        return .neutral
    }

    /// Rounds to the nearest multiple of 10 (half rounds away from zero, matching
    /// `Double.rounded()`'s default rule).
    public static func roundToNearestTen(_ value: Double) -> Int {
        Int((value / 10).rounded()) * 10
    }
}
