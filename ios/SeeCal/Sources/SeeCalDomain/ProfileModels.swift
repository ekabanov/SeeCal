import Foundation

public enum BiologicalSex: String, Codable, CaseIterable, Sendable {
    case male
    case female
}

public enum ActivityLevel: String, Codable, CaseIterable, Sendable {
    case sedentary
    case light
    case moderate
    case active

    /// TDEE multiplier applied to BMR (spec §2).
    public var multiplier: Double {
        switch self {
        case .sedentary:
            return 1.2
        case .light:
            return 1.375
        case .moderate:
            return 1.55
        case .active:
            return 1.725
        }
    }
}

/// Classification of a weekly-rate value against the spec §2 recommended band
/// (−0.75 … −0.25 kg/week). UI copy for each case is owned by the UI layer (P5);
/// this only classifies the number.
public enum WeeklyRateBand: Equatable, Sendable {
    /// Inside the recommended band, −0.75 … −0.25 kg/week (inclusive).
    case recommended
    /// Outside the recommended band, but not aggressive enough to warn about
    /// (e.g. maintain, or a slow gain up to +0.25 kg/week).
    case neutral
    /// Faster than 0.75 kg/week loss — "hard to sustain and not recommended".
    case aggressiveLoss
    /// Faster than 0.25 kg/week gain — "mostly adds fat".
    case aggressiveGain
}

public struct UserProfile: Codable, Equatable, Sendable {
    /// Spec §2: weeklyRateKg is clamped to this range, step 0.1, default −0.5.
    public static let weeklyRateRange: ClosedRange<Double> = -1.0...0.5
    public static let defaultWeeklyRateKg: Double = -0.5
    /// Spec §3 step 3: height stepper ±1 cm within 120–230.
    public static let heightRangeCm: ClosedRange<Int> = 120...230
    /// Spec §3 step 3: weight stepper ±0.5 kg within 35–250.
    public static let weightRangeKg: ClosedRange<Double> = 35...250

    public var sex: BiologicalSex
    public var dateOfBirth: Date
    public var heightCm: Int
    public var weightKg: Double
    public var activity: ActivityLevel
    public var weeklyRateKg: Double

    public init(
        sex: BiologicalSex,
        dateOfBirth: Date,
        heightCm: Int,
        weightKg: Double,
        activity: ActivityLevel,
        weeklyRateKg: Double = UserProfile.defaultWeeklyRateKg
    ) {
        self.sex = sex
        self.dateOfBirth = dateOfBirth
        self.heightCm = heightCm
        self.weightKg = weightKg
        self.activity = activity
        self.weeklyRateKg = UserProfile.clampWeeklyRate(weeklyRateKg)
    }

    public static func clampWeeklyRate(_ value: Double) -> Double {
        min(max(value, weeklyRateRange.lowerBound), weeklyRateRange.upperBound)
    }

    public static func clampHeight(_ value: Int) -> Int {
        min(max(value, heightRangeCm.lowerBound), heightRangeCm.upperBound)
    }

    public static func clampWeight(_ value: Double) -> Double {
        min(max(value, weightRangeKg.lowerBound), weightRangeKg.upperBound)
    }

    // MARK: - Migration-aware Codable
    //
    // P1 evolved this type's shape (biologicalSex/ageYears/heightCm(Double)/
    // activityLevel/goalPace -> sex/dateOfBirth/heightCm(Int)/activity/weeklyRateKg).
    // Decoding tries the current shape first; if the current-shape keys aren't
    // present (an old persisted file), it falls back to the legacy shape and
    // synthesizes defaults for fields that have no legacy equivalent, per the P1
    // migration contract ("missing new fields -> spec defaults").

    private enum CodingKeys: String, CodingKey {
        case sex, dateOfBirth, heightCm, weightKg, activity, weeklyRateKg
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case biologicalSex, ageYears, heightCm, weightKg, activityLevel, goalPace
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let sex = try? container.decode(BiologicalSex.self, forKey: .sex),
           let dateOfBirth = try? container.decode(Date.self, forKey: .dateOfBirth) {
            self.sex = sex
            self.dateOfBirth = dateOfBirth
            self.heightCm = try container.decode(Int.self, forKey: .heightCm)
            self.weightKg = try container.decode(Double.self, forKey: .weightKg)
            self.activity = try container.decode(ActivityLevel.self, forKey: .activity)
            let weeklyRateKg = (try? container.decode(Double.self, forKey: .weeklyRateKg))
                ?? UserProfile.defaultWeeklyRateKg
            self.weeklyRateKg = UserProfile.clampWeeklyRate(weeklyRateKg)
            return
        }

        // Legacy (pre-P1) shape.
        let legacy = try decoder.container(keyedBy: LegacyCodingKeys.self)
        self.sex = try legacy.decode(BiologicalSex.self, forKey: .biologicalSex)

        let legacyAgeYears = (try? legacy.decode(Int.self, forKey: .ageYears)) ?? 30
        self.dateOfBirth = UserProfile.migrateDateOfBirth(fromAgeYears: legacyAgeYears)

        if let heightInt = try? legacy.decode(Int.self, forKey: .heightCm) {
            self.heightCm = heightInt
        } else {
            let legacyHeight = (try? legacy.decode(Double.self, forKey: .heightCm)) ?? 175
            self.heightCm = Int(legacyHeight.rounded())
        }

        self.weightKg = (try? legacy.decode(Double.self, forKey: .weightKg)) ?? 75

        if let legacyActivityRaw = try? legacy.decode(String.self, forKey: .activityLevel) {
            self.activity = UserProfile.migrateActivity(legacyActivityRaw)
        } else {
            self.activity = .moderate
        }

        if let legacyGoalPaceRaw = try? legacy.decode(String.self, forKey: .goalPace) {
            self.weeklyRateKg = UserProfile.clampWeeklyRate(UserProfile.migrateGoalPace(legacyGoalPaceRaw))
        } else {
            self.weeklyRateKg = UserProfile.defaultWeeklyRateKg
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sex, forKey: .sex)
        try container.encode(dateOfBirth, forKey: .dateOfBirth)
        try container.encode(heightCm, forKey: .heightCm)
        try container.encode(weightKg, forKey: .weightKg)
        try container.encode(activity, forKey: .activity)
        try container.encode(weeklyRateKg, forKey: .weeklyRateKg)
    }

    private static func migrateDateOfBirth(fromAgeYears ageYears: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .current
        return calendar.date(byAdding: .year, value: -ageYears, to: Date()) ?? Date()
    }

    private static func migrateActivity(_ legacyRawValue: String) -> ActivityLevel {
        switch legacyRawValue {
        case "sedentary":
            return .sedentary
        case "lightlyActive":
            return .light
        case "moderatelyActive":
            return .moderate
        case "veryActive":
            return .active
        default:
            return .moderate
        }
    }

    private static func migrateGoalPace(_ legacyRawValue: String) -> Double {
        switch legacyRawValue {
        case "loseSlow":
            return -0.25
        case "loseModerate":
            return -0.5
        case "maintain":
            return 0.0
        case "gainSlow":
            return 0.25
        default:
            return UserProfile.defaultWeeklyRateKg
        }
    }
}
