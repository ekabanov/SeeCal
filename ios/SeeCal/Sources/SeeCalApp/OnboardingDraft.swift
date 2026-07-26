import Foundation
import SeeCalDomain

/// State for the spec §3 six-step onboarding wizard (prototype `.onboard`):
/// the collected profile fields plus the step machine (`advance`/`goBack`,
/// per-step button titles, back/skip visibility).
///
/// Seed values are the spec §2 reference profile (male, 1988-03-14, 183 cm,
/// 78.4 kg, moderate, −0.5 kg/wk) — the same values the prototype's
/// `state.profile` starts from, and the profile "Skip for now" persists
/// (spec §3: "defaults from §2 reference profile minus name").
public struct OnboardingDraft: Equatable, Sendable {
    // MARK: Steps

    /// The six wizard steps, in presentation order (spec §3 items 1–6).
    public enum Step: Int, CaseIterable, Comparable, Sendable {
        case welcome
        case aboutYou
        case body
        case activity
        case weeklyTarget
        case goal

        public static func < (lhs: Step, rhs: Step) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    public static let stepCount = Step.allCases.count

    public var step: Step

    // MARK: Collected fields

    public var sex: BiologicalSex
    public var dateOfBirth: Date
    public var heightCm: Int
    public var weightKg: Double
    public var activity: ActivityLevel
    public var weeklyRateKg: Double

    public init(
        step: Step = .welcome,
        sex: BiologicalSex = .male,
        dateOfBirth: Date = OnboardingDraft.defaultDateOfBirth,
        heightCm: Int = 183,
        weightKg: Double = 78.4,
        activity: ActivityLevel = .moderate,
        weeklyRateKg: Double = UserProfile.defaultWeeklyRateKg
    ) {
        self.step = step
        self.sex = sex
        self.dateOfBirth = dateOfBirth
        self.heightCm = UserProfile.clampHeight(heightCm)
        self.weightKg = UserProfile.clampWeight(weightKg)
        self.activity = activity
        self.weeklyRateKg = UserProfile.clampWeeklyRate(weeklyRateKg)
    }

    /// Spec §2 reference dob: 1988-03-14 (UTC), the prototype's seed value.
    public static var defaultDateOfBirth: Date {
        var components = DateComponents()
        components.year = 1988
        components.month = 3
        components.day = 14
        return GoalCalculator.defaultCalendar.date(from: components) ?? Date()
    }

    // MARK: Step machine

    public var isFirstStep: Bool { step == .welcome }
    public var isLastStep: Bool { step == .goal }

    /// Back button is hidden on step 1 (prototype: `visibility:hidden`).
    public var showsBack: Bool { !isFirstStep }

    /// "Skip for now" only appears on step 1 (prototype: `display:none` after).
    public var showsSkip: Bool { isFirstStep }

    /// Prototype `renderOb()`: "Get started" on step 1, "Start tracking" on the
    /// last step, "Continue" in between.
    public var primaryButtonTitle: String {
        if isLastStep { return "Start tracking" }
        if isFirstStep { return "Get started" }
        return "Continue"
    }

    public mutating func advance() {
        guard let next = Step(rawValue: step.rawValue + 1) else { return }
        step = next
    }

    public mutating func goBack() {
        guard let previous = Step(rawValue: step.rawValue - 1) else { return }
        step = previous
    }

    // MARK: Value stepping (spec §3: ±1 cm 120–230, ±0.5 kg 35–250)

    public mutating func stepHeight(by delta: Int) {
        heightCm = UserProfile.clampHeight(heightCm + delta)
    }

    public mutating func stepWeight(by delta: Double) {
        weightKg = UserProfile.clampWeight(weightKg + delta)
    }

    public mutating func setWeeklyRate(_ value: Double) {
        // Snap to the 0.1 step so slider float noise never leaks into the profile.
        weeklyRateKg = UserProfile.clampWeeklyRate((value * 10).rounded() / 10)
    }

    // MARK: Output

    /// The profile this draft persists on finish. All fields are kept clamped as
    /// they're edited, so this cannot fail.
    public func userProfile() -> UserProfile {
        UserProfile(
            sex: sex,
            dateOfBirth: dateOfBirth,
            heightCm: heightCm,
            weightKg: weightKg,
            activity: activity,
            weeklyRateKg: weeklyRateKg
        )
    }

    /// "Skip for now" (step 1) persists the spec §2 reference defaults untouched.
    public static func skipDefaults() -> UserProfile {
        OnboardingDraft().userProfile()
    }
}
