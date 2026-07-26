import Foundation
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence

@MainActor
public final class AppViewModel: ObservableObject {
    @Published public private(set) var mealEntries: [MealLogEntry] = []
    @Published public private(set) var weightEntries: [WeightEntry] = []
    @Published public private(set) var userProfile: UserProfile?
    /// The daily calorie/macro goal. Always derived from `userProfile` via
    /// `GoalCalculator` once a profile exists (spec §2) — it is never an
    /// independently user-editable value (see `GoalEditDraft`'s retirement, P1).
    /// Before a profile is loaded/set, this falls back to the constructor-supplied
    /// default, or to a legacy persisted target (see `loadEntries`).
    @Published public private(set) var dailyTarget: DailyNutritionTarget
    @Published public private(set) var isScanning = false
    @Published public private(set) var lastInferenceSeconds: Double?
    @Published public var lastError: String?

    private let orchestrator: RuntimeOrchestrator
    private let store: MealLogStore
    private let preferencesStore: UserPreferencesStore
    private let weightStore: WeightLogStore
    private let now: @Sendable () -> Date

    public var consumedToday: NutritionTotals {
        let todaysEntries = mealEntries.filter { Calendar.current.isDateInToday($0.createdAt) }
        return todaysEntries.reduce(NutritionTotals()) { $0 + $1.totals }
    }

    public var remainingToday: NutritionRemaining {
        NutritionTracker.remaining(target: dailyTarget, consumed: consumedToday)
    }

    public var weeklyProgress: [DailyProgressPoint] {
        ProgressAggregator.dailyPoints(
            from: mealEntries.map { AnyMealLogEntry(createdAt: $0.createdAt, totals: $0.totals) },
            days: 7
        )
    }

    public var weeklyWeightTrend: [WeeklyWeightPoint] {
        WeightTrend.weeklyAveragePoints(
            weights: weightEntries.map { AnyWeightEntry(date: $0.date, weightKg: $0.weightKg) },
            weeks: 8
        )
    }

    public var requiresOnboarding: Bool {
        userProfile == nil
    }

    /// Earliest logged meal's date — drives the Profile header's
    /// "Logging since <month year>" line (spec §7). `nil` before the first log.
    public var loggingSinceDate: Date? {
        mealEntries.map(\.createdAt).min()
    }

    /// Net weight change over the trailing 30 days, for the Profile weight row's
    /// trend note ("−1.2 kg this month"). `nil` when the weight log has fewer
    /// than two entries in the window.
    public var monthWeightChangeKg: Double? {
        WeightTrend.monthChangeKg(
            weights: weightEntries.map { AnyWeightEntry(date: $0.date, weightKg: $0.weightKg) },
            now: now()
        )
    }

    public init(
        orchestrator: RuntimeOrchestrator,
        store: MealLogStore,
        preferencesStore: UserPreferencesStore = InMemoryUserPreferencesStore(),
        weightStore: WeightLogStore = InMemoryWeightLogStore(),
        dailyTarget: DailyNutritionTarget = .defaultTarget,
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.orchestrator = orchestrator
        self.store = store
        self.preferencesStore = preferencesStore
        self.weightStore = weightStore
        self.dailyTarget = dailyTarget
        self.now = now
    }

    public func loadEntries() async {
        do {
            userProfile = try await preferencesStore.loadUserProfile()
            if let userProfile {
                // The goal is always computed from the profile once one exists.
                dailyTarget = GoalCalculator.dailyTarget(for: userProfile, now: now())
            } else if let savedTarget = try await preferencesStore.loadDailyTarget() {
                // Migration-reading only: a legacy install may have persisted a
                // manually-edited goal from before the goal became profile-derived
                // (retired `GoalEditDraft`, P1). Once a profile is saved this branch
                // never runs again.
                dailyTarget = savedTarget
            }
            mealEntries = try await store.fetchAll()
            weightEntries = try await weightStore.fetchAll()
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func completeOnboarding(with profile: UserProfile) async {
        await updateProfile(profile)
    }

    /// Single write-through path for every profile change — onboarding finish,
    /// onboarding skip, and each inline edit on the Profile screen (spec §7:
    /// "Every change recomputes the goal immediately"). Persists the profile and
    /// recomputes `dailyTarget` in the same step, so Today's ring reflects the
    /// edit as soon as the published values change.
    public func updateProfile(_ profile: UserProfile) async {
        do {
            let target = GoalCalculator.dailyTarget(for: profile, now: now())
            userProfile = profile
            dailyTarget = target
            try await preferencesStore.saveUserProfile(profile)
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func addMealPhoto(imagePath: String, mealType: MealType, userHint: String?) async {
        guard !isScanning else {
            print("[SeeCal][AppViewModel] addMealPhoto ignored: scan already in progress")
            return
        }
        do {
            isScanning = true
            let started = Date()
            print("[SeeCal][AppViewModel] addMealPhoto start mealType=\(mealType.rawValue) imagePath=\(imagePath)")
            let request = FoodScanRequest(imagePath: imagePath, mealType: mealType, userHint: userHint)
            let result = try await orchestrator.infer(request: request)
            let items = result.items.map(MealItem.init(scanItem:))
            let entry = MealLogEntry(mealType: mealType, imagePath: imagePath, items: items)
            try await store.save(entry)
            mealEntries = try await store.fetchAll()
            lastInferenceSeconds = Date().timeIntervalSince(started)
            isScanning = false
            print("[SeeCal][AppViewModel] addMealPhoto success in \(String(format: "%.2f", lastInferenceSeconds ?? 0))s")
        } catch {
            isScanning = false
            lastError = error.localizedDescription
            print("[SeeCal][AppViewModel] addMealPhoto failed error=\(error.localizedDescription)")
        }
    }

    /// Persists an edited entry (e.g. from `MealEditDraft.committedEntry()` in edit
    /// mode). The entry replaces the stored one by `id` — see `MealLogStore.update`.
    public func updateMeal(_ entry: MealLogEntry) async {
        do {
            try await store.update(entry)
            mealEntries = try await store.fetchAll()
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func deleteMeal(id: UUID) async {
        do {
            try await store.delete(id: id)
            mealEntries = try await store.fetchAll()
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func addWeightEntry(kg: Double, date: Date = Date()) async {
        do {
            guard kg > 0 else {
                throw NSError(domain: "AppViewModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Weight must be positive"])
            }
            try await weightStore.save(WeightEntry(date: date, weightKg: kg))
            weightEntries = try await weightStore.fetchAll()
        } catch {
            lastError = error.localizedDescription
        }
    }
}

// MARK: - Today display formatting (spec §4)

/// Pure formatting helpers shared by `TodayScreen` (and, later, other screens
/// reading the same totals/targets). Kept as static functions on `AppViewModel`
/// rather than embedded in the view so they're directly unit-testable without a
/// live view hierarchy.
public extension AppViewModel {
    /// `docs/design/prototype/seecal-prototype.html:1579`:
    /// `d.toLocaleDateString("en-US", {weekday:"long", month:"long", day:"numeric"})`
    /// e.g. "Sunday, July 26" (full weekday, full month, numeric day, no year).
    /// `PageHeader` upper-cases this at render time to match spec §4's
    /// "SUNDAY, JULY 26". Locale/calendar are pinned to en-US/Gregorian so the
    /// format is deterministic regardless of device locale, matching the
    /// prototype's explicit `"en-US"` argument.
    nonisolated static func dateSubtitle(for date: Date, calendar: Calendar = GoalCalculator.defaultCalendar) -> String {
        var fixedCalendar = calendar
        fixedCalendar.locale = Locale(identifier: "en_US_POSIX")
        let formatter = DateFormatter()
        formatter.calendar = fixedCalendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "EEEE, MMMM d"
        return formatter.string(from: date)
    }

    /// Ratio of `consumed`/`target`, clamped to `[0, 1]`. Shared by the calorie
    /// ring's arc and each macro bar's fill width (spec §4: "ring fraction clamp
    /// at >100%" / "fill fraction clamped to 1"). Guards `target <= 0` to avoid
    /// divide-by-zero before a profile/goal is loaded.
    nonisolated static func progressFraction(consumed: Double, target: Double) -> Double {
        guard target > 0 else { return 0 }
        return min(1, max(0, consumed / target))
    }

    /// Nearest-integer gram display (matches the prototype's `Math.round(...)`
    /// for macro values, e.g. `161`/`60`/`242` for the spec §2 reference goal).
    nonisolated static func roundedGrams(_ value: Double) -> Int {
        Int(value.rounded())
    }

    /// Thousands-grouped whole-number kcal display (matches the prototype's
    /// `fmt = n => n.toLocaleString("en-US")`, e.g. "2,150"). Used for the ring's
    /// center number, its "of N kcal" caption, and each meal row's kcal figure.
    nonisolated static func formattedKcal(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US")
        formatter.numberStyle = .decimal
        formatter.usesGroupingSeparator = true
        formatter.maximumFractionDigits = 0
        let rounded = value.rounded()
        return formatter.string(from: NSNumber(value: rounded)) ?? String(Int(rounded))
    }
}

// MARK: - Onboarding + Profile display formatting (spec §3 step 6, §7)

public extension AppViewModel {
    /// The transparent goal-math line, byte-matching the prototype's
    /// `renderProfile()` math string (with the typographic minus, matching the
    /// on-screen copy), e.g. for the spec §2 reference vector:
    /// "BMR 1,743 × 1.55 (moderate) = 2,701 kcal burned − 550 kcal (−0.5 kg/wk) = 2,150 kcal".
    /// At maintain (rate 0) the adjustment segment is omitted entirely, exactly
    /// as in the prototype: "… = 2,701 kcal burned = 2,700 kcal".
    nonisolated static func goalMathString(
        for profile: UserProfile,
        now: Date = Date(),
        calendar: Calendar = GoalCalculator.defaultCalendar
    ) -> String {
        let bmr = GoalCalculator.bmr(for: profile, now: now, calendar: calendar)
        let tdee = GoalCalculator.tdee(for: profile, now: now, calendar: calendar)
        let goal = GoalCalculator.goalCalories(for: profile, now: now, calendar: calendar)
        let dailyAdjustment = Int((profile.weeklyRateKg * 7700 / 7).rounded())

        var text = "BMR \(formattedKcal(bmr.rounded()))"
            + " × \(activityFactorString(profile.activity)) (\(profile.activity.rawValue))"
            + " = \(formattedKcal(tdee.rounded())) kcal burned"
        if dailyAdjustment != 0 {
            let sign = dailyAdjustment < 0 ? "−" : "+"
            let rateSign = profile.weeklyRateKg > 0 ? "+" : "−"
            let rate = rateSign + String(format: "%.1f", abs(profile.weeklyRateKg))
            text += " \(sign) \(formattedKcal(Double(abs(dailyAdjustment)))) kcal (\(rate) kg/wk)"
        }
        text += " = \(formattedKcal(Double(goal))) kcal"
        return text
    }

    /// TDEE multiplier rendered the way the prototype's JS renders the raw
    /// number ("1.2", "1.375", "1.55", "1.725") — fixed strings, no float
    /// formatting surprises.
    nonisolated static func activityFactorString(_ activity: ActivityLevel) -> String {
        switch activity {
        case .sedentary: return "1.2"
        case .light: return "1.375"
        case .moderate: return "1.55"
        case .active: return "1.725"
        }
    }

    /// "March 2026" — the Profile header's "Logging since <month year>" value.
    nonisolated static func loggingSinceString(for date: Date, calendar: Calendar = GoalCalculator.defaultCalendar) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "LLLL yyyy"
        return formatter.string(from: date)
    }

    /// "−1.2 kg this month" / "+0.4 kg this month" (typographic minus), or `nil`
    /// when there is no meaningful change to report (< 0.05 kg or no data).
    nonisolated static func monthTrendString(deltaKg: Double?) -> String? {
        guard let deltaKg, abs(deltaKg) >= 0.05 else { return nil }
        let sign = deltaKg < 0 ? "−" : "+"
        return sign + String(format: "%.1f", abs(deltaKg)) + " kg this month"
    }
}
