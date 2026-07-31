import Foundation
import SeeCalDiagnostics
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
    @Published public var lastError: String?
    /// Settings §8 capture coaching toggle. Defaults on
    /// (`CapturePreferences.default`) until a persisted value loads
    /// in `loadEntries()`, matching the prototype's initial switch state.
    @Published public private(set) var capturePreferences: CapturePreferences = .default

    /// Settings §8 "On-device model" card contents — resolved once at
    /// construction from whatever `modelPath`/`adapterPath` the caller passes
    /// (production: `ModelAssetResolver`'s resolved paths via `QwenRuntimeConfig`;
    /// simulator/tests that don't pass a config: `ModelInfo.notBundled`). Fixed
    /// for the process lifetime — the bundled model/adapter can't change
    /// without a relaunch.
    public let modelInfo: ModelInfo
    /// Tracks the one-time lazy model preparation so the analyzing screen can
    /// explain why the first scan takes longer. Injected/mock runtimes are
    /// considered ready by default.
    public let modelPreparationState: ModelPreparationState
    /// Optional local database used by the correction-first replacement tray.
    /// The review flow remains usable through Edit details when unavailable.
    public let nutritionCandidateProvider: (any NutritionCandidateProviding)?

    private let orchestrator: RuntimeOrchestrator
    private let store: MealLogStore

    /// Seam for the scan flow (P6): `ScanFlowController` runs inference itself —
    /// *without* persisting — so a result can be reviewed/adjusted in the result
    /// sheet before an explicit "Log meal" (spec §5: "NOTHING is persisted without
    /// explicit confirm"). Exposes the production orchestrator behind the
    /// mockable `ScanInferenceRunning` protocol.
    public var scanInferenceRunner: ScanInferenceRunning { orchestrator }
    private let preferencesStore: UserPreferencesStore
    private let weightStore: WeightLogStore
    private let now: @Sendable () -> Date

    /// Today's entries, where "today" comes from the injectable clock (like
    /// every other date-derived value here) so tests can pin it —
    /// Calendar.current still supplies the user's day boundaries. Shared by
    /// `consumedToday` and the Today screen's meals card.
    public var todaysMealEntries: [MealLogEntry] {
        let today = now()
        return mealEntries.filter { Calendar.current.isDate($0.createdAt, inSameDayAs: today) }
    }

    public var consumedToday: NutritionTotals {
        todaysMealEntries.reduce(NutritionTotals()) { $0 + $1.totals }
    }

    public var remainingToday: NutritionRemaining {
        NutritionTracker.remaining(target: dailyTarget, consumed: consumedToday)
    }

    /// History screen chart dataset for one range (spec §6) — see
    /// `ProgressAggregator.historyChartData` for the aggregation rules
    /// (thresholds, weekly averages, axis-zone label density).
    public func historyChartData(for range: HistoryRange) -> HistoryChartData {
        ProgressAggregator.historyChartData(
            from: mealEntries.map { AnyMealLogEntry(createdAt: $0.createdAt, totals: $0.totals) },
            goalCalories: dailyTarget.calories,
            range: range,
            referenceDate: now()
        )
    }

    /// History screen's "recent meals" card (spec §6): previous days' entries
    /// (today's own meals stay on the Today screen only), most recent first,
    /// capped at 10.
    public var recentMealEntries: [MealLogEntry] {
        let today = now()
        return Array(
            mealEntries
                .filter { !Calendar.current.isDate($0.createdAt, inSameDayAs: today) }
                .sorted { $0.createdAt > $1.createdAt }
                .prefix(10)
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
        modelPath: String? = nil,
        adapterPath: String? = nil,
        modelPreparationState: ModelPreparationState = ModelPreparationState(),
        nutritionCandidateProvider: (any NutritionCandidateProviding)? = nil,
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.orchestrator = orchestrator
        self.store = store
        self.preferencesStore = preferencesStore
        self.weightStore = weightStore
        self.dailyTarget = dailyTarget
        self.modelInfo = ModelInfoResolver.resolve(modelPath: modelPath, adapterPath: adapterPath)
        self.modelPreparationState = modelPreparationState
        self.nutritionCandidateProvider = nutritionCandidateProvider
        self.now = now
        SeeCalDiagnostics.record(
            .info,
            category: "app_state",
            name: "view_model_created",
            fields: [
                "model": self.modelInfo.modelLabel,
                "adapter": self.modelInfo.adapterVersionLabel ?? "none",
                "quantization": self.modelInfo.quantizationLabel ?? "unknown"
            ]
        )
    }

    public func loadEntries() async {
        SeeCalDiagnostics.record(.info, category: "app_state", name: "initial_data_load_started")
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
            if let savedCapturePreferences = try await preferencesStore.loadCapturePreferences() {
                capturePreferences = savedCapturePreferences
            }
            mealEntries = try await store.fetchAll()
            weightEntries = try await weightStore.fetchAll()
            SeeCalDiagnostics.record(
                .notice,
                category: "app_state",
                name: "initial_data_load_succeeded",
                fields: [
                    "has_profile": String(userProfile != nil),
                    "meal_count": String(mealEntries.count),
                    "weight_entry_count": String(weightEntries.count)
                ]
            )
        } catch {
            lastError = error.localizedDescription
            SeeCalDiagnostics.record(
                .error,
                category: "app_state",
                name: "initial_data_load_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
        }
    }

    /// Settings §8 Capture toggles — write-through, so a redisplayed Settings
    /// screen and the camera's coaching overlays stay in sync immediately.
    public func updateCapturePreferences(_ preferences: CapturePreferences) async {
        do {
            capturePreferences = preferences
            try await preferencesStore.saveCapturePreferences(preferences)
            SeeCalDiagnostics.record(
                .info,
                category: "app_state",
                name: "capture_preferences_saved",
                fields: ["coaching_enabled": String(preferences.captureCoachingEnabled)]
            )
        } catch {
            lastError = error.localizedDescription
            SeeCalDiagnostics.record(
                .error,
                category: "app_state",
                name: "capture_preferences_save_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
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
    ///
    /// A weight change additionally appends a `WeightEntry` dated now — the
    /// Profile weight stepper is the app's only weight input, so this is what
    /// feeds the spec §7 weight-trend note ("−1.2 kg this month") and the
    /// weekly weight trend. The very first profile save seeds the log's
    /// baseline entry.
    public func updateProfile(_ profile: UserProfile) async {
        do {
            let weightChanged = userProfile?.weightKg != profile.weightKg
            let target = GoalCalculator.dailyTarget(for: profile, now: now())
            userProfile = profile
            dailyTarget = target
            try await preferencesStore.saveUserProfile(profile)
            if weightChanged {
                try await weightStore.save(WeightEntry(date: now(), weightKg: profile.weightKg))
                weightEntries = try await weightStore.fetchAll()
            }
            SeeCalDiagnostics.record(
                .info,
                category: "app_state",
                name: "profile_saved",
                fields: ["weight_changed": String(weightChanged)]
            )
        } catch {
            lastError = error.localizedDescription
            SeeCalDiagnostics.record(
                .error,
                category: "app_state",
                name: "profile_save_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
        }
    }

    /// Persists a brand-new, already-inferred entry — the scan flow's explicit
    /// "Log meal" confirmation (spec §5). This is pure persistence: inference
    /// already happened in `ScanFlowController`, and the user has
    /// reviewed/adjusted the draft.
    public func logMeal(_ entry: MealLogEntry) async {
        do {
            try await store.save(entry)
            mealEntries = try await store.fetchAll()
            SeeCalDiagnostics.record(
                .notice,
                category: "app_state",
                name: "meal_saved",
                fields: ["meal_count": String(mealEntries.count)]
            )
        } catch {
            lastError = error.localizedDescription
            SeeCalDiagnostics.record(
                .error,
                category: "app_state",
                name: "meal_save_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
        }
    }

    /// Persists an edited entry (e.g. from `MealEditDraft.committedEntry()` in edit
    /// mode). The entry replaces the stored one by `id` — see `MealLogStore.update`.
    public func updateMeal(_ entry: MealLogEntry) async {
        do {
            try await store.update(entry)
            mealEntries = try await store.fetchAll()
            SeeCalDiagnostics.record(
                .notice,
                category: "app_state",
                name: "meal_updated",
                fields: ["meal_count": String(mealEntries.count)]
            )
        } catch {
            lastError = error.localizedDescription
            SeeCalDiagnostics.record(
                .error,
                category: "app_state",
                name: "meal_update_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
        }
    }

    public func deleteMeal(id: UUID) async {
        do {
            // Capture the photo path BEFORE the store delete refreshes the list.
            let imagePath = mealEntries.first(where: { $0.id == id })?.imagePath
            try await store.delete(id: id)
            mealEntries = try await store.fetchAll()
            // The entry is gone, so its captured photo has no remaining reader
            // (thumbnails only load for live entries). Best-effort cleanup.
            if let imagePath {
                try? FileManager.default.removeItem(atPath: imagePath)
            }
            SeeCalDiagnostics.record(
                .notice,
                category: "app_state",
                name: "meal_deleted",
                fields: ["meal_count": String(mealEntries.count), "photo_cleanup_attempted": String(imagePath != nil)]
            )
        } catch {
            lastError = error.localizedDescription
            SeeCalDiagnostics.record(
                .error,
                category: "app_state",
                name: "meal_delete_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
        }
    }

    public func addWeightEntry(kg: Double, date: Date = Date()) async {
        do {
            guard kg > 0 else {
                throw NSError(domain: "AppViewModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Weight must be positive"])
            }
            try await weightStore.save(WeightEntry(date: date, weightKg: kg))
            weightEntries = try await weightStore.fetchAll()
            SeeCalDiagnostics.record(
                .info,
                category: "app_state",
                name: "weight_entry_saved",
                fields: ["weight_entry_count": String(weightEntries.count)]
            )
        } catch {
            lastError = error.localizedDescription
            SeeCalDiagnostics.record(
                .error,
                category: "app_state",
                name: "weight_entry_save_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
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

// MARK: - Settings display formatting (spec §8)

public extension AppViewModel {
    /// "Qwen3.5-4B · SeeCal adapter v6" when an adapter version was resolved;
    /// "Qwen3.5-4B · base model (no adapter bundled)" when none was (no
    /// adapter configured, or one is configured but its version can't be
    /// determined) — never a fabricated version number (spec §8).
    nonisolated static func modelCardTitle(info: ModelInfo) -> String {
        if let version = info.adapterVersionLabel {
            return "\(info.modelLabel) · SeeCal adapter \(version)"
        }
        return "\(info.modelLabel) · base model (no adapter bundled)"
    }

    /// Prototype copy for the parts that are fixed dataset/product facts
    /// (fine-tune provenance, privacy line); only the quantization figure is
    /// dynamic, read from the bundled model's own config
    /// (`ModelInfoResolver`) — falls back to "Quantization not bundled"
    /// rather than a hardcoded bit count when unavailable.
    nonisolated static func modelCardSubtitle(info: ModelInfo) -> String {
        let quantization = info.quantizationLabel ?? "Quantization not bundled"
        return "\(quantization) · fine-tuned on 5,000 measured meals.\nPhotos are processed locally and never uploaded."
    }
}
