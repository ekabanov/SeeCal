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
        return NutritionTracker.totals(from: todaysEntries.map(\.scanResult))
    }

    public var remainingToday: NutritionRemaining {
        NutritionTracker.remaining(target: dailyTarget, consumed: consumedToday)
    }

    public var weeklyProgress: [DailyProgressPoint] {
        ProgressAggregator.dailyPoints(
            from: mealEntries.map { AnyMealLogEntry(createdAt: $0.createdAt, scanResult: $0.scanResult) },
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
            let entry = MealLogEntry(mealType: mealType, imagePath: imagePath, scanResult: result)
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

    public func updateMeal(_ entry: MealLogEntry, with result: FoodScanResult) async {
        do {
            var updated = entry
            updated.scanResult = try result.validated()
            try await store.update(updated)
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
