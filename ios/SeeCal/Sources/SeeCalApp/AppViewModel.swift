import Foundation
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence

@MainActor
public final class AppViewModel: ObservableObject {
    @Published public private(set) var mealEntries: [MealLogEntry] = []
    @Published public private(set) var weightEntries: [WeightEntry] = []
    @Published public private(set) var userProfile: UserProfile?
    @Published public var dailyTarget: DailyNutritionTarget
    @Published public private(set) var isScanning = false
    @Published public private(set) var lastInferenceSeconds: Double?
    @Published public var lastError: String?

    private let orchestrator: RuntimeOrchestrator
    private let store: MealLogStore
    private let preferencesStore: UserPreferencesStore
    private let weightStore: WeightLogStore

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
        dailyTarget: DailyNutritionTarget = .defaultTarget
    ) {
        self.orchestrator = orchestrator
        self.store = store
        self.preferencesStore = preferencesStore
        self.weightStore = weightStore
        self.dailyTarget = dailyTarget
    }

    public func loadEntries() async {
        do {
            if let savedTarget = try await preferencesStore.loadDailyTarget() {
                dailyTarget = savedTarget
            }
            userProfile = try await preferencesStore.loadUserProfile()
            mealEntries = try await store.fetchAll()
            weightEntries = try await weightStore.fetchAll()
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func updateDailyTarget(_ target: DailyNutritionTarget) async {
        do {
            dailyTarget = target
            try await preferencesStore.saveDailyTarget(target)
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func completeOnboarding(with profile: UserProfile) async {
        do {
            let target = DailyNutritionTargetPlanner.deriveTarget(from: profile)
            userProfile = profile
            dailyTarget = target
            try await preferencesStore.saveUserProfile(profile)
            try await preferencesStore.saveDailyTarget(target)
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
