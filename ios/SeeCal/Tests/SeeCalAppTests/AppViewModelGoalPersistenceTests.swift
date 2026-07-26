import XCTest
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
@testable import SeeCalApp

private struct EmptyRuntime: InferenceRuntime {
    let name = "empty"
    let modelFamily = "qwen3.5-native-multimodal"
    func isAvailable() async -> Bool { false }
    func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        _ = request
        throw InferenceError.runtimeUnavailable("none")
    }
}

/// AppViewModel exposes the daily goal as a computed value derived from
/// `userProfile` via `GoalCalculator` (spec §2) — `GoalEditDraft` and manual goal
/// editing were retired in P1. These tests cover that computed-goal behavior,
/// plus the migration-reading fallback to a legacy persisted target when no
/// profile has been saved yet.
final class AppViewModelGoalPersistenceTests: XCTestCase {
    private static let fixedNow = date(year: 2026, month: 7, day: 26)

    private static func date(year: Int, month: Int, day: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: DateComponents(year: year, month: month, day: day))!
    }

    private static let referenceProfile = UserProfile(
        sex: .male,
        dateOfBirth: date(year: 1988, month: 3, day: 14),
        heightCm: 183,
        weightKg: 78.4,
        activity: .moderate,
        weeklyRateKg: -0.5
    )

    @MainActor
    func testLoadsComputedGoalFromSavedProfile() async {
        let prefs = InMemoryUserPreferencesStore(initialUserProfile: Self.referenceProfile)
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [EmptyRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: prefs,
            now: { Self.fixedNow }
        )

        await vm.loadEntries()

        // Matches the spec §2 reference vector: goal 2150.
        XCTAssertEqual(vm.dailyTarget.calories, 2150)
        XCTAssertEqual(
            vm.dailyTarget,
            GoalCalculator.dailyTarget(for: Self.referenceProfile, now: Self.fixedNow)
        )
    }

    @MainActor
    func testFallsBackToLegacyPersistedTargetWhenNoProfileExists() async {
        // Migration-reading only: a pre-P1 install may have persisted a
        // manually-edited goal (via the now-retired GoalEditDraft) before any
        // profile existed. Once a profile is present this path never runs again.
        let saved = DailyNutritionTarget(calories: 2500, proteinGrams: 170, fatGrams: 80, carbsGrams: 260)
        let prefs = InMemoryUserPreferencesStore(initialDailyTarget: saved)
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [EmptyRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: prefs
        )

        await vm.loadEntries()
        XCTAssertEqual(vm.dailyTarget, saved)
        XCTAssertNil(vm.userProfile)
    }

    @MainActor
    func testSavedProfileTakesPrecedenceOverLegacyPersistedTarget() async {
        let legacyTarget = DailyNutritionTarget(calories: 2500, proteinGrams: 170, fatGrams: 80, carbsGrams: 260)
        let prefs = InMemoryUserPreferencesStore(
            initialDailyTarget: legacyTarget,
            initialUserProfile: Self.referenceProfile
        )
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [EmptyRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: prefs,
            now: { Self.fixedNow }
        )

        await vm.loadEntries()

        XCTAssertEqual(vm.dailyTarget.calories, 2150)
        XCTAssertNotEqual(vm.dailyTarget, legacyTarget)
    }

    @MainActor
    func testCompleteOnboardingComputesGoalFromProfileAndPersistsOnlyTheProfile() async throws {
        let prefs = InMemoryUserPreferencesStore()
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [EmptyRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: prefs,
            now: { Self.fixedNow }
        )

        await vm.completeOnboarding(with: Self.referenceProfile)

        XCTAssertEqual(vm.dailyTarget.calories, 2150)

        let persistedProfile = try await prefs.loadUserProfile()
        XCTAssertEqual(persistedProfile, Self.referenceProfile)

        // The goal itself is never written as an independent value anymore — only
        // the profile is persisted, and the goal is recomputed from it on load.
        let persistedTarget = try await prefs.loadDailyTarget()
        XCTAssertNil(persistedTarget)
    }

    @MainActor
    func testInlineProfileEditRecomputesGoalImmediatelyAndPersists() async throws {
        // Spec §7: every Profile-screen change writes through to the persisted
        // profile and recomputes the goal immediately (Today's ring reads
        // `dailyTarget`, so it reflects the edit on the next render).
        let prefs = InMemoryUserPreferencesStore(initialUserProfile: Self.referenceProfile)
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [EmptyRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: prefs,
            now: { Self.fixedNow }
        )
        await vm.loadEntries()
        XCTAssertEqual(vm.dailyTarget.calories, 2150)

        // Inline edit: weekly target −0.5 → 0 (maintain).
        var edited = Self.referenceProfile
        edited.weeklyRateKg = 0
        await vm.updateProfile(edited)

        XCTAssertEqual(vm.userProfile, edited)
        XCTAssertEqual(vm.dailyTarget.calories, 2700)
        XCTAssertEqual(
            vm.dailyTarget,
            GoalCalculator.dailyTarget(for: edited, now: Self.fixedNow)
        )

        let persisted = try await prefs.loadUserProfile()
        XCTAssertEqual(persisted, edited)
    }
}
