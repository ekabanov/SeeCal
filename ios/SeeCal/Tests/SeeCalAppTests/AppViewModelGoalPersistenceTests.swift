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

final class AppViewModelGoalPersistenceTests: XCTestCase {
    @MainActor
    func testLoadsSavedDailyTarget() async {
        let saved = DailyNutritionTarget(calories: 2500, proteinGrams: 170, fatGrams: 80, carbsGrams: 260)
        let prefs = InMemoryUserPreferencesStore(initialDailyTarget: saved)
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [EmptyRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: prefs
        )

        await vm.loadEntries()
        XCTAssertEqual(vm.dailyTarget, saved)
    }

    @MainActor
    func testUpdateDailyTargetPersists() async throws {
        let prefs = InMemoryUserPreferencesStore()
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [EmptyRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: prefs
        )

        let target = DailyNutritionTarget(calories: 2100, proteinGrams: 150, fatGrams: 70, carbsGrams: 220)
        await vm.updateDailyTarget(target)

        let loaded = try await prefs.loadDailyTarget()
        XCTAssertEqual(loaded, target)
    }
}
