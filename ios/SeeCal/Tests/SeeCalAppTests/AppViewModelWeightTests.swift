import XCTest
import SeeCalInference
import SeeCalPersistence
import SeeCalDomain
@testable import SeeCalApp

private struct NoopRuntime: InferenceRuntime {
    let name = "noop"
    let modelFamily = "qwen3.5-native-multimodal"
    func isAvailable() async -> Bool { false }
    func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        _ = request
        throw InferenceError.runtimeUnavailable("none")
    }
}

final class AppViewModelWeightTests: XCTestCase {
    @MainActor
    func testAddWeightEntry() async {
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: InMemoryUserPreferencesStore(),
            weightStore: InMemoryWeightLogStore()
        )

        await vm.addWeightEntry(kg: 79.2)

        XCTAssertEqual(vm.weightEntries.count, 1)
        XCTAssertEqual(vm.weightEntries.first?.weightKg, 79.2)
        XCTAssertEqual(vm.weeklyWeightTrend.count, 8)
    }
}
