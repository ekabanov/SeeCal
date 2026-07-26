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

    /// Spec §7: the Profile weight stepper is the app's only weight input, so
    /// every profile weight change must feed the weight log — two edits across
    /// dates make the "this month" trend note computable.
    @MainActor
    func testProfileWeightEditsPopulateWeightLogAndMonthTrend() async {
        // Injectable clock: first save "10 days ago", second save "now".
        let realNow = Date()
        let tenDaysAgo = Calendar.current.date(byAdding: .day, value: -10, to: realNow)!
        let clock = ClockBox(now: tenDaysAgo)
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: InMemoryUserPreferencesStore(),
            weightStore: InMemoryWeightLogStore(),
            now: { clock.now }
        )

        var profile = UserProfile(
            sex: .male,
            dateOfBirth: Calendar.current.date(byAdding: .year, value: -30, to: realNow)!,
            heightCm: 180,
            weightKg: 80.0,
            activity: .moderate
        )
        await vm.updateProfile(profile)
        XCTAssertEqual(vm.weightEntries.count, 1, "First profile save seeds the weight log baseline")

        // Second edit, later date, lower weight.
        clock.now = realNow
        profile.weightKg = 78.8
        await vm.updateProfile(profile)

        XCTAssertEqual(vm.weightEntries.count, 2)
        XCTAssertEqual(vm.weightEntries.map(\.weightKg).sorted(), [78.8, 80.0])
        guard let change = vm.monthWeightChangeKg else {
            return XCTFail("Two weight entries across dates must yield a month trend")
        }
        XCTAssertEqual(change, -1.2, accuracy: 0.0001)
    }

    /// A profile edit that does NOT touch weight (e.g. activity) must not spam
    /// the weight log with duplicate entries.
    @MainActor
    func testProfileEditWithoutWeightChangeDoesNotAppendWeightEntry() async {
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: InMemoryUserPreferencesStore(),
            weightStore: InMemoryWeightLogStore()
        )

        var profile = UserProfile(
            sex: .female,
            dateOfBirth: Calendar.current.date(byAdding: .year, value: -25, to: Date())!,
            heightCm: 165,
            weightKg: 62.0,
            activity: .light
        )
        await vm.updateProfile(profile)
        XCTAssertEqual(vm.weightEntries.count, 1)

        profile.activity = .active
        await vm.updateProfile(profile)

        XCTAssertEqual(vm.weightEntries.count, 1, "No weight change, no new weight entry")
    }
}

/// Mutable reference wrapper so the test can advance the view model's
/// injected clock between calls.
private final class ClockBox: @unchecked Sendable {
    var now: Date
    init(now: Date) { self.now = now }
}
