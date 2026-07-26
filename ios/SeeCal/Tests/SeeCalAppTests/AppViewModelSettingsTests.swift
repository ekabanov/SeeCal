import XCTest
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
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

/// Settings §8 (capture toggles + on-device model card) and History §6
/// (recent-meals filtering) behavior on `AppViewModel`.
final class AppViewModelSettingsTests: XCTestCase {
    // MARK: - Capture preferences

    @MainActor
    func testCapturePreferencesDefaultToBothOnBeforeAnyLoad() {
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore()
        )
        XCTAssertTrue(vm.capturePreferences.useLiDARDepth)
        XCTAssertTrue(vm.capturePreferences.captureCoachingEnabled)
    }

    @MainActor
    func testLoadEntriesPicksUpPersistedCapturePreferences() async {
        let preferencesStore = InMemoryUserPreferencesStore(
            initialCapturePreferences: CapturePreferences(useLiDARDepth: false, captureCoachingEnabled: false)
        )
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: preferencesStore
        )

        await vm.loadEntries()

        XCTAssertFalse(vm.capturePreferences.useLiDARDepth)
        XCTAssertFalse(vm.capturePreferences.captureCoachingEnabled)
    }

    @MainActor
    func testUpdateCapturePreferencesPersistsAndRoundTripsThroughANewViewModel() async {
        let preferencesStore = InMemoryUserPreferencesStore()
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: preferencesStore
        )
        await vm.loadEntries()

        await vm.updateCapturePreferences(CapturePreferences(useLiDARDepth: false, captureCoachingEnabled: true))
        XCTAssertFalse(vm.capturePreferences.useLiDARDepth)
        XCTAssertTrue(vm.capturePreferences.captureCoachingEnabled)

        // A freshly-constructed view model backed by the SAME store sees the change.
        let reloaded = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            preferencesStore: preferencesStore
        )
        await reloaded.loadEntries()
        XCTAssertFalse(reloaded.capturePreferences.useLiDARDepth)
        XCTAssertTrue(reloaded.capturePreferences.captureCoachingEnabled)
    }

    // MARK: - Model info wiring (spec §8: read from bundled config, never hardcoded)

    @MainActor
    func testModelInfoFallsBackToNotBundledWhenNoPathsGiven() {
        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore()
        )
        XCTAssertEqual(vm.modelInfo, ModelInfo.notBundled)
    }

    @MainActor
    func testModelInfoResolvesAdapterVersionFromConfiguredPath() throws {
        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppViewModelSettingsTests-\(UUID().uuidString)", isDirectory: true)
        let adapterDir = tempDirectory.appendingPathComponent("adapters_v9_swift", isDirectory: true)
        try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)
        try #"{"fine_tune_type":"lora"}"#.write(
            to: adapterDir.appendingPathComponent("adapter_config.json"),
            atomically: true,
            encoding: .utf8
        )
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        let vm = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            adapterPath: adapterDir.path
        )
        XCTAssertEqual(vm.modelInfo.adapterVersionLabel, "v9")
    }

    // MARK: - Model card copy formatting

    func testModelCardTitleIncludesAdapterVersionWhenPresent() {
        let info = ModelInfo(modelLabel: "Qwen3.5-4B", adapterVersionLabel: "v6", quantizationLabel: "4-bit quantized")
        XCTAssertEqual(AppViewModel.modelCardTitle(info: info), "Qwen3.5-4B · SeeCal adapter v6")
    }

    func testModelCardTitleFallsBackWhenNoAdapterVersion() {
        let info = ModelInfo(modelLabel: "Qwen3.5-4B", adapterVersionLabel: nil, quantizationLabel: nil)
        XCTAssertEqual(AppViewModel.modelCardTitle(info: info), "Qwen3.5-4B · base model (no adapter bundled)")
    }

    func testModelCardSubtitleIncludesQuantizationAndFixedProvenanceCopy() {
        let info = ModelInfo(modelLabel: "Qwen3.5-4B", adapterVersionLabel: "v6", quantizationLabel: "4-bit quantized")
        let subtitle = AppViewModel.modelCardSubtitle(info: info)
        XCTAssertTrue(subtitle.hasPrefix("4-bit quantized ·"))
        XCTAssertTrue(subtitle.contains("typical calorie error \u{00B1}12%."))
        XCTAssertTrue(subtitle.contains("Photos are processed locally and never uploaded."))
    }

    func testModelCardSubtitleFallsBackWhenQuantizationUnknown() {
        let info = ModelInfo(modelLabel: "Qwen3.5-4B", adapterVersionLabel: nil, quantizationLabel: nil)
        XCTAssertTrue(AppViewModel.modelCardSubtitle(info: info).hasPrefix("Quantization not bundled ·"))
    }

    // MARK: - Recent meals (History §6: previous days only, most recent first, capped 10)

    @MainActor
    func testRecentMealEntriesExcludesTodayAndCapsAtTen() async throws {
        let store = InMemoryMealLogStore()

        let todayEntry = MealLogEntry(
            createdAt: Date(),
            mealType: .lunch,
            imagePath: "/tmp/today.jpg",
            items: [MealItem(name: "today", grams: 100, base: MealItemBase(grams: 100, kcal: 500, protein: 30, fat: 10, carbs: 40))]
        )
        try await store.save(todayEntry)

        for daysAgo in 1...12 {
            let date = Calendar.current.date(byAdding: .day, value: -daysAgo, to: Date())!
            let entry = MealLogEntry(
                name: "meal\(daysAgo)",
                createdAt: date,
                mealType: .dinner,
                imagePath: "/tmp/day\(daysAgo).jpg",
                items: [MealItem(name: "meal\(daysAgo)", grams: 100, base: MealItemBase(grams: 100, kcal: Double(daysAgo * 10), protein: 1, fat: 1, carbs: 1))]
            )
            try await store.save(entry)
        }

        let vm = AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]), store: store)
        await vm.loadEntries()

        let recent = vm.recentMealEntries
        XCTAssertEqual(recent.count, 10)
        XCTAssertFalse(recent.contains { $0.id == todayEntry.id })
        // Most recent (1 day ago) first.
        XCTAssertEqual(recent.first?.name, "meal1")
        // Strictly descending by createdAt.
        for (a, b) in zip(recent, recent.dropFirst()) {
            XCTAssertGreaterThan(a.createdAt, b.createdAt)
        }
    }
}
