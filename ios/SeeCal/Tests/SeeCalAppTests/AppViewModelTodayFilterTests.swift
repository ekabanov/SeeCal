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

final class AppViewModelTodayFilterTests: XCTestCase {
    @MainActor
    func testConsumedTodayExcludesEntriesFromOtherDays() async throws {
        let store = InMemoryMealLogStore()

        let todayEntry = MealLogEntry(
            createdAt: Date(),
            mealType: .lunch,
            imagePath: "/tmp/today.jpg",
            scanResult: FoodScanResult(
                totalCalories: 500,
                proteinGrams: 30,
                fatGrams: 10,
                carbsGrams: 40,
                items: [ScanItem(name: "a", estimatedGrams: 100, calories: 500, proteinGrams: 30, fatGrams: 10, carbsGrams: 40)]
            )
        )

        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        let yesterdayEntry = MealLogEntry(
            createdAt: yesterday,
            mealType: .dinner,
            imagePath: "/tmp/yesterday.jpg",
            scanResult: FoodScanResult(
                totalCalories: 900,
                proteinGrams: 50,
                fatGrams: 30,
                carbsGrams: 80,
                items: [ScanItem(name: "b", estimatedGrams: 200, calories: 900, proteinGrams: 50, fatGrams: 30, carbsGrams: 80)]
            )
        )

        try await store.save(todayEntry)
        try await store.save(yesterdayEntry)

        let vm = AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]), store: store)
        await vm.loadEntries()

        XCTAssertEqual(vm.mealEntries.count, 2, "both entries should still be loaded and visible in the log")
        XCTAssertEqual(vm.consumedToday.calories, 500, "only today's entry should count toward consumedToday")
        XCTAssertEqual(vm.consumedToday.proteinGrams, 30)
    }

    @MainActor
    func testConsumedTodayIsZeroWhenNoEntriesAreFromToday() async throws {
        let store = InMemoryMealLogStore()
        let lastWeek = Calendar.current.date(byAdding: .day, value: -7, to: Date())!
        let oldEntry = MealLogEntry(
            createdAt: lastWeek,
            mealType: .breakfast,
            imagePath: "/tmp/old.jpg",
            scanResult: FoodScanResult(
                totalCalories: 300,
                proteinGrams: 20,
                fatGrams: 8,
                carbsGrams: 30,
                items: [ScanItem(name: "c", estimatedGrams: 80, calories: 300, proteinGrams: 20, fatGrams: 8, carbsGrams: 30)]
            )
        )
        try await store.save(oldEntry)

        let vm = AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]), store: store)
        await vm.loadEntries()

        XCTAssertEqual(vm.consumedToday.calories, 0)
    }
}
