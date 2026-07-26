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
            items: [MealItem(name: "a", grams: 100, base: MealItemBase(grams: 100, kcal: 500, protein: 30, fat: 10, carbs: 40))]
        )

        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        let yesterdayEntry = MealLogEntry(
            createdAt: yesterday,
            mealType: .dinner,
            imagePath: "/tmp/yesterday.jpg",
            items: [MealItem(name: "b", grams: 200, base: MealItemBase(grams: 200, kcal: 900, protein: 50, fat: 30, carbs: 80))]
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
            items: [MealItem(name: "c", grams: 80, base: MealItemBase(grams: 80, kcal: 300, protein: 20, fat: 8, carbs: 30))]
        )
        try await store.save(oldEntry)

        let vm = AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]), store: store)
        await vm.loadEntries()

        XCTAssertEqual(vm.consumedToday.calories, 0)
    }
}
