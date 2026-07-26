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

final class AppViewModelTrackingTests: XCTestCase {
    @MainActor
    func testConsumedAndRemainingAfterLog() async {
        let target = DailyNutritionTarget(calories: 2000, proteinGrams: 150, fatGrams: 70, carbsGrams: 220)
        let viewModel = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore(),
            dailyTarget: target
        )

        let entry = MealLogEntry(
            mealType: .dinner,
            imagePath: "/tmp/meal.jpg",
            items: [
                MealItem(
                    name: "meal",
                    grams: 250,
                    base: MealItemBase(grams: 250, kcal: 700, protein: 40, fat: 20, carbs: 70)
                )
            ]
        )
        await viewModel.logMeal(entry)

        XCTAssertEqual(viewModel.consumedToday.calories, 700)
        XCTAssertEqual(viewModel.remainingToday.calories, 1300)
        XCTAssertEqual(viewModel.remainingToday.proteinGrams, 110)
    }
}
