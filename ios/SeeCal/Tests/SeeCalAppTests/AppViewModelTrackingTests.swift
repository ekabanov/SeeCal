import XCTest
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
@testable import SeeCalApp

private struct StubRunnerEngine: NativeQwenVisionEngine {
    let json: String
    func run(imagePath: String, prompt: String) async throws -> String {
        _ = imagePath
        _ = prompt
        return json
    }
}

final class AppViewModelTrackingTests: XCTestCase {
    @MainActor
    func testConsumedAndRemainingAfterAdd() async {
        let json = """
        {
          "total_calories": 700,
          "protein_g": 40,
          "fat_g": 20,
          "carbs_g": 70,
          "confidence": 0.88,
          "items": [{"name": "meal", "estimated_grams": 250, "calories": 700, "protein_g": 40, "fat_g": 20, "carbs_g": 70}],
          "uncertainty_flags": []
        }
        """

        let runtime = MLXQwenRuntime(engine: StubRunnerEngine(json: json))
        let orchestrator = RuntimeOrchestrator(runtimes: [runtime])
        let target = DailyNutritionTarget(calories: 2000, proteinGrams: 150, fatGrams: 70, carbsGrams: 220)
        let viewModel = AppViewModel(orchestrator: orchestrator, store: InMemoryMealLogStore(), dailyTarget: target)

        await viewModel.addMealPhoto(imagePath: "/tmp/meal.jpg", mealType: .dinner, userHint: nil)

        XCTAssertEqual(viewModel.consumedToday.calories, 700)
        XCTAssertEqual(viewModel.remainingToday.calories, 1300)
        XCTAssertEqual(viewModel.remainingToday.proteinGrams, 110)
    }
}
