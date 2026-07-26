import XCTest
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
@testable import SeeCalApp

private struct SingleResultEngine: NativeQwenVisionEngine {
    func run(imagePath: String, prompt: String) async throws -> String {
        _ = imagePath
        _ = prompt
        return """
        {
          "total_calories": 400,
          "protein_g": 25,
          "fat_g": 12,
          "carbs_g": 38,
          "confidence": 0.85,
          "items": [{"name": "meal", "estimated_grams": 200, "calories": 400, "protein_g": 25, "fat_g": 12, "carbs_g": 38}],
          "uncertainty_flags": []
        }
        """
    }
}

final class AppViewModelEditFlowTests: XCTestCase {
    @MainActor
    func testEditFlowUpdatesTotals() async {
        let runtime = MLXQwenRuntime(engine: SingleResultEngine())
        let orchestrator = RuntimeOrchestrator(runtimes: [runtime])
        let store = InMemoryMealLogStore()
        let vm = AppViewModel(orchestrator: orchestrator, store: store)

        await vm.addMealPhoto(imagePath: "/tmp/meal.jpg", mealType: .lunch, userHint: nil)
        guard let first = vm.mealEntries.first else {
            XCTFail("Expected meal entry")
            return
        }

        let updated = FoodScanResult(
            totalCalories: 550,
            proteinGrams: 35,
            fatGrams: 15,
            carbsGrams: 50,
            confidence: 0.8,
            items: [ScanItem(name: "meal", estimatedGrams: 220, calories: 550, proteinGrams: 35, fatGrams: 15, carbsGrams: 50)],
            uncertaintyFlags: []
        )

        await vm.updateMeal(first, with: updated)

        XCTAssertEqual(vm.consumedToday.calories, 550)
        XCTAssertEqual(vm.consumedToday.proteinGrams, 35)
    }
}
