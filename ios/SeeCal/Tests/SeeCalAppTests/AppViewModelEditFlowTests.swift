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
    func testEditFlowUpdatesTotals() async throws {
        let runtime = MLXQwenRuntime(engine: SingleResultEngine())
        let orchestrator = RuntimeOrchestrator(runtimes: [runtime])
        let store = InMemoryMealLogStore()
        let vm = AppViewModel(orchestrator: orchestrator, store: store)

        await vm.addMealPhoto(imagePath: "/tmp/meal.jpg", mealType: .lunch, userHint: nil)
        guard let first = vm.mealEntries.first else {
            XCTFail("Expected meal entry")
            return
        }

        // Edit flow: build a draft from the logged entry, bump the item's grams
        // (200g base -> 220g, a 1.1x scale), and commit the update.
        var draft = MealEditDraft(entry: first)
        guard let itemID = draft.items.first?.id else {
            XCTFail("Expected at least one item")
            return
        }
        draft.setGrams(itemID: itemID, to: 220)

        let updatedEntry = try draft.committedEntry()
        await vm.updateMeal(updatedEntry)

        XCTAssertEqual(vm.consumedToday.calories, 440, accuracy: 0.0001)
        XCTAssertEqual(vm.consumedToday.proteinGrams, 27.5, accuracy: 0.0001)
    }
}
