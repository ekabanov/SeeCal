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

final class AppViewModelEditFlowTests: XCTestCase {
    @MainActor
    private func makeViewModel(store: MealLogStore = InMemoryMealLogStore()) -> AppViewModel {
        AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]), store: store)
    }

    @MainActor
    func testEditFlowUpdatesTotals() async throws {
        let vm = makeViewModel()

        let entry = MealLogEntry(
            mealType: .lunch,
            imagePath: "/tmp/meal.jpg",
            items: [
                MealItem(
                    name: "meal",
                    grams: 200,
                    base: MealItemBase(grams: 200, kcal: 400, protein: 25, fat: 12, carbs: 38)
                )
            ]
        )
        await vm.logMeal(entry)
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

    @MainActor
    func testDeleteMealRemovesItsPhotoFile() async throws {
        let vm = makeViewModel()

        // A real file standing in for the captured photo.
        let photoURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("seecal-delete-test-\(UUID().uuidString).jpg")
        try Data([0xFF, 0xD8, 0xFF]).write(to: photoURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: photoURL.path))

        let entry = MealLogEntry(
            mealType: .dinner,
            imagePath: photoURL.path,
            items: [
                MealItem(
                    name: "meal",
                    grams: 100,
                    base: MealItemBase(grams: 100, kcal: 200, protein: 10, fat: 5, carbs: 20)
                )
            ]
        )
        await vm.logMeal(entry)

        await vm.deleteMeal(id: entry.id)

        XCTAssertTrue(vm.mealEntries.isEmpty)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: photoURL.path),
            "Deleting a meal must clean up its captured photo file"
        )
    }
}
