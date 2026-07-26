import XCTest
import SeeCalDomain
@testable import SeeCalPersistence

final class InMemoryMealLogStoreTests: XCTestCase {
    func testUpdateAndDelete() async throws {
        let store = InMemoryMealLogStore()
        let entry = MealLogEntry(
            mealType: .lunch,
            imagePath: "/tmp/photo.jpg",
            scanResult: FoodScanResult(
                totalCalories: 500,
                proteinGrams: 30,
                fatGrams: 12,
                carbsGrams: 50,
                confidence: 0.8,
                items: [ScanItem(name: "a", estimatedGrams: 100, calories: 500, proteinGrams: 30, fatGrams: 12, carbsGrams: 50)],
                uncertaintyFlags: []
            )
        )

        try await store.save(entry)

        var updated = entry
        updated.scanResult.totalCalories = 620
        try await store.update(updated)

        var all = try await store.fetchAll()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all[0].scanResult.totalCalories, 620)

        try await store.delete(id: entry.id)
        all = try await store.fetchAll()
        XCTAssertTrue(all.isEmpty)
    }
}
