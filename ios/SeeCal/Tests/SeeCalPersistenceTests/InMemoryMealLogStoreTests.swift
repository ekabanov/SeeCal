import XCTest
import SeeCalDomain
@testable import SeeCalPersistence

final class InMemoryMealLogStoreTests: XCTestCase {
    func testUpdateAndDelete() async throws {
        let store = InMemoryMealLogStore()
        let entry = MealLogEntry(
            mealType: .lunch,
            imagePath: "/tmp/photo.jpg",
            items: [MealItem(name: "a", grams: 100, base: MealItemBase(grams: 100, kcal: 500, protein: 30, fat: 12, carbs: 50))]
        )

        try await store.save(entry)

        var updated = entry
        updated.items[0].grams = 124 // 1.24x scale -> 620 kcal
        try await store.update(updated)

        var all = try await store.fetchAll()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all[0].totals.calories, 620, accuracy: 0.001)

        try await store.delete(id: entry.id)
        all = try await store.fetchAll()
        XCTAssertTrue(all.isEmpty)
    }
}
