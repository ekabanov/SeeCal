import XCTest
import SeeCalDomain
@testable import SeeCalPersistence

final class FileBackedMealLogStoreTests: XCTestCase {
    private var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    private func makeEntry(calories: Double = 500) -> MealLogEntry {
        MealLogEntry(
            mealType: .lunch,
            imagePath: "/tmp/photo.jpg",
            scanResult: FoodScanResult(
                totalCalories: calories,
                proteinGrams: 30,
                fatGrams: 12,
                carbsGrams: 50,
                confidence: 0.8,
                items: [ScanItem(name: "a", estimatedGrams: 100, calories: calories, proteinGrams: 30, fatGrams: 12, carbsGrams: 50)],
                uncertaintyFlags: []
            )
        )
    }

    func testMissingFileStartsEmpty() async throws {
        let fileURL = tempDirectory.appendingPathComponent("does_not_exist.json")
        let store = FileBackedMealLogStore(fileURL: fileURL)
        let all = try await store.fetchAll()
        XCTAssertTrue(all.isEmpty)
    }

    func testSaveRoundTripsAcrossFreshInstances() async throws {
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        let entry = makeEntry()

        let store = FileBackedMealLogStore(fileURL: fileURL)
        try await store.save(entry)

        XCTAssertTrue(FileManager.default.fileExists(atPath: fileURL.path))

        // Simulate a process restart: a brand new store instance backed by the same
        // file must load what was previously persisted, without any in-memory state
        // carried over.
        let reloaded = FileBackedMealLogStore(fileURL: fileURL)
        let all = try await reloaded.fetchAll()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all.first?.id, entry.id)
        XCTAssertEqual(all.first?.scanResult.totalCalories, 500)
        XCTAssertEqual(all.first?.imagePath, "/tmp/photo.jpg")
    }

    func testUpdateAndDeletePersistAcrossFreshInstances() async throws {
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        let entry = makeEntry()

        let store = FileBackedMealLogStore(fileURL: fileURL)
        try await store.save(entry)

        var updated = entry
        updated.scanResult.totalCalories = 640
        try await store.update(updated)

        let afterUpdate = FileBackedMealLogStore(fileURL: fileURL)
        var all = try await afterUpdate.fetchAll()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all.first?.scanResult.totalCalories, 640)

        try await afterUpdate.delete(id: entry.id)

        let afterDelete = FileBackedMealLogStore(fileURL: fileURL)
        all = try await afterDelete.fetchAll()
        XCTAssertTrue(all.isEmpty)
    }
}
