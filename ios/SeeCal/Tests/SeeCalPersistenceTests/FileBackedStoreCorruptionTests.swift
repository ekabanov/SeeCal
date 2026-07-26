import XCTest
import SeeCalDomain
@testable import SeeCalPersistence

/// Covers the decode-failure recovery path in `FileBackedStoreIO`: an unreadable
/// store file must never be silently overwritten by the next save. Instead it is
/// renamed to `<name>.bak-<timestamp>` (original bytes preserved) and the store
/// comes up empty.
final class FileBackedStoreCorruptionTests: XCTestCase {
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

    private func backupFiles(for fileURL: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: tempDirectory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("\(fileURL.lastPathComponent).bak-") }
    }

    func testCorruptFileIsBackedUpAndStoreRecovers() async throws {
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        let corruptBytes = Data("{\"this is\": not valid JSON at all]]".utf8)
        try corruptBytes.write(to: fileURL)

        // (1) Store initializes empty without throwing.
        let store = FileBackedMealLogStore(fileURL: fileURL)
        let all = try await store.fetchAll()
        XCTAssertTrue(all.isEmpty)

        // (2) The unreadable file was renamed to <name>.bak-<timestamp> next to the
        // original, with the original bytes intact.
        let backups = try backupFiles(for: fileURL)
        XCTAssertEqual(backups.count, 1)
        let backupBytes = try Data(contentsOf: XCTUnwrap(backups.first))
        XCTAssertEqual(backupBytes, corruptBytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fileURL.path))

        // (3) A subsequent save + reload round-trips fresh state, and the backup
        // is left untouched.
        let entry = makeEntry()
        try await store.save(entry)

        let reloaded = FileBackedMealLogStore(fileURL: fileURL)
        let reloadedEntries = try await reloaded.fetchAll()
        XCTAssertEqual(reloadedEntries.count, 1)
        XCTAssertEqual(reloadedEntries.first?.id, entry.id)

        let backupsAfterSave = try backupFiles(for: fileURL)
        XCTAssertEqual(backupsAfterSave.count, 1)
        XCTAssertEqual(try Data(contentsOf: XCTUnwrap(backupsAfterSave.first)), corruptBytes)
    }

    func testSchemaMismatchIsBackedUpAndStoreRecovers() async throws {
        // Valid JSON, wrong shape — the forward-schema-change case (e.g. a newer app
        // version wrote a format this version cannot decode).
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        let futureSchemaBytes = Data("{\"schema_version\": 99, \"entries\": []}".utf8)
        try futureSchemaBytes.write(to: fileURL)

        let store = FileBackedMealLogStore(fileURL: fileURL)
        let all = try await store.fetchAll()
        XCTAssertTrue(all.isEmpty)

        let backups = try backupFiles(for: fileURL)
        XCTAssertEqual(backups.count, 1)
        XCTAssertEqual(try Data(contentsOf: XCTUnwrap(backups.first)), futureSchemaBytes)
    }

    func testEmptyFileDoesNotCreateBackup() async throws {
        // An empty file is not "unreadable user data" — it should behave like a
        // missing file, with nothing to preserve.
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        try Data().write(to: fileURL)

        let store = FileBackedMealLogStore(fileURL: fileURL)
        let all = try await store.fetchAll()
        XCTAssertTrue(all.isEmpty)

        let backups = try backupFiles(for: fileURL)
        XCTAssertTrue(backups.isEmpty)
    }
}
