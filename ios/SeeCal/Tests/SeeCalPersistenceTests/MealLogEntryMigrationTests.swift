import XCTest
import SeeCalDomain
@testable import SeeCalPersistence

/// Covers loading pre-P2 persisted meal log files: entries stored a whole-entry
/// `FoodScanResult` (totals + its own `items: [ScanItem]`) under `scanResult`,
/// with no top-level `items: [MealItem]` or `name` field at all. Spec §2: these
/// must load as a single synthetic item named after the entry, based at the
/// entry's stored totals.
final class MealLogEntryMigrationTests: XCTestCase {
    private var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    /// Legacy (pre-P2) on-disk shape: no `items`/`name`/`volumeMl`/`maxHeightMm`
    /// keys at all, totals + confidence + its own nested `items` living entirely
    /// under `scanResult`.
    private func legacyFixtureJSON(id: UUID, createdAt: String) -> String {
        """
        [
          {
            "id": "\(id.uuidString)",
            "createdAt": "\(createdAt)",
            "mealType": "lunch",
            "imagePath": "/tmp/legacy.jpg",
            "scanResult": {
              "total_calories": 640,
              "protein_g": 38,
              "fat_g": 18,
              "carbs_g": 66,
              "confidence": 0.82,
              "items": [
                {"name": "chicken bowl", "estimated_grams": 320, "calories": 640, "protein_g": 38, "fat_g": 18, "carbs_g": 66}
              ],
              "uncertainty_flags": []
            }
          }
        ]
        """
    }

    /// Even-older/degenerate legacy shape: a `scanResult` whose own `items` array is
    /// empty, so there is no nested grams value to recover at all — nominal grams
    /// must fall back to 100 g.
    private func legacyFixtureJSONWithoutNestedItems(id: UUID, createdAt: String) -> String {
        """
        [
          {
            "id": "\(id.uuidString)",
            "createdAt": "\(createdAt)",
            "mealType": "breakfast",
            "imagePath": "/tmp/legacy2.jpg",
            "scanResult": {
              "total_calories": 300,
              "protein_g": 20,
              "fat_g": 8,
              "carbs_g": 30,
              "items": []
            }
          }
        ]
        """
    }

    func testLegacyEntryMigratesToSingleSyntheticItemUsingNestedGrams() async throws {
        let id = UUID()
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        try legacyFixtureJSON(id: id, createdAt: "2026-05-01T12:00:00Z").write(to: fileURL, atomically: true, encoding: .utf8)

        let store = FileBackedMealLogStore(fileURL: fileURL)
        let all = try await store.fetchAll()

        XCTAssertEqual(all.count, 1)
        let entry = try XCTUnwrap(all.first)
        XCTAssertEqual(entry.id, id)
        XCTAssertEqual(entry.mealType, .lunch)

        // Single synthetic item, named after the entry (mealType, the only naming
        // signal a legacy entry carries) since there was no top-level `name`.
        XCTAssertEqual(entry.items.count, 1)
        XCTAssertEqual(entry.name, "Lunch")
        XCTAssertEqual(entry.items[0].name, "Lunch")

        // Nominal grams recovered from the legacy scanResult's own nested item.
        XCTAssertEqual(entry.items[0].grams, 320)
        XCTAssertEqual(entry.items[0].base.grams, 320)

        // Base values are the entry's stored totals (spec: "base = stored totals").
        XCTAssertEqual(entry.items[0].base.kcal, 640)
        XCTAssertEqual(entry.items[0].base.protein, 38)
        XCTAssertEqual(entry.items[0].base.fat, 18)
        XCTAssertEqual(entry.items[0].base.carbs, 66)

        // Freshly-synthesized item starts unscaled (current grams == base grams),
        // so derived totals match the legacy stored totals exactly.
        XCTAssertEqual(entry.totals.calories, 640, accuracy: 0.001)
        XCTAssertEqual(entry.totals.proteinGrams, 38, accuracy: 0.001)
        XCTAssertEqual(entry.totals.fatGrams, 18, accuracy: 0.001)
        XCTAssertEqual(entry.totals.carbsGrams, 66, accuracy: 0.001)
    }

    func testLegacyEntryWithoutNestedItemsFallsBackTo100Grams() async throws {
        let id = UUID()
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        try legacyFixtureJSONWithoutNestedItems(id: id, createdAt: "2026-04-01T08:00:00Z")
            .write(to: fileURL, atomically: true, encoding: .utf8)

        let store = FileBackedMealLogStore(fileURL: fileURL)
        let all = try await store.fetchAll()

        XCTAssertEqual(all.count, 1)
        let entry = try XCTUnwrap(all.first)
        XCTAssertEqual(entry.name, "Breakfast")
        XCTAssertEqual(entry.items.count, 1)
        XCTAssertEqual(entry.items[0].grams, 100)
        XCTAssertEqual(entry.items[0].base.grams, 100)
        XCTAssertEqual(entry.totals.calories, 300, accuracy: 0.001)
    }

    func testMigratedEntryRoundTripsToNewSchemaOnNextSave() async throws {
        // After a migrated entry is re-saved, the file no longer needs the legacy
        // `scanResult` shape to be read back correctly — the new `items` field is
        // now authoritative.
        let id = UUID()
        let fileURL = tempDirectory.appendingPathComponent("meal_log.json")
        try legacyFixtureJSON(id: id, createdAt: "2026-05-01T12:00:00Z").write(to: fileURL, atomically: true, encoding: .utf8)

        let store = FileBackedMealLogStore(fileURL: fileURL)
        var all = try await store.fetchAll()
        let migrated = try XCTUnwrap(all.first)
        try await store.update(migrated) // forces a fresh write in the new schema

        let raw = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertTrue(raw.contains("\"items\""))
        XCTAssertFalse(raw.contains("\"scanResult\""))

        let reloaded = FileBackedMealLogStore(fileURL: fileURL)
        all = try await reloaded.fetchAll()
        XCTAssertEqual(try XCTUnwrap(all.first).totals.calories, 640, accuracy: 0.001)
    }

    // The corruption-preserving `.bak-<ts>` behavior is unrelated to schema version
    // and is already covered end-to-end by FileBackedStoreCorruptionTests; nothing
    // about this migration path changes it (both branches only read via the same
    // FileBackedStoreIO.read, which backs up on genuine decode failure, not on a
    // recognized legacy shape).
}
