import XCTest
import SeeCalDomain
@testable import SeeCalPersistence

final class FileBackedWeightLogStoreTests: XCTestCase {
    private var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func testMissingFileStartsEmpty() async throws {
        let fileURL = tempDirectory.appendingPathComponent("does_not_exist.json")
        let store = FileBackedWeightLogStore(fileURL: fileURL)
        let all = try await store.fetchAll()
        XCTAssertTrue(all.isEmpty)
    }

    func testSaveRoundTripsAcrossFreshInstances() async throws {
        let fileURL = tempDirectory.appendingPathComponent("weight_log.json")
        let entry = WeightEntry(date: Date(), weightKg: 78.4)

        let store = FileBackedWeightLogStore(fileURL: fileURL)
        try await store.save(entry)

        let reloaded = FileBackedWeightLogStore(fileURL: fileURL)
        let all = try await reloaded.fetchAll()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all.first?.id, entry.id)
        XCTAssertEqual(all.first?.weightKg, 78.4)
    }
}
