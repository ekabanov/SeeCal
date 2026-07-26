import XCTest
import SeeCalDomain
@testable import SeeCalPersistence

final class FileBackedUserPreferencesStoreTests: XCTestCase {
    private var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func testMissingFileStartsWithNoSavedState() async throws {
        let fileURL = tempDirectory.appendingPathComponent("does_not_exist.json")
        let store = FileBackedUserPreferencesStore(fileURL: fileURL)

        let target = try await store.loadDailyTarget()
        let profile = try await store.loadUserProfile()
        XCTAssertNil(target)
        XCTAssertNil(profile)
    }

    func testDailyTargetAndProfileRoundTripAcrossFreshInstances() async throws {
        let fileURL = tempDirectory.appendingPathComponent("preferences.json")
        let target = DailyNutritionTarget(calories: 2400, proteinGrams: 160, fatGrams: 75, carbsGrams: 230)
        let profile = UserProfile(
            biologicalSex: .female,
            ageYears: 29,
            heightCm: 168,
            weightKg: 62.5,
            activityLevel: .moderatelyActive,
            goalPace: .loseSlow
        )

        let store = FileBackedUserPreferencesStore(fileURL: fileURL)
        try await store.saveDailyTarget(target)
        try await store.saveUserProfile(profile)

        // Simulate a process restart with a fresh actor backed by the same file.
        let reloaded = FileBackedUserPreferencesStore(fileURL: fileURL)
        let loadedTarget = try await reloaded.loadDailyTarget()
        let loadedProfile = try await reloaded.loadUserProfile()

        XCTAssertEqual(loadedTarget, target)
        XCTAssertEqual(loadedProfile, profile)
    }
}
