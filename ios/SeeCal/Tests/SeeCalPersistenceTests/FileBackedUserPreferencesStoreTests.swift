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
            sex: .female,
            dateOfBirth: Self.date(year: 1997, month: 4, day: 2),
            heightCm: 168,
            weightKg: 62.5,
            activity: .moderate,
            weeklyRateKg: -0.25
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

    /// P1 migration: UserProfile's persisted shape changed from
    /// biologicalSex/ageYears/heightCm(Double)/activityLevel/goalPace to
    /// sex/dateOfBirth/heightCm(Int)/activity/weeklyRateKg. A preferences.json
    /// written by the pre-P1 app must still load, with the new fields
    /// (dateOfBirth, weeklyRateKg) synthesized to spec defaults.
    func testLegacyShapeProfileMigratesWithSpecDefaults() async throws {
        let fileURL = tempDirectory.appendingPathComponent("preferences.json")
        let legacyJSON = """
        {
          "userProfile": {
            "biologicalSex": "female",
            "ageYears": 29,
            "heightCm": 168.0,
            "weightKg": 62.5,
            "activityLevel": "moderatelyActive",
            "goalPace": "loseSlow"
          }
        }
        """
        try Data(legacyJSON.utf8).write(to: fileURL)

        let store = FileBackedUserPreferencesStore(fileURL: fileURL)
        let profile = try await store.loadUserProfile()

        let loaded = try XCTUnwrap(profile)
        XCTAssertEqual(loaded.sex, .female)
        XCTAssertEqual(loaded.heightCm, 168) // Double -> Int
        XCTAssertEqual(loaded.weightKg, 62.5)
        XCTAssertEqual(loaded.activity, .moderate) // moderatelyActive -> moderate
        XCTAssertEqual(loaded.weeklyRateKg, -0.25, accuracy: 0.0001) // loseSlow -> -0.25
        // dateOfBirth has no legacy equivalent value to migrate exactly, but it
        // must be synthesized (roughly) from the legacy ageYears so the computed
        // age stays close to 29, rather than defaulting to some arbitrary age.
        let age = GoalCalculator.age(dateOfBirth: loaded.dateOfBirth)
        XCTAssertEqual(age, 29, accuracy: 1)
    }

    /// A legacy profile missing goalPace entirely (shouldn't happen in practice,
    /// but exercises the "missing new fields -> spec defaults" contract) falls
    /// back to the spec default weeklyRateKg of -0.5.
    func testLegacyShapeProfileWithoutGoalPaceDefaultsWeeklyRate() async throws {
        let fileURL = tempDirectory.appendingPathComponent("preferences.json")
        let legacyJSON = """
        {
          "userProfile": {
            "biologicalSex": "male",
            "ageYears": 40,
            "heightCm": 180.0,
            "weightKg": 82.0,
            "activityLevel": "sedentary"
          }
        }
        """
        try Data(legacyJSON.utf8).write(to: fileURL)

        let store = FileBackedUserPreferencesStore(fileURL: fileURL)
        let profile = try await store.loadUserProfile()

        let loaded = try XCTUnwrap(profile)
        XCTAssertEqual(loaded.weeklyRateKg, UserProfile.defaultWeeklyRateKg)
        XCTAssertEqual(loaded.activity, .sedentary)
    }

    private static func date(year: Int, month: Int, day: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: DateComponents(year: year, month: month, day: day))!
    }
}
