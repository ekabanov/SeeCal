import XCTest
@testable import SeeCalDomain

final class UserProfileTests: XCTestCase {
    private func makeProfile(weeklyRateKg: Double) -> UserProfile {
        UserProfile(
            sex: .male,
            dateOfBirth: Date(),
            heightCm: 180,
            weightKg: 80,
            activity: .moderate,
            weeklyRateKg: weeklyRateKg
        )
    }

    func testWeeklyRateKgClampsToSpecRange() {
        XCTAssertEqual(makeProfile(weeklyRateKg: -5.0).weeklyRateKg, -1.0)
        XCTAssertEqual(makeProfile(weeklyRateKg: 5.0).weeklyRateKg, 0.5)
        XCTAssertEqual(makeProfile(weeklyRateKg: -0.5).weeklyRateKg, -0.5)
        XCTAssertEqual(makeProfile(weeklyRateKg: -1.0).weeklyRateKg, -1.0)
        XCTAssertEqual(makeProfile(weeklyRateKg: 0.5).weeklyRateKg, 0.5)
    }

    func testDefaultWeeklyRateKgIsMinusPointFive() {
        let profile = UserProfile(
            sex: .female,
            dateOfBirth: Date(),
            heightCm: 165,
            weightKg: 60,
            activity: .light
        )
        XCTAssertEqual(profile.weeklyRateKg, -0.5)
    }

    func testCurrentShapeRoundTripsThroughJSON() throws {
        let profile = UserProfile(
            sex: .female,
            dateOfBirth: Date(timeIntervalSince1970: 500_000_000),
            heightCm: 168,
            weightKg: 62.5,
            activity: .active,
            weeklyRateKg: 0.25
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let data = try encoder.encode(profile)
        let decoded = try decoder.decode(UserProfile.self, from: data)

        XCTAssertEqual(decoded, profile)
    }
}
