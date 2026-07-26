import XCTest
@testable import SeeCalDomain

final class GoalCalculatorTests: XCTestCase {
    private func date(year: Int, month: Int, day: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: DateComponents(year: year, month: month, day: day))!
    }

    // MARK: - Spec §2 reference vector
    //
    // male, dob 1988-03-14 (age 38 on 2026-07-26), 183 cm, 78.4 kg, moderate,
    // -0.5 kg/wk -> BMR 1743, TDEE 2701, goal 2150.

    func testReferenceVectorAge() {
        let profile = UserProfile(
            sex: .male,
            dateOfBirth: date(year: 1988, month: 3, day: 14),
            heightCm: 183,
            weightKg: 78.4,
            activity: .moderate,
            weeklyRateKg: -0.5
        )
        let now = date(year: 2026, month: 7, day: 26)

        XCTAssertEqual(GoalCalculator.age(dateOfBirth: profile.dateOfBirth, now: now), 38)
    }

    func testReferenceVectorBMR() {
        let profile = UserProfile(
            sex: .male,
            dateOfBirth: date(year: 1988, month: 3, day: 14),
            heightCm: 183,
            weightKg: 78.4,
            activity: .moderate,
            weeklyRateKg: -0.5
        )
        let now = date(year: 2026, month: 7, day: 26)

        let bmr = GoalCalculator.bmr(for: profile, now: now)
        XCTAssertEqual(bmr.rounded(), 1743)
    }

    func testReferenceVectorTDEE() {
        let profile = UserProfile(
            sex: .male,
            dateOfBirth: date(year: 1988, month: 3, day: 14),
            heightCm: 183,
            weightKg: 78.4,
            activity: .moderate,
            weeklyRateKg: -0.5
        )
        let now = date(year: 2026, month: 7, day: 26)

        let tdee = GoalCalculator.tdee(for: profile, now: now)
        XCTAssertEqual(tdee.rounded(), 2701)
    }

    func testReferenceVectorGoal() {
        let profile = UserProfile(
            sex: .male,
            dateOfBirth: date(year: 1988, month: 3, day: 14),
            heightCm: 183,
            weightKg: 78.4,
            activity: .moderate,
            weeklyRateKg: -0.5
        )
        let now = date(year: 2026, month: 7, day: 26)

        XCTAssertEqual(GoalCalculator.goalCalories(for: profile, now: now), 2150)
    }

    func testReferenceVectorMacroSplit() {
        let target = GoalCalculator.macroTargets(forGoalCalories: 2150)
        XCTAssertEqual(target.calories, 2150)
        // 30/25/45% of 2150 kcal via 4/9/4 kcal/g.
        XCTAssertEqual(target.proteinGrams, (2150 * 0.30) / 4, accuracy: 0.001)
        XCTAssertEqual(target.fatGrams, (2150 * 0.25) / 9, accuracy: 0.001)
        XCTAssertEqual(target.carbsGrams, (2150 * 0.45) / 4, accuracy: 0.001)
    }

    // MARK: - Female offset (-161)

    func testFemaleOffsetBMR() {
        let profile = UserProfile(
            sex: .female,
            dateOfBirth: date(year: 1990, month: 1, day: 1),
            heightCm: 165,
            weightKg: 60,
            activity: .sedentary,
            weeklyRateKg: 0
        )
        let now = date(year: 2026, month: 1, day: 1) // exactly 36 years old

        let expected = 10 * 60.0 + 6.25 * 165.0 - 5 * 36.0 - 161
        XCTAssertEqual(GoalCalculator.bmr(for: profile, now: now), expected, accuracy: 0.0001)
    }

    // MARK: - 1200 kcal floor

    func testGoalFloorsAt1200() {
        // Small, sedentary, older profile with an aggressive loss rate: TDEE plus
        // the weekly-rate adjustment goes well below 1200 without the floor.
        let profile = UserProfile(
            sex: .female,
            dateOfBirth: date(year: 1966, month: 1, day: 1),
            heightCm: 150,
            weightKg: 45,
            activity: .sedentary,
            weeklyRateKg: -1.0
        )
        let now = date(year: 2026, month: 1, day: 1) // age 60

        let bmr = 10 * 45.0 + 6.25 * 150.0 - 5 * 60.0 - 161 // 926.5
        let tdee = bmr * 1.2 // 1111.8
        let adjusted = tdee + (-1.0 * 7700 / 7) // 11.8
        XCTAssertLessThan(adjusted, 1200, "test setup should actually exercise the floor")

        XCTAssertEqual(GoalCalculator.goalCalories(for: profile, now: now), 1200)
    }

    // MARK: - Round-to-10

    func testRoundToNearestTen() {
        XCTAssertEqual(GoalCalculator.roundToNearestTen(2151.2625), 2150)
        XCTAssertEqual(GoalCalculator.roundToNearestTen(2154.9), 2150)
        XCTAssertEqual(GoalCalculator.roundToNearestTen(2155.0), 2160)
        XCTAssertEqual(GoalCalculator.roundToNearestTen(1205.0), 1210)
        XCTAssertEqual(GoalCalculator.roundToNearestTen(1204.9), 1200)
        XCTAssertEqual(GoalCalculator.roundToNearestTen(0), 0)
    }

    // MARK: - Weekly-rate band classification edges

    func testWeeklyRateBandEdges() {
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: -1.0), .aggressiveLoss)
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: -0.76), .aggressiveLoss)
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: -0.75), .recommended) // lower edge, inclusive
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: -0.5), .recommended)
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: -0.25), .recommended) // upper edge, inclusive
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: -0.24), .neutral)
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: 0.0), .neutral)
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: 0.25), .neutral) // exactly +0.25 does not warn
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: 0.26), .aggressiveGain)
        XCTAssertEqual(GoalCalculator.weeklyRateBand(for: 0.5), .aggressiveGain)
    }
}
