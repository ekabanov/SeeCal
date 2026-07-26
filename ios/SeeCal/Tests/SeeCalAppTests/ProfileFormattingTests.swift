import XCTest
import SeeCalDomain
@testable import SeeCalApp

/// Exact display strings for the goal-math line (spec §3 step 6 / §7) and the
/// weekly-rate slider (spec §2/§3 step 5), matching the prototype's
/// `renderProfile()` output verbatim — typographic minus (U+2212) throughout.
final class ProfileFormattingTests: XCTestCase {
    private static func utcDate(year: Int, month: Int, day: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: DateComponents(year: year, month: month, day: day))!
    }

    private static let fixedNow = utcDate(year: 2026, month: 7, day: 26)

    private static func referenceProfile(weeklyRateKg: Double = -0.5) -> UserProfile {
        UserProfile(
            sex: .male,
            dateOfBirth: utcDate(year: 1988, month: 3, day: 14),
            heightCm: 183,
            weightKg: 78.4,
            activity: .moderate,
            weeklyRateKg: weeklyRateKg
        )
    }

    // MARK: - Goal math string (spec §3 step 6 exact format)

    func testGoalMathStringMatchesReferenceVectorExactly() {
        let text = AppViewModel.goalMathString(for: Self.referenceProfile(), now: Self.fixedNow)
        XCTAssertEqual(
            text,
            "BMR 1,743 × 1.55 (moderate) = 2,701 kcal burned − 550 kcal (−0.5 kg/wk) = 2,150 kcal"
        )
    }

    func testGoalMathStringOmitsAdjustmentAtMaintain() {
        let text = AppViewModel.goalMathString(for: Self.referenceProfile(weeklyRateKg: 0), now: Self.fixedNow)
        XCTAssertEqual(text, "BMR 1,743 × 1.55 (moderate) = 2,701 kcal burned = 2,700 kcal")
    }

    func testGoalMathStringGainVariantUsesPlusSigns() {
        let text = AppViewModel.goalMathString(for: Self.referenceProfile(weeklyRateKg: 0.5), now: Self.fixedNow)
        XCTAssertEqual(
            text,
            "BMR 1,743 × 1.55 (moderate) = 2,701 kcal burned + 550 kcal (+0.5 kg/wk) = 3,250 kcal"
        )
    }

    // MARK: - Weekly-rate label ("−0.5 kg / week", "0 — maintain")

    func testRateLabelUsesTypographicMinus() {
        XCTAssertEqual(WeeklyRateFormat.label(for: -0.5), "−0.5 kg / week")
        XCTAssertEqual(WeeklyRateFormat.label(for: -1.0), "−1.0 kg / week")
        XCTAssertEqual(WeeklyRateFormat.label(for: 0.3), "+0.3 kg / week")
        XCTAssertEqual(WeeklyRateFormat.label(for: 0.5), "+0.5 kg / week")
    }

    func testRateLabelAtZeroIsMaintain() {
        XCTAssertEqual(WeeklyRateFormat.label(for: 0), "0 — maintain")
        // The prototype treats |r| < 0.049 as zero (float-noise guard).
        XCTAssertEqual(WeeklyRateFormat.label(for: 0.001), "0 — maintain")
        XCTAssertEqual(WeeklyRateFormat.label(for: -0.001), "0 — maintain")
    }

    func testRateKind() {
        XCTAssertEqual(WeeklyRateFormat.kind(for: -0.5), "Lose weight")
        XCTAssertEqual(WeeklyRateFormat.kind(for: 0), "Maintain")
        XCTAssertEqual(WeeklyRateFormat.kind(for: 0.3), "Gain weight")
    }

    // MARK: - Band classification → note copy + warn state (spec §2)

    func testNoteInsideBandIsRecommendation() {
        for rate in [-0.75, -0.5, -0.25, 0.0, 0.25] {
            XCTAssertEqual(
                WeeklyRateFormat.note(for: rate),
                "Recommended: lose 0.25–0.75 kg per week.",
                "rate \(rate)"
            )
            XCTAssertFalse(WeeklyRateFormat.isWarning(for: rate), "rate \(rate)")
        }
    }

    func testNoteBelowBandWarnsAboutSustainability() {
        for rate in [-0.8, -1.0] {
            XCTAssertEqual(
                WeeklyRateFormat.note(for: rate),
                "Faster than 0.75 kg/week is hard to sustain and not recommended.",
                "rate \(rate)"
            )
            XCTAssertTrue(WeeklyRateFormat.isWarning(for: rate), "rate \(rate)")
        }
    }

    func testNoteAboveBandWarnsAboutFatGain() {
        for rate in [0.3, 0.5] {
            XCTAssertEqual(
                WeeklyRateFormat.note(for: rate),
                "Gaining faster than 0.25 kg/week mostly adds fat.",
                "rate \(rate)"
            )
            XCTAssertTrue(WeeklyRateFormat.isWarning(for: rate), "rate \(rate)")
        }
    }

    // MARK: - Profile header + weight trend strings (spec §7)

    func testLoggingSinceString() {
        XCTAssertEqual(
            AppViewModel.loggingSinceString(for: Self.utcDate(year: 2026, month: 3, day: 15)),
            "March 2026"
        )
    }

    func testMonthTrendString() {
        XCTAssertEqual(AppViewModel.monthTrendString(deltaKg: -1.2), "−1.2 kg this month")
        XCTAssertEqual(AppViewModel.monthTrendString(deltaKg: 0.4), "+0.4 kg this month")
        XCTAssertNil(AppViewModel.monthTrendString(deltaKg: 0.0))
        XCTAssertNil(AppViewModel.monthTrendString(deltaKg: nil))
    }
}
