import XCTest
import SeeCalDomain
@testable import SeeCalApp

/// Spec §3 wizard state machine: step advance/back with clamping, per-step
/// button titles and back/skip visibility, skip defaults (spec §2 reference
/// profile), and value stepping clamps.
final class OnboardingDraftTests: XCTestCase {
    private static func utcDate(year: Int, month: Int, day: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: DateComponents(year: year, month: month, day: day))!
    }

    // MARK: - Step machine

    func testStepsAdvanceThroughAllSixInOrderAndClampAtEnd() {
        var draft = OnboardingDraft()
        XCTAssertEqual(OnboardingDraft.stepCount, 6)
        XCTAssertEqual(draft.step, .welcome)

        var visited: [OnboardingDraft.Step] = [draft.step]
        while !draft.isLastStep {
            draft.advance()
            visited.append(draft.step)
        }
        XCTAssertEqual(visited, [.welcome, .aboutYou, .body, .activity, .weeklyTarget, .goal])

        // Advancing past the last step is a no-op.
        draft.advance()
        XCTAssertEqual(draft.step, .goal)
    }

    func testGoBackClampsAtFirstStep() {
        var draft = OnboardingDraft(step: .aboutYou)
        draft.goBack()
        XCTAssertEqual(draft.step, .welcome)
        draft.goBack()
        XCTAssertEqual(draft.step, .welcome)
    }

    func testPrimaryButtonTitlesPerStep() {
        XCTAssertEqual(OnboardingDraft(step: .welcome).primaryButtonTitle, "Get started")
        XCTAssertEqual(OnboardingDraft(step: .aboutYou).primaryButtonTitle, "Continue")
        XCTAssertEqual(OnboardingDraft(step: .body).primaryButtonTitle, "Continue")
        XCTAssertEqual(OnboardingDraft(step: .activity).primaryButtonTitle, "Continue")
        XCTAssertEqual(OnboardingDraft(step: .weeklyTarget).primaryButtonTitle, "Continue")
        XCTAssertEqual(OnboardingDraft(step: .goal).primaryButtonTitle, "Start tracking")
    }

    func testBackHiddenOnFirstStepOnly() {
        XCTAssertFalse(OnboardingDraft(step: .welcome).showsBack)
        for step in OnboardingDraft.Step.allCases where step != .welcome {
            XCTAssertTrue(OnboardingDraft(step: step).showsBack, "\(step)")
        }
    }

    func testSkipShownOnFirstStepOnly() {
        XCTAssertTrue(OnboardingDraft(step: .welcome).showsSkip)
        for step in OnboardingDraft.Step.allCases where step != .welcome {
            XCTAssertFalse(OnboardingDraft(step: step).showsSkip, "\(step)")
        }
    }

    // MARK: - Skip defaults (spec §3: "defaults from §2 reference profile minus name")

    func testSkipDefaultsMatchSpecReferenceProfile() {
        let profile = OnboardingDraft.skipDefaults()

        XCTAssertEqual(profile.sex, .male)
        XCTAssertEqual(profile.dateOfBirth, Self.utcDate(year: 1988, month: 3, day: 14))
        XCTAssertEqual(profile.heightCm, 183)
        XCTAssertEqual(profile.weightKg, 78.4)
        XCTAssertEqual(profile.activity, .moderate)
        XCTAssertEqual(profile.weeklyRateKg, -0.5)

        // And the reference vector's goal follows from them (spec §2: 2150 at 2026-07-26).
        let fixedNow = Self.utcDate(year: 2026, month: 7, day: 26)
        XCTAssertEqual(GoalCalculator.goalCalories(for: profile, now: fixedNow), 2150)
    }

    // MARK: - Value stepping (spec §3 step 3)

    func testHeightSteppingClampsTo120Through230() {
        var draft = OnboardingDraft(heightCm: 121)
        draft.stepHeight(by: -1)
        XCTAssertEqual(draft.heightCm, 120)
        draft.stepHeight(by: -1)
        XCTAssertEqual(draft.heightCm, 120)

        draft = OnboardingDraft(heightCm: 229)
        draft.stepHeight(by: 1)
        XCTAssertEqual(draft.heightCm, 230)
        draft.stepHeight(by: 1)
        XCTAssertEqual(draft.heightCm, 230)
    }

    func testWeightSteppingClampsTo35Through250() {
        var draft = OnboardingDraft(weightKg: 35.4)
        draft.stepWeight(by: -0.5)
        XCTAssertEqual(draft.weightKg, 35)
        draft.stepWeight(by: -0.5)
        XCTAssertEqual(draft.weightKg, 35)

        draft = OnboardingDraft(weightKg: 249.6)
        draft.stepWeight(by: 0.5)
        XCTAssertEqual(draft.weightKg, 250)
    }

    func testSetWeeklyRateSnapsToTenthAndClamps() {
        var draft = OnboardingDraft()
        draft.setWeeklyRate(-0.30000000000000004) // slider float noise
        XCTAssertEqual(draft.weeklyRateKg, -0.3)
        draft.setWeeklyRate(-2)
        XCTAssertEqual(draft.weeklyRateKg, -1.0)
        draft.setWeeklyRate(2)
        XCTAssertEqual(draft.weeklyRateKg, 0.5)
    }

    // MARK: - Finish

    func testUserProfileCarriesAllEditedFields() {
        var draft = OnboardingDraft()
        draft.sex = .female
        draft.dateOfBirth = Self.utcDate(year: 1995, month: 6, day: 1)
        draft.stepHeight(by: -13) // 170
        draft.stepWeight(by: -8.4) // 70.0
        draft.activity = .light
        draft.setWeeklyRate(-0.3)

        let profile = draft.userProfile()
        XCTAssertEqual(profile.sex, .female)
        XCTAssertEqual(profile.dateOfBirth, Self.utcDate(year: 1995, month: 6, day: 1))
        XCTAssertEqual(profile.heightCm, 170)
        XCTAssertEqual(profile.weightKg, 70.0)
        XCTAssertEqual(profile.activity, .light)
        XCTAssertEqual(profile.weeklyRateKg, -0.3)
    }
}
