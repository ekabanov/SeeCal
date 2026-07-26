import XCTest
import SeeCalDomain
@testable import SeeCalApp

final class GoalEditDraftTests: XCTestCase {
    func testConvertsToDailyTarget() throws {
        let draft = GoalEditDraft(target: DailyNutritionTarget(calories: 2000, proteinGrams: 140, fatGrams: 65, carbsGrams: 210))
        let target = try draft.toDailyTarget()

        XCTAssertEqual(target.calories, 2000)
        XCTAssertEqual(target.proteinGrams, 140)
    }
}
