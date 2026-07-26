import XCTest
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
@testable import SeeCalApp

private struct NoopRuntime: InferenceRuntime {
    let name = "noop"
    let modelFamily = "qwen3.5-native-multimodal"
    func isAvailable() async -> Bool { false }
    func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        _ = request
        throw InferenceError.runtimeUnavailable("none")
    }
}

/// Spec §4 / P4 (`TodayScreen`) display-formatting tests: date subtitle, consumed
/// totals from mixed entries, macro target display grams, ring/bar fraction clamp.
final class AppViewModelTodayDisplayTests: XCTestCase {
    private func date(year: Int, month: Int, day: Int, hour: Int = 0, minute: Int = 0) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: DateComponents(year: year, month: month, day: day, hour: hour, minute: minute))!
    }

    // MARK: - Date subtitle formatting

    func testDateSubtitleFormatsFullWeekdayMonthDayNoYear() {
        // 2026-07-26 is a Sunday (spec §4 example: "SUNDAY, JULY 26" once
        // `PageHeader` upper-cases the raw subtitle string).
        let sunday = date(year: 2026, month: 7, day: 26)
        XCTAssertEqual(AppViewModel.dateSubtitle(for: sunday), "Sunday, July 26")
    }

    func testDateSubtitleUsesFullMonthNameNotAbbreviated() {
        let newYearsDay = date(year: 2026, month: 1, day: 1)
        XCTAssertEqual(AppViewModel.dateSubtitle(for: newYearsDay), "Thursday, January 1")
    }

    // MARK: - Consumed totals from mixed entries

    @MainActor
    func testConsumedTodayAggregatesAllMacrosFromMixedTodayEntries() async throws {
        let store = InMemoryMealLogStore()
        let today = Date()

        // Entry with a single item.
        let breakfast = MealLogEntry(
            createdAt: today,
            mealType: .breakfast,
            imagePath: "/tmp/breakfast.jpg",
            items: [
                MealItem(name: "yogurt", grams: 200, base: MealItemBase(grams: 200, kcal: 250, protein: 18, fat: 6, carbs: 30))
            ]
        )
        // Entry with several items (mixed shapes: multi-item entry, different macros).
        let lunch = MealLogEntry(
            createdAt: today,
            mealType: .lunch,
            imagePath: "/tmp/lunch.jpg",
            items: [
                MealItem(name: "chicken", grams: 150, base: MealItemBase(grams: 150, kcal: 250, protein: 40, fat: 6, carbs: 0)),
                MealItem(name: "rice", grams: 180, base: MealItemBase(grams: 180, kcal: 210, protein: 4, fat: 1, carbs: 45)),
                MealItem(name: "broccoli", grams: 100, base: MealItemBase(grams: 100, kcal: 35, protein: 3, fat: 0.5, carbs: 7))
            ]
        )
        // An entry from a different day must not be counted.
        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: today)!
        let stale = MealLogEntry(
            createdAt: yesterday,
            mealType: .dinner,
            imagePath: "/tmp/stale.jpg",
            items: [MealItem(name: "pizza", grams: 300, base: MealItemBase(grams: 300, kcal: 900, protein: 40, fat: 35, carbs: 100))]
        )

        try await store.save(breakfast)
        try await store.save(lunch)
        try await store.save(stale)

        let vm = AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]), store: store)
        await vm.loadEntries()

        let consumed = vm.consumedToday
        XCTAssertEqual(consumed.calories, 250 + 250 + 210 + 35, accuracy: 0.001)
        XCTAssertEqual(consumed.proteinGrams, 18 + 40 + 4 + 3, accuracy: 0.001)
        XCTAssertEqual(consumed.fatGrams, 6 + 6 + 1 + 0.5, accuracy: 0.001)
        XCTAssertEqual(consumed.carbsGrams, 30 + 0 + 45 + 7, accuracy: 0.001)
    }

    // MARK: - Macro target display values (spec §2 reference goal, 2150 kcal)

    func testMacroTargetDisplayGramsForReferenceGoal() {
        let target = GoalCalculator.macroTargets(forGoalCalories: 2150)
        // 30/25/45% of 2150 kcal via 4/9/4 kcal/g, rounded to whole grams for
        // display (matches the prototype's `Math.round(...)`):
        // protein 2150*0.30/4 = 161.25 -> 161, fat 2150*0.25/9 = 59.72 -> 60,
        // carbs 2150*0.45/4 = 241.875 -> 242.
        XCTAssertEqual(AppViewModel.roundedGrams(target.proteinGrams), 161)
        XCTAssertEqual(AppViewModel.roundedGrams(target.fatGrams), 60)
        XCTAssertEqual(AppViewModel.roundedGrams(target.carbsGrams), 242)
    }

    // MARK: - Ring/bar fraction clamp

    func testProgressFractionClampsAtOneWhenConsumedExceedsGoal() {
        XCTAssertEqual(AppViewModel.progressFraction(consumed: 3200, target: 2150), 1.0)
    }

    func testProgressFractionIsProportionalBelowGoal() {
        XCTAssertEqual(AppViewModel.progressFraction(consumed: 1075, target: 2150), 0.5, accuracy: 0.0001)
    }

    func testProgressFractionIsZeroWhenTargetIsZeroOrNegative() {
        XCTAssertEqual(AppViewModel.progressFraction(consumed: 500, target: 0), 0)
        XCTAssertEqual(AppViewModel.progressFraction(consumed: 500, target: -10), 0)
    }

    // MARK: - Kcal display formatting

    func testFormattedKcalGroupsThousands() {
        XCTAssertEqual(AppViewModel.formattedKcal(2150), "2,150")
        XCTAssertEqual(AppViewModel.formattedKcal(712), "712")
    }
}
