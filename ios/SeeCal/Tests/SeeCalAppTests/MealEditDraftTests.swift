import XCTest
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
@testable import SeeCalApp

final class MealEditDraftTests: XCTestCase {
    // MARK: - New-scan mode

    private func makeScanResult() -> FoodScanResult {
        FoodScanResult(
            totalCalories: 500,
            proteinGrams: 30,
            fatGrams: 10,
            carbsGrams: 60,
            confidence: 0.9,
            items: [
                ScanItem(name: "chicken breast", estimatedGrams: 150, calories: 300, proteinGrams: 24, fatGrams: 6, carbsGrams: 0),
                ScanItem(name: "rice", estimatedGrams: 200, calories: 200, proteinGrams: 6, fatGrams: 4, carbsGrams: 60)
            ],
            uncertaintyFlags: []
        )
    }

    func testNewScanDraftMapsItemsFromScanResultAtBaseGrams() {
        let draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)

        XCTAssertFalse(draft.isEditingExisting)
        XCTAssertEqual(draft.items.count, 2)
        XCTAssertEqual(draft.items[0].name, "chicken breast")
        XCTAssertEqual(draft.items[0].grams, 150)
        XCTAssertEqual(draft.items[0].base.grams, 150)
        // Freshly mapped items start unscaled: current == base.
        XCTAssertEqual(draft.items[0].kcal, 300)
        XCTAssertEqual(draft.items[1].kcal, 200)
    }

    func testDraftDefaultNameFallsBackToFirstItemName() {
        let draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        XCTAssertEqual(draft.name, "Chicken Breast")
    }

    // MARK: - Scaling math

    func testStepGramsRescalesItemLinearly() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        let chickenID = draft.items[0].id

        // 150g base -> 300 kcal / 24 protein / 6 fat / 0 carbs. Step to 165g (1.1x).
        draft.stepGrams(itemID: chickenID, by: 15)

        let chicken = draft.items[0]
        XCTAssertEqual(chicken.grams, 165)
        XCTAssertEqual(chicken.kcal, 330, accuracy: 0.0001)
        XCTAssertEqual(chicken.protein, 26.4, accuracy: 0.0001)
        XCTAssertEqual(chicken.fat, 6.6, accuracy: 0.0001)
        XCTAssertEqual(chicken.carbs, 0, accuracy: 0.0001)

        // The other item is untouched.
        XCTAssertEqual(draft.items[1].grams, 200)
        XCTAssertEqual(draft.items[1].kcal, 200)
    }

    func testSetGramsRescalesDirectly() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        let riceID = draft.items[1].id

        draft.setGrams(itemID: riceID, to: 100) // half of the 200g base
        let rice = draft.items[1]
        XCTAssertEqual(rice.grams, 100)
        XCTAssertEqual(rice.kcal, 100)
        XCTAssertEqual(rice.protein, 3)
        XCTAssertEqual(rice.fat, 2)
        XCTAssertEqual(rice.carbs, 30)
    }

    // MARK: - Minimum-grams clamp (focused editor accepts positive whole grams)

    func testStepGramsNeverDropsBelowOneGramFloor() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        let riceID = draft.items[1].id

        // Step down far past zero: 200 - 5*100 would be deeply negative unclamped.
        for _ in 0..<100 {
            draft.stepGrams(itemID: riceID, by: -MealItem.gramStep)
        }

        XCTAssertEqual(draft.items[1].grams, MealItem.minimumGrams)
        XCTAssertEqual(draft.items[1].grams, 1)
    }

    func testSetGramsClampsDirectAssignmentBelowFloor() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        let chickenID = draft.items[0].id

        draft.setGrams(itemID: chickenID, to: -20)
        XCTAssertEqual(draft.items[0].grams, 1)
    }

    // MARK: - Totals derivation

    func testTotalsAreSumOfScaledItems() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        // Baseline: totals should match the unscaled scan result's own totals.
        XCTAssertEqual(draft.totals.calories, 500)
        XCTAssertEqual(draft.totals.proteinGrams, 30)
        XCTAssertEqual(draft.totals.fatGrams, 10)
        XCTAssertEqual(draft.totals.carbsGrams, 60)

        let chickenID = draft.items[0].id
        draft.stepGrams(itemID: chickenID, by: 15) // 150 -> 165, +30 kcal / +2.4 protein / +0.6 fat

        XCTAssertEqual(draft.totals.calories, 530, accuracy: 0.0001)
        XCTAssertEqual(draft.totals.proteinGrams, 32.4, accuracy: 0.0001)
        XCTAssertEqual(draft.totals.fatGrams, 10.6, accuracy: 0.0001)
        XCTAssertEqual(draft.totals.carbsGrams, 60, accuracy: 0.0001)
    }

    // MARK: - Commit: new-scan mode -> brand new entry

    func testCommitInNewScanModeProducesFreshEntry() throws {
        var draft = MealEditDraft(
            scanResult: makeScanResult(),
            imagePath: "/tmp/a.jpg",
            mealType: .lunch,
            volumeMl: 350,
            maxHeightMm: 42
        )
        draft.name = "Chicken rice bowl"
        draft.setGrams(itemID: draft.items[0].id, to: 165)

        let entry = try draft.committedEntry()

        XCTAssertEqual(entry.name, "Chicken rice bowl")
        XCTAssertEqual(entry.mealType, .lunch)
        XCTAssertEqual(entry.imagePath, "/tmp/a.jpg")
        XCTAssertEqual(entry.volumeMl, 350)
        XCTAssertEqual(entry.maxHeightMm, 42)
        XCTAssertEqual(entry.items.count, 2)
        XCTAssertEqual(entry.totals.calories, 530, accuracy: 0.0001)
    }

    func testCommitWithNoItemsThrows() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        draft.items = []
        XCTAssertThrowsError(try draft.committedEntry()) { error in
            XCTAssertEqual(error as? MealEditDraftError, .noItems)
        }
    }

    func testCommitWithWhitespaceNameThrows() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        draft.name = "   "

        XCTAssertThrowsError(try draft.committedEntry()) { error in
            XCTAssertEqual(error as? MealEditDraftError, .emptyName)
        }
    }

    func testManualMealStartsPhotoLessAndCommitsManualIngredient() throws {
        var draft = MealEditDraft(manualMealType: .snack)

        XCTAssertEqual(draft.origin, .manual)
        XCTAssertNil(draft.imagePath)
        XCTAssertTrue(draft.items.isEmpty)
        XCTAssertFalse(draft.isValid)

        draft.addItem(
            MealItem(
                name: "apple",
                grams: 150,
                base: MealItemBase(grams: 150, kcal: 78, protein: 0.4, fat: 0.3, carbs: 20),
                modelEstimate: nil
            )
        )
        let entry = try draft.committedEntry()

        XCTAssertEqual(entry.origin, .manual)
        XCTAssertNil(entry.imagePath)
        XCTAssertNil(entry.barcodeSource)
        XCTAssertTrue(entry.items[0].isManual)
        XCTAssertEqual(entry.totals.calories, 78)
    }

    func testAddRemoveRestoreAndReplaceItem() {
        var draft = MealEditDraft(scanResult: makeScanResult(), imagePath: "/tmp/a.jpg", mealType: .lunch)
        let manual = MealItem(
            name: "olive oil",
            grams: 10,
            base: MealItemBase(grams: 10, kcal: 88, protein: 0, fat: 10, carbs: 0),
            modelEstimate: nil
        )
        draft.addItem(manual)
        XCTAssertEqual(draft.items.last, manual)

        var edited = manual
        edited.name = "extra virgin olive oil"
        draft.replaceItem(edited)
        XCTAssertEqual(draft.items.last?.name, "extra virgin olive oil")

        let removed = draft.removeItem(id: edited.id)
        XCTAssertEqual(removed?.index, 2)
        XCTAssertEqual(draft.items.count, 2)

        if let removed {
            draft.restoreItem(removed.item, at: removed.index)
        }
        XCTAssertEqual(draft.items.map(\.id), [
            draft.items[0].id,
            draft.items[1].id,
            edited.id
        ])
    }

    func testReplaceFoodPreservesAmountAndResetReferenceButChangesDensity() throws {
        var draft = MealEditDraft(
            scanResult: makeScanResult(),
            imagePath: "/tmp/a.jpg",
            mealType: .lunch
        )
        let original = draft.items[0]
        let tofu = try ResolvedNutritionProfile(
            fdcID: 99,
            name: "Tofu, firm",
            category: "Legumes",
            kcalPer100g: 120,
            proteinPer100g: 13,
            fatPer100g: 7,
            carbsPer100g: 2,
            dataType: "Foundation"
        )

        draft.replaceFood(
            itemID: original.id,
            with: NutritionProfileCandidate(
                profile: tofu,
                displayName: "Tofu",
                score: 1
            )
        )

        let replaced = draft.items[0]
        XCTAssertEqual(replaced.id, original.id)
        XCTAssertEqual(replaced.name, "Tofu")
        XCTAssertEqual(replaced.grams, 150)
        XCTAssertEqual(replaced.base.grams, 100)
        XCTAssertEqual(replaced.kcal, 180, accuracy: 0.0001)
        XCTAssertEqual(replaced.protein, 19.5, accuracy: 0.0001)
        XCTAssertEqual(replaced.sourceReference, original.sourceReference)
        XCTAssertTrue(replaced.isEdited)

        var reset = replaced
        reset.resetToEstimate()
        XCTAssertEqual(reset.name, "chicken breast")
        XCTAssertEqual(reset.kcal, 300, accuracy: 0.0001)
    }

    // MARK: - Commit: edit mode -> updates the existing entry in place

    func testCommitInEditModeUpdatesExistingEntry() throws {
        let existing = MealLogEntry(
            id: UUID(),
            name: "Yesterday's lunch",
            mealType: .lunch,
            imagePath: "/tmp/existing.jpg",
            items: [MealItem(name: "salmon", grams: 180, base: MealItemBase(grams: 180, kcal: 360, protein: 34, fat: 20, carbs: 0))],
            volumeMl: 280,
            maxHeightMm: 30
        )

        var draft = MealEditDraft(entry: existing)
        XCTAssertTrue(draft.isEditingExisting)
        XCTAssertEqual(draft.name, "Yesterday's lunch")
        XCTAssertEqual(draft.items, existing.items)

        draft.setGrams(itemID: draft.items[0].id, to: 90) // half -> 180 kcal
        draft.name = "Yesterday's lunch (smaller)"

        let updated = try draft.committedEntry()

        // Identity, photo, and depth metadata are preserved from the original entry.
        XCTAssertEqual(updated.id, existing.id)
        XCTAssertEqual(updated.imagePath, existing.imagePath)
        XCTAssertEqual(updated.volumeMl, existing.volumeMl)
        XCTAssertEqual(updated.maxHeightMm, existing.maxHeightMm)
        // Name and items reflect the edit.
        XCTAssertEqual(updated.name, "Yesterday's lunch (smaller)")
        XCTAssertEqual(updated.items[0].grams, 90)
        XCTAssertEqual(updated.totals.calories, 180, accuracy: 0.0001)
    }
}
