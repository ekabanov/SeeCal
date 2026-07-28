import XCTest
@testable import SeeCalDomain

final class MealItemTests: XCTestCase {
    func testCorrectedCaloriesBecomeDensityAndScaleWithGrams() {
        var item = MealItem(
            name: "rice",
            grams: 180,
            base: MealItemBase(grams: 180, kcal: 234, protein: 4.3, fat: 0.4, carbs: 50.9)
        )

        item.setCurrentValue(200, for: .kcal)
        item.setGrams(360)

        XCTAssertEqual(item.kcal, 400, accuracy: 0.000_001)
        XCTAssertEqual(item.protein, 8.6, accuracy: 0.000_001)
        XCTAssertTrue(item.isEdited)
    }

    func testCorrectingOneFieldDoesNotRewriteOtherNutrition() {
        var item = MealItem(
            name: "meal",
            grams: 100,
            base: MealItemBase(grams: 100, kcal: 250, protein: 20, fat: 8, carbs: 30)
        )

        item.setCurrentValue(35, for: .protein)

        XCTAssertEqual(item.kcal, 250, accuracy: 0.000_001)
        XCTAssertEqual(item.protein, 35, accuracy: 0.000_001)
        XCTAssertEqual(item.fat, 8, accuracy: 0.000_001)
        XCTAssertEqual(item.carbs, 30, accuracy: 0.000_001)
    }

    func testPerFieldAndWholeResetSemantics() {
        var item = MealItem(
            name: "salmon",
            grams: 150,
            base: MealItemBase(grams: 150, kcal: 312, protein: 37.5, fat: 20, carbs: 0)
        )
        item.name = "fish"
        item.setCurrentValue(250, for: .kcal)
        item.setGrams(300)

        item.resetNutritionToEstimate(.kcal)
        XCTAssertEqual(item.kcal, 624, accuracy: 0.000_001)
        XCTAssertEqual(item.name, "fish")

        item.setCurrentValue(500, for: .kcal)
        item.resetGramsToEstimate()
        XCTAssertEqual(item.grams, 150)
        XCTAssertEqual(item.kcal, 250, accuracy: 0.000_001)

        item.resetToEstimate()
        XCTAssertEqual(item.name, "salmon")
        XCTAssertEqual(item.grams, 150)
        XCTAssertEqual(item.kcal, 312, accuracy: 0.000_001)
        XCTAssertFalse(item.isEdited)
    }

    func testManualItemRoundTripsWithoutModelEstimate() throws {
        let item = MealItem(
            name: "olive oil",
            grams: 10,
            base: MealItemBase(grams: 10, kcal: 88, protein: 0, fat: 10, carbs: 0),
            modelEstimate: nil
        )

        let data = try JSONEncoder().encode(item)
        let decoded = try JSONDecoder().decode(MealItem.self, from: data)

        XCTAssertTrue(decoded.isManual)
        XCTAssertNil(decoded.modelEstimate)
        XCTAssertEqual(decoded, item)
    }

    func testCorrectedModelBackedItemRoundTripsWithOriginalEstimate() throws {
        var item = MealItem(
            name: "rice",
            grams: 180,
            base: MealItemBase(grams: 180, kcal: 234, protein: 4.3, fat: 0.4, carbs: 50.9)
        )
        item.name = "brown rice"
        item.setCurrentValue(200, for: .kcal)

        let decoded = try JSONDecoder().decode(MealItem.self, from: JSONEncoder().encode(item))

        XCTAssertEqual(decoded.name, "brown rice")
        XCTAssertEqual(decoded.kcal, 200, accuracy: 0.000_001)
        XCTAssertEqual(decoded.modelEstimate?.name, "rice")
        XCTAssertEqual(decoded.modelEstimate?.base.kcal, 234)
        XCTAssertTrue(decoded.isEdited)
    }

    func testOldSchemaSynthesizesModelEstimate() throws {
        let json = """
        {
          "id": "11111111-1111-1111-1111-111111111111",
          "name": "legacy",
          "grams": 125,
          "base": {"grams": 125, "kcal": 300, "protein": 20, "fat": 10, "carbs": 30}
        }
        """

        let item = try JSONDecoder().decode(MealItem.self, from: Data(json.utf8))

        XCTAssertEqual(item.modelEstimate?.name, "legacy")
        XCTAssertEqual(item.modelEstimate?.base.kcal, 300)
        XCTAssertFalse(item.isEdited)
    }
}
