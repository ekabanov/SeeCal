import Foundation
import XCTest
import SeeCalDomain
import SeeCalPersistence
@testable import SeeCalApp

private actor StubBarcodeHTTPClient: BarcodeHTTPDataLoading {
    let data: Data
    let statusCode: Int
    private(set) var requests: [URLRequest] = []

    init(json: String, statusCode: Int = 200) {
        self.data = Data(json.utf8)
        self.statusCode = statusCode
    }

    func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        requests.append(request)
        return (
            data,
            HTTPURLResponse(
                url: request.url!,
                statusCode: statusCode,
                httpVersion: nil,
                headerFields: nil
            )!
        )
    }

    var requestCount: Int { requests.count }
}

private actor CountingBarcodeLookup: BarcodeProductLookup {
    let product: BarcodeProduct
    private(set) var callCount = 0

    init(product: BarcodeProduct) {
        self.product = product
    }

    func lookup(barcode: DetectedBarcode) async throws -> BarcodeProduct {
        callCount += 1
        return product
    }
}

private final class StubRateLimitClock: @unchecked Sendable {
    private let lock = NSLock()
    private var date = Date(timeIntervalSince1970: 1_000)
    private var recordedDelays: [TimeInterval] = []

    func now() -> Date {
        lock.withLock { date }
    }

    func sleep(for delay: TimeInterval) async throws {
        lock.withLock {
            recordedDelays.append(delay)
            date = date.addingTimeInterval(delay)
        }
    }

    var delays: [TimeInterval] {
        lock.withLock { recordedDelays }
    }
}

final class BarcodeLookupTests: XCTestCase {
    func testGTINNormalizationAndCheckDigitValidation() {
        XCTAssertEqual(GTIN.normalized("3017 6204-22003"), "3017620422003")
        XCTAssertEqual(GTIN.normalized("036000291452"), "036000291452")
        XCTAssertNil(GTIN.normalized("3017620422004"))
        XCTAssertNil(GTIN.normalized("not-a-code"))
    }

    func testOpenFoodFactsV3ProductMapsNutritionServingAndProvenance() async throws {
        let client = StubBarcodeHTTPClient(json: """
        {
          "status": "success",
          "product": {
            "code": "3017620422003",
            "product_name": "Oat drink",
            "ingredients_text": "Water, oats",
            "serving_size": "250 ml",
            "serving_quantity": "250",
            "serving_quantity_unit": "ml",
            "product_quantity_unit": "ml",
            "nutriments": {
              "energy-kcal_100g": 46,
              "proteins_100g": "1.0",
              "fat_100g": 1.5,
              "carbohydrates_100g": 6.7,
              "sodium_modifier": "~",
              "nutrition-score-fr_100g": 3,
              "unrelated_object": {"value": 12}
            }
          }
        }
        """)
        let date = Date(timeIntervalSince1970: 123)
        let lookup = OpenFoodFactsProductLookup(
            session: client,
            now: { date }
        )

        let product = try await lookup.lookup(
            barcode: DetectedBarcode(value: "3017620422003", symbology: .ean13)
        )

        XCTAssertEqual(product.name, "Oat drink")
        XCTAssertEqual(product.amountUnit, .milliliters)
        XCTAssertEqual(product.defaultAmount, 250)
        XCTAssertEqual(product.nutritionPer100, BarcodeNutrition(kcal: 46, protein: 1, fat: 1.5, carbs: 6.7))
        XCTAssertEqual(product.provider, "Open Food Facts")
        XCTAssertEqual(product.lookedUpAt, date)
        let requestCount = await client.requestCount
        let requests = await client.requests
        XCTAssertEqual(requestCount, 1)
        let request = try XCTUnwrap(requests.first)
        XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), "SeeCal/0.1 (https://github.com/ekabanov/SeeCal)")
        XCTAssertTrue(request.url?.path.contains("/api/v3/product/3017620422003") == true)
    }

    func testMalformedOptionalServingValueDoesNotHideUsableNutrition() async throws {
        let client = StubBarcodeHTTPClient(json: """
        {
          "status": "success",
          "product": {
            "product_name": "Community product",
            "serving_quantity": "unknown",
            "serving_quantity_unit": "g",
            "nutriments": {
              "energy-kcal_100g": 210,
              "proteins_100g": 8,
              "fat_100g": 6,
              "carbohydrates_100g": 29,
              "sodium_modifier": "~"
            }
          }
        }
        """)

        let product = try await OpenFoodFactsProductLookup(session: client).lookup(
            barcode: DetectedBarcode(value: "3017620422003")
        )

        XCTAssertNil(product.defaultAmount)
        XCTAssertEqual(
            product.nutritionPer100,
            BarcodeNutrition(kcal: 210, protein: 8, fat: 6, carbs: 29)
        )
    }

    func testIncompleteProductIsReturnedWithoutInventingZeroNutrition() async throws {
        let client = StubBarcodeHTTPClient(json: """
        {
          "product": {
            "product_name": "Incomplete cereal",
            "product_quantity_unit": "g",
            "nutriments": {
              "energy-kcal_100g": 370,
              "proteins_100g": 8
            }
          }
        }
        """)
        let product = try await OpenFoodFactsProductLookup(session: client).lookup(
            barcode: DetectedBarcode(value: "3017620422003")
        )

        XCTAssertNil(product.nutritionPer100)
        XCTAssertNil(product.fatPer100)
        XCTAssertNil(product.carbsPer100)
        XCTAssertNil(product.mealDraft(amount: 100, mealType: .snack))
    }

    func testCachedLookupAvoidsSecondRemoteRequest() async throws {
        let product = BarcodeProduct(
            barcode: "3017620422003",
            name: "Cached",
            kcalPer100: 100,
            proteinPer100: 1,
            fatPer100: 2,
            carbsPer100: 3
        )
        let remote = CountingBarcodeLookup(product: product)
        let lookup = CachedBarcodeProductLookup(
            remote: remote,
            cache: InMemoryBarcodeProductCache()
        )
        let detected = DetectedBarcode(value: product.barcode)

        _ = try await lookup.lookup(barcode: detected)
        _ = try await lookup.lookup(barcode: detected)

        let callCount = await remote.callCount
        XCTAssertEqual(callCount, 1)
    }

    func testRemoteLookupLimiterReservesFourSecondSlots() async throws {
        let product = BarcodeProduct(
            barcode: "3017620422003",
            name: "Rate limited",
            kcalPer100: 100,
            proteinPer100: 1,
            fatPer100: 2,
            carbsPer100: 3
        )
        let remote = CountingBarcodeLookup(product: product)
        let clock = StubRateLimitClock()
        let lookup = RateLimitedBarcodeProductLookup(
            remote: remote,
            minimumIntervalSeconds: 4,
            now: clock.now,
            sleeper: clock.sleep
        )

        _ = try await lookup.lookup(barcode: DetectedBarcode(value: product.barcode))
        _ = try await lookup.lookup(barcode: DetectedBarcode(value: "036000291452"))

        let callCount = await remote.callCount
        XCTAssertEqual(callCount, 2)
        XCTAssertEqual(clock.delays, [4])
    }

    func testBarcodeMealDraftScalesAmountAndKeepsImmutableLabelReference() throws {
        let product = BarcodeProduct(
            barcode: "3017620422003",
            name: "Oat drink",
            ingredientsText: "Water, oats",
            defaultAmount: 250,
            amountUnit: .milliliters,
            kcalPer100: 46,
            proteinPer100: 1,
            fatPer100: 1.5,
            carbsPer100: 6.7
        )

        let draft = try XCTUnwrap(product.mealDraft(amount: 250, mealType: .snack))
        let entry = try draft.committedEntry()

        XCTAssertEqual(entry.origin, .barcode)
        XCTAssertNil(entry.imagePath)
        XCTAssertEqual(entry.barcodeSource?.barcode, product.barcode)
        XCTAssertEqual(entry.items[0].amountUnit, .milliliters)
        XCTAssertEqual(entry.items[0].grams, 250)
        XCTAssertEqual(entry.items[0].sourceReference?.source, .barcode)
        XCTAssertEqual(entry.totals.calories, 115, accuracy: 0.000_001)
        XCTAssertEqual(entry.totals.carbsGrams, 16.75, accuracy: 0.000_001)

        var edited = entry.items[0]
        edited.setCurrentValue(999, for: .kcal)
        edited.resetToEstimate()
        XCTAssertEqual(edited.kcal, 115, accuracy: 0.000_001)
        XCTAssertEqual(edited.grams, 250)
    }
}
