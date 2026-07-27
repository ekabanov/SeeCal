import Foundation
import XCTest
@testable import SeeCalDomain

final class FoodScanResultTests: XCTestCase {
    func testParsesValidJSONFixture() throws {
        let data = try fixture(named: "valid_scan_result.json")
        let parsed = try ScanJSONParser.parseStrict(from: data)

        XCTAssertEqual(parsed.items.count, 2)
        XCTAssertEqual(parsed.totalCalories, 512)
        XCTAssertEqual(parsed.confidence, 0.91)
    }

    func testRejectsMissingItemsFixture() throws {
        let data = try fixture(named: "invalid_scan_result_missing_items.json")

        XCTAssertThrowsError(try ScanJSONParser.parseStrict(from: data)) { error in
            XCTAssertEqual(error as? ScanValidationError, .missingItems)
        }
    }

    // MARK: - v7 not-food refusal detection

    func testDetectsNotFoodRefusal() {
        XCTAssertTrue(ScanJSONParser.isNotFood(#"{"not_food": true}"#))
        // Tolerant of surrounding prose / whitespace.
        XCTAssertTrue(ScanJSONParser.isNotFood("\n {\"not_food\": true} \n"))
        XCTAssertTrue(ScanJSONParser.isNotFood(#"Sure: {"not_food": true}"#))
    }

    func testFoodResultIsNotDetectedAsRefusal() throws {
        let data = try fixture(named: "valid_scan_result.json")
        let json = String(decoding: data, as: UTF8.self)
        XCTAssertFalse(ScanJSONParser.isNotFood(json))
        // not_food:false and a missing key are both "this is food".
        XCTAssertFalse(ScanJSONParser.isNotFood(#"{"not_food": false, "total_calories": 100}"#))
        XCTAssertFalse(ScanJSONParser.isNotFood(#"{"total_calories": 100, "items": []}"#))
    }

    private func fixture(named: String) throws -> Data {
        #if SWIFT_PACKAGE
        let url = Bundle.module.url(forResource: named.replacingOccurrences(of: ".json", with: ""), withExtension: "json")
        #else
        let url = Bundle(for: FoodScanResultTests.self).url(forResource: named.replacingOccurrences(of: ".json", with: ""), withExtension: "json")
        #endif
        guard let url else {
            throw NSError(domain: "FoodScanResultTests", code: 1)
        }
        return try Data(contentsOf: url)
    }
}
