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
