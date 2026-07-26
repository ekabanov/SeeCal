import XCTest
@testable import SeeCalInference

final class SeeCalMLXEngineTests: XCTestCase {
    func testGenerateDelegatesToRunner() async throws {
        let engine = SeeCalMLXEngine { imagePath, prompt in
            "{\"echo\":\"\(imagePath)::\(prompt)\"}"
        }

        let output = try await engine.generate(imagePath: "/tmp/image.jpg", prompt: "hello")
        XCTAssertEqual(output, "{\"echo\":\"/tmp/image.jpg::hello\"}")
    }
}
