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

    func testLazyEngineDoesNotLoadUntilFirstGenerateAndReusesRunner() async throws {
        let loads = IntRecorder()
        let states = LoadStateRecorder()
        let config = QwenRuntimeConfig(modelPath: "test-model")
        let engine = SeeCalMLXEngine(
            config: config,
            loadStateObserver: { state in
                await states.append(state)
            },
            loader: { loadedConfig in
                await loads.increment()
                XCTAssertEqual(loadedConfig.modelPath, "test-model")
                return { imagePath, prompt in "\(imagePath)::\(prompt)" }
            }
        )

        let loadsBeforeGenerate = await loads.value
        let statesBeforeGenerate = await states.values
        XCTAssertEqual(loadsBeforeGenerate, 0)
        XCTAssertEqual(statesBeforeGenerate, [])

        async let firstRequest = engine.generate(imagePath: "one.jpg", prompt: "first")
        async let secondRequest = engine.generate(imagePath: "two.jpg", prompt: "second")
        let (first, second) = try await (firstRequest, secondRequest)

        XCTAssertEqual(first, "one.jpg::first")
        XCTAssertEqual(second, "two.jpg::second")
        let loadsAfterGenerate = await loads.value
        let statesAfterGenerate = await states.values
        XCTAssertEqual(loadsAfterGenerate, 1)
        XCTAssertEqual(statesAfterGenerate, [.loading, .ready])
    }
}

private actor IntRecorder {
    private(set) var value = 0

    func increment() {
        value += 1
    }
}

private actor LoadStateRecorder {
    private(set) var values: [MLXModelLoadState] = []

    func append(_ state: MLXModelLoadState) {
        values.append(state)
    }
}
