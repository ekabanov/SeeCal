import XCTest
import SeeCalDomain
@testable import SeeCalInference

private struct StubEngine: NativeQwenVisionEngine {
    let output: String

    func run(imagePath: String, prompt: String) async throws -> String {
        _ = imagePath
        _ = prompt
        return output
    }
}

private struct UnavailableRuntime: InferenceRuntime {
    let name = "unavailable"
    let modelFamily = "qwen3.5-native-multimodal"

    func isAvailable() async -> Bool { false }
    func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        throw InferenceError.runtimeUnavailable(name)
    }
}

private struct SlowRuntime: InferenceRuntime {
    let name = "slow"
    let modelFamily = "qwen3.5-native-multimodal"

    func isAvailable() async -> Bool { true }

    func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        _ = request
        try await Task.sleep(nanoseconds: 2_000_000_000)
        throw InferenceError.runtimeFailed("slow failure")
    }
}

final class RuntimeOrchestratorTests: XCTestCase {
    func testFallsBackToSecondRuntime() async throws {
        let validJSON = """
        {
          "total_calories": 400,
          "protein_g": 20,
          "fat_g": 10,
          "carbs_g": 50,
          "confidence": 0.8,
          "items": [{"name": "meal", "estimated_grams": 300, "calories": 400, "protein_g": 20, "fat_g": 10, "carbs_g": 50}],
          "uncertainty_flags": []
        }
        """

        let primary = UnavailableRuntime()
        let fallback = MLXQwenRuntime(engine: StubEngine(output: validJSON))
        let orchestrator = RuntimeOrchestrator(runtimes: [primary, fallback], timeoutNanoseconds: 1_000_000_000)

        let result = try await orchestrator.infer(request: FoodScanRequest(imagePath: "/tmp/a.jpg", mealType: .lunch))

        XCTAssertEqual(result.totalCalories, 400)
        XCTAssertEqual(result.items.count, 1)
    }

    func testNotFoodShortCircuitsWithoutFallthrough() async throws {
        // A refusal from the primary is a DEFINITIVE answer: the orchestrator
        // must rethrow .notFood as-is, never fall through to a runtime that
        // would hallucinate food. The fallback returns valid food JSON; if it
        // were consulted, infer() would return instead of throwing.
        let foodJSON = """
        {"total_calories": 400, "protein_g": 20, "fat_g": 10, "carbs_g": 50,
         "items": [{"name": "meal", "estimated_grams": 300, "calories": 400, "protein_g": 20, "fat_g": 10, "carbs_g": 50}]}
        """
        let primary = MLXQwenRuntime(engine: StubEngine(output: #"{"not_food": true}"#))
        let fallback = MLXQwenRuntime(engine: StubEngine(output: foodJSON))
        let orchestrator = RuntimeOrchestrator(runtimes: [primary, fallback], timeoutNanoseconds: 1_000_000_000)

        do {
            _ = try await orchestrator.infer(request: FoodScanRequest(imagePath: "/tmp/a.jpg", mealType: .lunch))
            XCTFail("Expected .notFood to be rethrown, not a fallback food result")
        } catch InferenceError.notFood {
            // expected
        }
    }

    func testTimesOutAndFallsBack() async throws {
        let validJSON = """
        {
          "total_calories": 320,
          "protein_g": 21,
          "fat_g": 9,
          "carbs_g": 30,
          "confidence": 0.77,
          "items": [{"name": "meal", "estimated_grams": 200, "calories": 320, "protein_g": 21, "fat_g": 9, "carbs_g": 30}],
          "uncertainty_flags": []
        }
        """

        let slow = SlowRuntime()
        let fallback = MLXQwenRuntime(engine: StubEngine(output: validJSON))
        let orchestrator = RuntimeOrchestrator(
            runtimes: [slow, fallback],
            timeoutNanoseconds: 100_000_000,
            maxAttemptsPerRuntime: 1
        )

        let result = try await orchestrator.infer(request: FoodScanRequest(imagePath: "/tmp/a.jpg", mealType: .dinner))
        XCTAssertEqual(result.totalCalories, 320)
    }
}
