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
