import SeeCalDomain
import XCTest
@testable import SeeCalInference

private actor PromptCapturingEngine: NativeQwenVisionEngine {
    private(set) var prompt: String?

    func run(imagePath: String, prompt: String) async throws -> String {
        self.prompt = prompt
        return """
        {"total_calories":100,"protein_g":10,"fat_g":2,"carbs_g":8,
         "items":[{"name":"egg","estimated_grams":50,"calories":100,
         "protein_g":10,"fat_g":2,"carbs_g":8}]}
        """
    }
}

private struct StubVisualSpecialist: VisualSpecialistPredicting {
    let prediction: VisualSpecialistPrediction

    func predict(imagePath: String) async throws -> VisualSpecialistPrediction {
        _ = imagePath
        return prediction
    }
}

private struct FailingVisualSpecialist: VisualSpecialistPredicting {
    func predict(imagePath: String) async throws -> VisualSpecialistPrediction {
        throw VisualSpecialistError.imageNotFound(imagePath)
    }
}

final class VisualSpecialistTests: XCTestCase {
    private let prediction = VisualSpecialistPrediction(
        massG: .init(estimate: 123.04, low: 80, high: 170.06),
        calories: .init(estimate: 321, low: 210.04, high: 450),
        proteinG: .init(estimate: 17.26, low: 9, high: 25),
        fatG: .init(estimate: 11, low: 4, high: 19),
        carbsG: .init(estimate: 38.94, low: 22, high: 58)
    )

    func testRendererMatchesPythonCanonicalFieldOrderAndRounding() {
        XCTAssertEqual(
            VisualSpecialistPromptRenderer.render(prediction),
            """
            Auxiliary visual measurement (fallible; use as evidence, not ground truth):
            {"available":true,"mass_g":{"estimate":123.0,"low":80.0,"high":170.1},"calories":{"estimate":321.0,"low":210.0,"high":450.0},"protein_g":{"estimate":17.3,"low":9.0,"high":25.0},"fat_g":{"estimate":11.0,"low":4.0,"high":19.0},"carbs_g":{"estimate":38.9,"low":22.0,"high":58.0}}
            """
        )
    }

    func testUnavailableRendererMatchesTrainedFallback() {
        XCTAssertEqual(
            VisualSpecialistPromptRenderer.render(nil),
            """
            Auxiliary visual measurement (fallible; use as evidence, not ground truth):
            {"available":false}
            """
        )
    }

    func testConditionedRuntimeAppendsPredictionToLegacyPrompt() async throws {
        let engine = PromptCapturingEngine()
        let runtime = MLXQwenRuntime(
            engine: engine,
            visualSpecialist: StubVisualSpecialist(prediction: prediction),
            includeVisualSpecialistBlock: true
        )

        _ = try await runtime.infer(
            request: FoodScanRequest(imagePath: "/tmp/photo.jpg", mealType: .lunch)
        )

        let capturedPrompt = await engine.prompt
        let prompt = try XCTUnwrap(capturedPrompt)
        XCTAssertTrue(prompt.hasSuffix("\n\n" + VisualSpecialistPromptRenderer.render(prediction)))
        XCTAssertEqual(
            prompt.components(separatedBy: VisualSpecialistPromptRenderer.prefix).count,
            2
        )
    }

    func testConditionedRuntimeUsesUnavailableBlockWhenSpecialistFails() async throws {
        let engine = PromptCapturingEngine()
        let runtime = MLXQwenRuntime(
            engine: engine,
            visualSpecialist: FailingVisualSpecialist(),
            includeVisualSpecialistBlock: true
        )

        _ = try await runtime.infer(
            request: FoodScanRequest(imagePath: "/tmp/missing.jpg", mealType: .lunch)
        )

        let capturedPrompt = await engine.prompt
        let prompt = try XCTUnwrap(capturedPrompt)
        XCTAssertTrue(prompt.hasSuffix("\n\n" + VisualSpecialistPromptRenderer.render(nil)))
    }

    func testLegacyRuntimeDoesNotChangePrompt() {
        let request = FoodScanRequest(imagePath: "/tmp/photo.jpg", mealType: .lunch)
        let prompt = QwenPromptBuilder().buildPrompt(request: request)
        XCTAssertFalse(prompt.contains(VisualSpecialistPromptRenderer.prefix))
    }

    func testCoreMLSpecialistIntegrationScaffold() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["SEECAL_SPECIALIST_MODEL"],
              let imagePath = environment["SEECAL_SPECIALIST_IMAGE"] else {
            throw XCTSkip("Set SEECAL_SPECIALIST_MODEL and SEECAL_SPECIALIST_IMAGE to run")
        }

        let specialist = try CoreMLVisualSpecialist(modelPath: modelPath)
        let result = try await specialist.predict(imagePath: imagePath)
        for interval in [
            result.massG,
            result.calories,
            result.proteinG,
            result.fatG,
            result.carbsG,
        ] {
            XCTAssertGreaterThanOrEqual(interval.estimate, 0)
            XCTAssertGreaterThanOrEqual(interval.low, 0)
            XCTAssertGreaterThanOrEqual(interval.high, interval.low)
        }
    }

    func testCoreMLSpecialistParityScaffold() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["SEECAL_SPECIALIST_MODEL"],
              let predictionsPath = environment["SEECAL_SPECIALIST_PREDICTIONS"],
              let mlRoot = environment["SEECAL_ML_ROOT"] else {
            throw XCTSkip("Set SEECAL_SPECIALIST_MODEL, SEECAL_SPECIALIST_PREDICTIONS, and SEECAL_ML_ROOT to run")
        }

        let content = try String(contentsOfFile: predictionsPath, encoding: .utf8)
        let specialist = try CoreMLVisualSpecialist(modelPath: modelPath)
        var relativeErrors: [Double] = []
        var calorieRelativeErrors: [Double] = []

        for line in content.split(separator: "\n") {
            let data = try XCTUnwrap(String(line).data(using: .utf8))
            let row = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data) as? [String: Any]
            )
            let id = try XCTUnwrap(row["id"] as? String)
            let parts = id.split(separator: ":")
            guard parts.count == 3, parts[0] == "nutrition5k" else { continue }
            let imagePath = URL(fileURLWithPath: mlRoot, isDirectory: true)
                .appendingPathComponent("dataset_clean/\(parts[1])/overhead.jpg")
                .path
            let expectedNumeric = try XCTUnwrap(row["numeric"] as? [String: Any])
            let actual = try await specialist.predict(imagePath: imagePath)
            let fields: [(String, Double)] = [
                ("mass_g", actual.massG.estimate),
                ("calories", actual.calories.estimate),
                ("protein_g", actual.proteinG.estimate),
                ("fat_g", actual.fatG.estimate),
                ("carbs_g", actual.carbsG.estimate),
            ]
            for (name, value) in fields {
                let field = try XCTUnwrap(expectedNumeric[name] as? [String: Any])
                let expected = try XCTUnwrap(field["p50"] as? Double)
                let relative = abs(value - expected) / max(expected, 1)
                relativeErrors.append(relative)
                if name == "calories" { calorieRelativeErrors.append(relative) }
            }
            if calorieRelativeErrors.count == 50 { break }
        }

        XCTAssertEqual(calorieRelativeErrors.count, 50)
        let meanRelative = relativeErrors.reduce(0, +) / Double(relativeErrors.count)
        let meanCalorieRelative =
            calorieRelativeErrors.reduce(0, +) / Double(calorieRelativeErrors.count)
        print(
            "[VisualSpecialistTests] 50-image Core ML parity: all=\(meanRelative), calories=\(meanCalorieRelative)"
        )
        XCTAssertLessThan(meanRelative, 0.25)
        XCTAssertLessThan(meanCalorieRelative, 0.20)
    }
}
