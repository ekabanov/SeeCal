import Foundation
import SeeCalDiagnostics
import SeeCalDomain

public protocol NativeQwenVisionEngine: Sendable {
    func run(imagePath: String, prompt: String) async throws -> String
}

public struct MLXSwiftQwenVisionEngine: NativeQwenVisionEngine {
    public typealias Runner = @Sendable (_ imagePath: String, _ prompt: String) async throws -> String
    private let runner: Runner

    public init(runner: @escaping Runner) {
        self.runner = runner
    }

    public func run(imagePath: String, prompt: String) async throws -> String {
        try await runner(imagePath, prompt)
    }
}

public struct MNNQwenVisionEngine: NativeQwenVisionEngine {
    public typealias Runner = @Sendable (_ imagePath: String, _ prompt: String) async throws -> String
    private let runner: Runner

    public init(runner: @escaping Runner) {
        self.runner = runner
    }

    public func run(imagePath: String, prompt: String) async throws -> String {
        try await runner(imagePath, prompt)
    }
}

public struct QwenPromptBuilder: Sendable {
    public init() {}

    public func buildPrompt(
        request: FoodScanRequest,
        visualSpecialistPrediction: VisualSpecialistPrediction? = nil,
        includeVisualSpecialistBlock: Bool = false
    ) -> String {
        // Must match 02_prepare_finetune.py exactly (SYSTEM_PROMPT + "\n\n" + USER_PROMPT)
        // so the fine-tuned model sees the same conditioning it trained on.
        var prompt = """
        You are a nutrition expert. When shown a photo of a meal, \
        you identify the ingredients with their weights and estimate the total \
        nutritional content with high accuracy. Always respond with a valid JSON object.

        Look at this meal and identify its ingredients and nutritional content. \
        Provide your answer as a JSON object with the following keys: \
        total_calories (kcal), protein_g, fat_g, carbs_g, and items \
        (a list of objects with name, estimated_grams, calories, protein_g, fat_g, carbs_g, \
        sorted by weight descending).
        """

        if let hint = request.userHint, !hint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            prompt += "\nUser hint: \(hint)"
        }
        if includeVisualSpecialistBlock {
            prompt += "\n\n" + VisualSpecialistPromptRenderer.render(visualSpecialistPrediction)
        }

        return prompt
    }
}

public struct MLXQwenRuntime: InferenceRuntime {
    public let name = "mlx_swift"
    public let modelFamily = "qwen3.5-native-multimodal"

    private let engine: NativeQwenVisionEngine
    private let promptBuilder: QwenPromptBuilder
    private let visualSpecialist: (any VisualSpecialistPredicting)?
    private let includeVisualSpecialistBlock: Bool

    public init(
        engine: NativeQwenVisionEngine,
        promptBuilder: QwenPromptBuilder = QwenPromptBuilder(),
        visualSpecialist: (any VisualSpecialistPredicting)? = nil,
        includeVisualSpecialistBlock: Bool = false
    ) {
        self.engine = engine
        self.promptBuilder = promptBuilder
        self.visualSpecialist = visualSpecialist
        self.includeVisualSpecialistBlock = includeVisualSpecialistBlock
    }

    public func isAvailable() async -> Bool {
        true
    }

    public func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        let measurement: VisualSpecialistPrediction?
        if includeVisualSpecialistBlock, let visualSpecialist {
            do {
                measurement = try await visualSpecialist.predict(imagePath: request.imagePath)
            } catch {
                SeeCalDiagnostics.record(
                    .error,
                    category: "inference",
                    name: "visual_specialist_prediction_failed",
                    fields: ["runtime": name]
                        .merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
                )
                measurement = nil
            }
        } else {
            measurement = nil
        }
        let prompt = promptBuilder.buildPrompt(
            request: request,
            visualSpecialistPrediction: measurement,
            includeVisualSpecialistBlock: includeVisualSpecialistBlock
        )
        let raw = try await engine.run(imagePath: request.imagePath, prompt: prompt)
        SeeCalDiagnostics.record(
            .info,
            category: "inference",
            name: "generation_received",
            fields: ["runtime": name, "character_count": String(raw.count)]
        )
        // v7: a not-food refusal (`{"not_food": true}`) is a definitive answer,
        // checked before parseStrict (which would throw missingItems on it).
        if ScanJSONParser.isNotFood(raw) { throw InferenceError.notFood }
        do {
            let result = try ScanJSONParser.parseStrict(from: raw)
            SeeCalDiagnostics.record(
                .info,
                category: "inference",
                name: "generation_parsed",
                fields: ["runtime": name]
            )
            return result
        } catch {
            SeeCalDiagnostics.record(
                .error,
                category: "inference",
                name: "generation_parse_failed",
                fields: ["runtime": name]
                    .merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
            )
            throw InferenceError.parsingFailed(error.localizedDescription)
        }
    }
}

public struct MNNQwenRuntime: InferenceRuntime {
    public let name = "mnn"
    public let modelFamily = "qwen3.5-native-multimodal"

    private let engine: NativeQwenVisionEngine
    private let promptBuilder: QwenPromptBuilder
    private let visualSpecialist: (any VisualSpecialistPredicting)?
    private let includeVisualSpecialistBlock: Bool

    public init(
        engine: NativeQwenVisionEngine,
        promptBuilder: QwenPromptBuilder = QwenPromptBuilder(),
        visualSpecialist: (any VisualSpecialistPredicting)? = nil,
        includeVisualSpecialistBlock: Bool = false
    ) {
        self.engine = engine
        self.promptBuilder = promptBuilder
        self.visualSpecialist = visualSpecialist
        self.includeVisualSpecialistBlock = includeVisualSpecialistBlock
    }

    public func isAvailable() async -> Bool {
        true
    }

    public func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        let measurement: VisualSpecialistPrediction?
        if includeVisualSpecialistBlock, let visualSpecialist {
            do {
                measurement = try await visualSpecialist.predict(imagePath: request.imagePath)
            } catch {
                SeeCalDiagnostics.record(
                    .error,
                    category: "inference",
                    name: "visual_specialist_prediction_failed",
                    fields: ["runtime": name]
                        .merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
                )
                measurement = nil
            }
        } else {
            measurement = nil
        }
        let prompt = promptBuilder.buildPrompt(
            request: request,
            visualSpecialistPrediction: measurement,
            includeVisualSpecialistBlock: includeVisualSpecialistBlock
        )
        let raw = try await engine.run(imagePath: request.imagePath, prompt: prompt)
        SeeCalDiagnostics.record(
            .info,
            category: "inference",
            name: "generation_received",
            fields: ["runtime": name, "character_count": String(raw.count)]
        )
        if ScanJSONParser.isNotFood(raw) { throw InferenceError.notFood }
        do {
            let result = try ScanJSONParser.parseStrict(from: raw)
            SeeCalDiagnostics.record(
                .info,
                category: "inference",
                name: "generation_parsed",
                fields: ["runtime": name]
            )
            return result
        } catch {
            SeeCalDiagnostics.record(
                .error,
                category: "inference",
                name: "generation_parse_failed",
                fields: ["runtime": name]
                    .merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
            )
            throw InferenceError.parsingFailed(error.localizedDescription)
        }
    }
}

public struct CoreMLSpikeRuntime: InferenceRuntime {
    public let name = "coreml_spike"
    public let modelFamily = "qwen3.5-native-multimodal"

    public init() {}

    public func isAvailable() async -> Bool {
        false
    }

    public func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        throw InferenceError.runtimeUnavailable("Core ML runtime is R&D only and not on the critical path")
    }
}
