import Foundation
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

    public func buildPrompt(request: FoodScanRequest) -> String {
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

        return prompt
    }
}

public struct MLXQwenRuntime: InferenceRuntime {
    public let name = "mlx_swift"
    public let modelFamily = "qwen3.5-native-multimodal"

    private let engine: NativeQwenVisionEngine
    private let promptBuilder: QwenPromptBuilder

    public init(engine: NativeQwenVisionEngine, promptBuilder: QwenPromptBuilder = QwenPromptBuilder()) {
        self.engine = engine
        self.promptBuilder = promptBuilder
    }

    public func isAvailable() async -> Bool {
        true
    }

    public func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        let prompt = promptBuilder.buildPrompt(request: request)
        let raw = try await engine.run(imagePath: request.imagePath, prompt: prompt)
        do {
            return try ScanJSONParser.parseStrict(from: raw)
        } catch {
            throw InferenceError.parsingFailed(error.localizedDescription)
        }
    }
}

public struct MNNQwenRuntime: InferenceRuntime {
    public let name = "mnn"
    public let modelFamily = "qwen3.5-native-multimodal"

    private let engine: NativeQwenVisionEngine
    private let promptBuilder: QwenPromptBuilder

    public init(engine: NativeQwenVisionEngine, promptBuilder: QwenPromptBuilder = QwenPromptBuilder()) {
        self.engine = engine
        self.promptBuilder = promptBuilder
    }

    public func isAvailable() async -> Bool {
        true
    }

    public func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        let prompt = promptBuilder.buildPrompt(request: request)
        let raw = try await engine.run(imagePath: request.imagePath, prompt: prompt)
        do {
            return try ScanJSONParser.parseStrict(from: raw)
        } catch {
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
