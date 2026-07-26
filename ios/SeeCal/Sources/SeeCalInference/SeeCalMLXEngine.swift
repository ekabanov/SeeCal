import Foundation
import os

public actor SeeCalMLXEngine {
    private let runner: MLXSwiftQwenVisionEngine.Runner
    private static let logger = Logger(subsystem: "SeeCal", category: "MLXEngine")

    public init(runner: @escaping MLXSwiftQwenVisionEngine.Runner) {
        self.runner = runner
    }

    public static func make(config: QwenRuntimeConfig) async throws -> SeeCalMLXEngine {
        do {
            let runner = try await MLXQwen35RunnerBuilder.makeRunner(config: config)
            logger.log("MLX engine initialized for model id: \(config.modelPath, privacy: .public)")
            return SeeCalMLXEngine(runner: runner)
        } catch {
            logger.error("MLX engine init failed for model id: \(config.modelPath, privacy: .public). Error: \(String(describing: error), privacy: .public)")
            throw error
        }
    }

    public func generate(imagePath: String, prompt: String) async throws -> String {
        try await runner(imagePath, prompt)
    }
}
