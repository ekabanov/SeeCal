import Foundation
import SeeCalDiagnostics

public actor SeeCalMLXEngine {
    private let runner: MLXSwiftQwenVisionEngine.Runner

    public init(runner: @escaping MLXSwiftQwenVisionEngine.Runner) {
        self.runner = runner
    }

    public static func make(config: QwenRuntimeConfig) async throws -> SeeCalMLXEngine {
        do {
            let runner = try await MLXQwen35RunnerBuilder.makeRunner(config: config)
            SeeCalDiagnostics.record(
                .notice,
                category: "model_load",
                name: "mlx_engine_initialized"
            )
            return SeeCalMLXEngine(runner: runner)
        } catch {
            SeeCalDiagnostics.record(
                .fault,
                category: "model_load",
                name: "mlx_engine_initialization_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
            throw error
        }
    }

    public func generate(imagePath: String, prompt: String) async throws -> String {
        try await runner(imagePath, prompt)
    }
}
