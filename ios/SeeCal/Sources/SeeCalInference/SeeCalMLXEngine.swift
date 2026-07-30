import Foundation
import SeeCalDiagnostics

public enum MLXModelLoadState: Equatable, Sendable {
    case notStarted
    case loading
    case ready
}

public actor SeeCalMLXEngine {
    public typealias Runner = MLXSwiftQwenVisionEngine.Runner
    public typealias RunnerLoader = @Sendable (QwenRuntimeConfig) async throws -> Runner
    public typealias LoadStateObserver = @Sendable (MLXModelLoadState) async -> Void

    private let config: QwenRuntimeConfig?
    private let loader: RunnerLoader?
    private let loadStateObserver: LoadStateObserver?
    private var runner: Runner?
    private var loadTask: Task<Runner, Error>?

    /// Creates an already-loaded engine. Kept as the lightweight seam used by
    /// tests and callers that manage their own runner lifecycle.
    public init(runner: @escaping Runner) {
        self.config = nil
        self.loader = nil
        self.loadStateObserver = nil
        self.runner = runner
    }

    /// Creates an engine without allocating the model. The first `generate`
    /// call performs model load, adapter fusion, and vision warmup as one
    /// single-flight task; subsequent calls reuse the warmed runner.
    public init(
        config: QwenRuntimeConfig,
        loadStateObserver: LoadStateObserver? = nil,
        loader: @escaping RunnerLoader = MLXQwen35RunnerBuilder.makeRunner
    ) {
        self.config = config
        self.loader = loader
        self.loadStateObserver = loadStateObserver
        self.runner = nil
    }

    /// Eager construction remains available for non-app clients. Production
    /// app startup deliberately uses the lazy initializer above.
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
        let runner = try await loadedRunner()
        return try await runner(imagePath, prompt)
    }

    private func loadedRunner() async throws -> Runner {
        if let runner {
            return runner
        }
        if let loadTask {
            return try await loadTask.value
        }
        guard let config, let loader else {
            preconditionFailure("Lazy MLX engine is missing its loader configuration")
        }

        let startedAt = DispatchTime.now().uptimeNanoseconds
        SeeCalDiagnostics.record(
            .notice,
            category: "model_load",
            name: "mlx_engine_initialization_started",
            fields: ["adapter_configured": String(config.adapterPath != nil)]
        )

        let task = Task {
            try await loader(config)
        }
        // Publish the task before the first suspension so actor reentrancy can
        // never start a second load while the observer hops to the main actor.
        loadTask = task
        await loadStateObserver?(.loading)

        do {
            let loadedRunner = try await task.value
            runner = loadedRunner
            loadTask = nil
            await loadStateObserver?(.ready)
            SeeCalDiagnostics.record(
                .notice,
                category: "model_load",
                name: "mlx_engine_initialized",
                fields: [
                    "duration_ms": String(
                        (DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000
                    )
                ]
            )
            return loadedRunner
        } catch {
            loadTask = nil
            await loadStateObserver?(.notStarted)
            SeeCalDiagnostics.record(
                .fault,
                category: "model_load",
                name: "mlx_engine_initialization_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
            throw error
        }
    }
}
