import Foundation
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence

public enum ProductionFactoryError: Error, Equatable, CustomStringConvertible {
    case mnnRunnerRequiredForPolicy

    public var description: String {
        switch self {
        case .mnnRunnerRequiredForPolicy:
            return "runtimePolicy requires an MNN runner, but none was provided"
        }
    }
}

public enum SeeCalProductionFactory {
    @MainActor
    public static func makeViewModel(
        config: QwenRuntimeConfig,
        mlxRunner: @escaping MLXSwiftQwenVisionEngine.Runner,
        mnnRunner: MNNQwenVisionEngine.Runner? = nil,
        visualSpecialist: (any VisualSpecialistPredicting)? = nil,
        store: MealLogStore = FileBackedMealLogStore(),
        preferencesStore: UserPreferencesStore = FileBackedUserPreferencesStore(),
        weightStore: WeightLogStore = FileBackedWeightLogStore()
    ) throws -> AppViewModel {
        let validatedConfig = try config.validated()
        let conditioned = validatedConfig.visualSpecialistModelPath != nil
        let mlxRuntime = MLXQwenRuntime(
            engine: MLXSwiftQwenVisionEngine(runner: mlxRunner),
            visualSpecialist: visualSpecialist,
            includeVisualSpecialistBlock: conditioned
        )

        let runtimes: [InferenceRuntime]
        switch validatedConfig.runtimePolicy {
        case .mlxOnly:
            runtimes = [mlxRuntime]
        case .mlxWithMNNFallback:
            guard let mnnRunner else {
                throw ProductionFactoryError.mnnRunnerRequiredForPolicy
            }
            let mnnRuntime = MNNQwenRuntime(
                engine: MNNQwenVisionEngine(runner: mnnRunner),
                visualSpecialist: visualSpecialist,
                includeVisualSpecialistBlock: conditioned
            )
            runtimes = [mlxRuntime, mnnRuntime]
        }

        let timeoutNanoseconds = UInt64(validatedConfig.timeoutSeconds * 1_000_000_000)
        let orchestrator = RuntimeOrchestrator(
            runtimes: runtimes,
            timeoutNanoseconds: timeoutNanoseconds,
            maxAttemptsPerRuntime: validatedConfig.maxAttemptsPerRuntime
        )
        return AppViewModel(
            orchestrator: orchestrator,
            store: store,
            preferencesStore: preferencesStore,
            weightStore: weightStore,
            modelPath: validatedConfig.modelPath,
            adapterPath: validatedConfig.adapterPath
        )
    }
}
