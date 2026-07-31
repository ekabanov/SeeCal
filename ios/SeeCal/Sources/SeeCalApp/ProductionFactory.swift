import Foundation
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence

public enum ProductionFactoryError: Error, Equatable, CustomStringConvertible {
    case mnnRunnerRequiredForPolicy
    case incompleteFactoredConfiguration

    public var description: String {
        switch self {
        case .mnnRunnerRequiredForPolicy:
            return "runtimePolicy requires an MNN runner, but none was provided"
        case .incompleteFactoredConfiguration:
            return "factored inference requires both SCALE and nutrition resolver"
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
        factoredScalePredictor: (any ScalePredicting)? = nil,
        nutritionResolver: (any NutritionResolving)? = nil,
        store: MealLogStore = FileBackedMealLogStore(),
        preferencesStore: UserPreferencesStore = FileBackedUserPreferencesStore(),
        weightStore: WeightLogStore = FileBackedWeightLogStore(),
        modelPreparationState: ModelPreparationState = ModelPreparationState(),
        nutritionCandidateProvider: (any NutritionCandidateProviding)? = nil
    ) throws -> AppViewModel {
        let validatedConfig = try config.validated()
        guard (factoredScalePredictor == nil) == (nutritionResolver == nil) else {
            throw ProductionFactoryError.incompleteFactoredConfiguration
        }
        let conditioned = validatedConfig.visualSpecialistModelPath != nil
        let mlxEngine = MLXSwiftQwenVisionEngine(runner: mlxRunner)
        let mlxRuntime: any InferenceRuntime
        if let factoredScalePredictor, let nutritionResolver {
            mlxRuntime = FactoredNutritionRuntime(
                pipeline: FactoredNutritionInferencePipeline(
                    identifier: QwenFoodIdentifier(engine: mlxEngine),
                    scalePredictor: factoredScalePredictor,
                    resolver: nutritionResolver
                )
            )
        } else {
            mlxRuntime = MLXQwenRuntime(
                engine: mlxEngine,
                visualSpecialist: visualSpecialist,
                includeVisualSpecialistBlock: conditioned
            )
        }

        let runtimes: [InferenceRuntime]
        switch validatedConfig.runtimePolicy {
        case .mlxOnly:
            runtimes = [mlxRuntime]
        case .mlxWithMNNFallback:
            guard let mnnRunner else {
                throw ProductionFactoryError.mnnRunnerRequiredForPolicy
            }
            let mnnEngine = MNNQwenVisionEngine(runner: mnnRunner)
            let mnnRuntime: any InferenceRuntime
            if let factoredScalePredictor, let nutritionResolver {
                mnnRuntime = FactoredNutritionRuntime(
                    pipeline: FactoredNutritionInferencePipeline(
                        identifier: QwenFoodIdentifier(engine: mnnEngine),
                        scalePredictor: factoredScalePredictor,
                        resolver: nutritionResolver
                    )
                )
            } else {
                mnnRuntime = MNNQwenRuntime(
                    engine: mnnEngine,
                    visualSpecialist: visualSpecialist,
                    includeVisualSpecialistBlock: conditioned
                )
            }
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
            adapterPath: validatedConfig.adapterPath,
            modelPreparationState: modelPreparationState,
            nutritionCandidateProvider: nutritionCandidateProvider
        )
    }
}
