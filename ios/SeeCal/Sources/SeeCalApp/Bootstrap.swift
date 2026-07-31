import Foundation
import SeeCalDiagnostics
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence

public struct DevelopmentMockQwenVisionEngine: NativeQwenVisionEngine {
    public init() {}

    public func run(imagePath: String, prompt: String) async throws -> String {
        _ = prompt
        return """
        {
          "total_calories": 640,
          "protein_g": 42,
          "fat_g": 22,
          "carbs_g": 71,
          "confidence": 0.86,
          "items": [
            {
              "name": "chicken",
              "estimated_grams": 150,
              "calories": 280,
              "protein_g": 34,
              "fat_g": 8,
              "carbs_g": 0
            },
            {
              "name": "rice",
              "estimated_grams": 180,
              "calories": 260,
              "protein_g": 5,
              "fat_g": 2,
              "carbs_g": 56
            },
            {
              "name": "vegetables",
              "estimated_grams": 120,
              "calories": 100,
              "protein_g": 3,
              "fat_g": 12,
              "carbs_g": 15
            }
          ],
          "uncertainty_flags": ["portion_uncertain"]
        }
        """
    }
}

public enum SeeCalBootstrap {
    @MainActor
    public static func makeProductionViewModel(
        config: QwenRuntimeConfig,
        mlxRunner: @escaping MLXSwiftQwenVisionEngine.Runner,
        mnnRunner: MNNQwenVisionEngine.Runner? = nil,
        visualSpecialist: (any VisualSpecialistPredicting)? = nil,
        factoredScalePredictor: (any ScalePredicting)? = nil,
        nutritionResolver: (any NutritionResolving)? = nil,
        nutritionDatabaseURL: URL? = nil
    ) throws -> AppViewModel {
        return try SeeCalProductionFactory.makeViewModel(
            config: config,
            mlxRunner: mlxRunner,
            mnnRunner: mnnRunner,
            visualSpecialist: visualSpecialist,
            factoredScalePredictor: factoredScalePredictor,
            nutritionResolver: nutritionResolver,
            nutritionCandidateProvider: makeCandidateProvider(databaseURL: nutritionDatabaseURL)
        )
    }

    @MainActor
    public static func makeProductionViewModelUsingMLX(
        config: QwenRuntimeConfig,
        mnnRunner: MNNQwenVisionEngine.Runner? = nil,
        nutritionDatabaseURL: URL? = nil,
        factoredScaleModelPath: String? = nil,
        factoredScaleCalibrationMarginGrams: Double = 0
    ) throws -> AppViewModel {
        let modelPreparationState = ModelPreparationState(phase: .notStarted)
        let engine = SeeCalMLXEngine(
            config: config,
            loadStateObserver: { state in
                await modelPreparationState.update(from: state)
            }
        )
        let visualSpecialist: (any VisualSpecialistPredicting)?
        if let modelPath = config.visualSpecialistModelPath {
            visualSpecialist = LazyCoreMLVisualSpecialist(modelPath: modelPath)
        } else {
            visualSpecialist = nil
        }
        var factoredScalePredictor: (any ScalePredicting)?
        var nutritionResolver: (any NutritionResolving)?
        var nutritionCandidateProvider = makeCandidateProvider(
            databaseURL: nutritionDatabaseURL
        )
        if let factoredScaleModelPath {
            guard let nutritionDatabaseURL else {
                throw NutritionDatabaseError.openFailed(
                    "factored inference requires the bundled nutrition database"
                )
            }
            let resolver = try SQLiteNutritionResolver(databaseURL: nutritionDatabaseURL)
            factoredScalePredictor = try CoreMLScalePredictor(
                modelPath: factoredScaleModelPath,
                calibrationMarginGrams: factoredScaleCalibrationMarginGrams
            )
            nutritionResolver = resolver
            nutritionCandidateProvider = resolver
        }
        return try SeeCalProductionFactory.makeViewModel(
            config: config,
            mlxRunner: { imagePath, prompt in
                try await engine.generate(imagePath: imagePath, prompt: prompt)
            },
            mnnRunner: mnnRunner,
            visualSpecialist: visualSpecialist,
            factoredScalePredictor: factoredScalePredictor,
            nutritionResolver: nutritionResolver,
            modelPreparationState: modelPreparationState,
            nutritionCandidateProvider: nutritionCandidateProvider
        )
    }

    @MainActor
    public static func makeDevelopmentViewModel(
        nutritionDatabaseURL: URL? = nil
    ) -> AppViewModel {
        let mlxRuntime = MLXQwenRuntime(engine: DevelopmentMockQwenVisionEngine())
        let mnnRuntime = MNNQwenRuntime(engine: DevelopmentMockQwenVisionEngine())
        let orchestrator = RuntimeOrchestrator(runtimes: [mlxRuntime, mnnRuntime])
        let store = InMemoryMealLogStore()
        return AppViewModel(
            orchestrator: orchestrator,
            store: store,
            nutritionCandidateProvider: makeCandidateProvider(databaseURL: nutritionDatabaseURL)
        )
    }

    private static func makeCandidateProvider(
        databaseURL: URL?
    ) -> (any NutritionCandidateProviding)? {
        guard let databaseURL else {
            SeeCalDiagnostics.record(
                .info,
                category: "nutrition_database",
                name: "replacement_database_unavailable"
            )
            return nil
        }
        do {
            let resolver = try SQLiteNutritionResolver(databaseURL: databaseURL)
            SeeCalDiagnostics.record(
                .notice,
                category: "nutrition_database",
                name: "replacement_database_loaded"
            )
            return resolver
        } catch {
            SeeCalDiagnostics.record(
                .error,
                category: "nutrition_database",
                name: "replacement_database_load_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
            return nil
        }
    }
}
