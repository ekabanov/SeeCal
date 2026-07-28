import Foundation
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
        visualSpecialist: (any VisualSpecialistPredicting)? = nil
    ) throws -> AppViewModel {
        return try SeeCalProductionFactory.makeViewModel(
            config: config,
            mlxRunner: mlxRunner,
            mnnRunner: mnnRunner,
            visualSpecialist: visualSpecialist
        )
    }

    @MainActor
    public static func makeProductionViewModelUsingMLX(
        config: QwenRuntimeConfig,
        mnnRunner: MNNQwenVisionEngine.Runner? = nil
    ) async throws -> AppViewModel {
        let engine = try await SeeCalMLXEngine.make(config: config)
        let visualSpecialist: (any VisualSpecialistPredicting)?
        if let modelPath = config.visualSpecialistModelPath {
            visualSpecialist = try CoreMLVisualSpecialist(modelPath: modelPath)
        } else {
            visualSpecialist = nil
        }
        return try SeeCalProductionFactory.makeViewModel(
            config: config,
            mlxRunner: { imagePath, prompt in
                try await engine.generate(imagePath: imagePath, prompt: prompt)
            },
            mnnRunner: mnnRunner,
            visualSpecialist: visualSpecialist
        )
    }

    @MainActor
    public static func makeDevelopmentViewModel() -> AppViewModel {
        let mlxRuntime = MLXQwenRuntime(engine: DevelopmentMockQwenVisionEngine())
        let mnnRuntime = MNNQwenRuntime(engine: DevelopmentMockQwenVisionEngine())
        let orchestrator = RuntimeOrchestrator(runtimes: [mlxRuntime, mnnRuntime])
        let store = InMemoryMealLogStore()
        return AppViewModel(orchestrator: orchestrator, store: store)
    }
}
