import SwiftUI
import SeeCalApp
import SeeCalDiagnostics
import SeeCalInference

/// Thin app shell around the SeeCal SwiftPM package (ios/SeeCal).
///
/// All screens, view models, and inference live in the package's `SeeCalApp`
/// product; this target only hosts `ProductionRootView` and resolves where
/// the model weights are (bundled, side-loaded, or simulator fallback paths —
/// see `ModelAssetResolver`).
@main
struct SeeCalHostApp: App {
    private let runtimeConfig: QwenRuntimeConfig
    private let nutritionDatabaseURL: URL?
    private let factoredScaleModelPath: String?
    private let factoredScaleCalibrationMarginGrams: Double

    init() {
        SeeCalDiagnostics.record(.notice, category: "app", name: "process_started")
        let assets = ModelAssetResolver.resolveInferenceAssets()
        runtimeConfig = QwenRuntimeConfig(
            modelPath: ModelAssetResolver.resolveModelPath(),
            adapterPath: assets.adapterPath,
            visualSpecialistModelPath: assets.visualSpecialistModelPath,
            runtimePolicy: .mlxOnly,
            // IDENTIFY emits a much smaller object than the monolith, while
            // retaining headroom for unusually long visible-component lists.
            maxOutputTokens: assets.factoredScaleModelPath == nil ? 1536 : 512,
            temperature: 0.1,
            timeoutSeconds: 180,
            maxAttemptsPerRuntime: 1
        )
        nutritionDatabaseURL = ModelAssetResolver.resolveNutritionDatabaseURL()
        factoredScaleModelPath = assets.factoredScaleModelPath
        factoredScaleCalibrationMarginGrams =
            assets.factoredScaleCalibrationMarginGrams
    }

    var body: some Scene {
        WindowGroup {
            ProductionRootView(
                config: runtimeConfig,
                nutritionDatabaseURL: nutritionDatabaseURL,
                factoredScaleModelPath: factoredScaleModelPath,
                factoredScaleCalibrationMarginGrams:
                    factoredScaleCalibrationMarginGrams
            )
        }
    }
}
