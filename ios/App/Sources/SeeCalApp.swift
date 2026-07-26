import SwiftUI
import SeeCalApp
import SeeCalInference

/// Thin app shell around the SeeCal SwiftPM package (ios/SeeCal).
///
/// All screens, view models, and inference live in the package's `SeeCalApp`
/// product; this target only hosts `ProductionRootView` and resolves where
/// the model weights are (bundled, side-loaded, or simulator fallback paths —
/// see `ModelAssetResolver`).
@main
struct SeeCalHostApp: App {
    var body: some Scene {
        WindowGroup {
            ProductionRootView(
                config: QwenRuntimeConfig(
                    modelPath: ModelAssetResolver.resolveModelPath(),
                    adapterPath: ModelAssetResolver.resolveAdapterPath(),
                    runtimePolicy: .mlxOnly,
                    maxOutputTokens: 1024,
                    temperature: 0.1,
                    timeoutSeconds: 180,
                    maxAttemptsPerRuntime: 1
                )
            )
        }
    }
}
