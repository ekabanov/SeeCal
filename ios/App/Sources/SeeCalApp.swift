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
    init() {
        SeeCalDiagnostics.record(.notice, category: "app", name: "process_started")
    }

    var body: some Scene {
        WindowGroup {
            ProductionRootView(
                config: QwenRuntimeConfig(
                    modelPath: ModelAssetResolver.resolveModelPath(),
                    adapterPath: ModelAssetResolver.resolveAdapterPath(),
                    visualSpecialistModelPath: ModelAssetResolver.resolveVisualSpecialistModelPath(),
                    runtimePolicy: .mlxOnly,
                    // Matches the eval's max_tokens (ml/infer.py) so the app can't
                    // truncate a long ingredient list where the eval had headroom.
                    // Ground-truth JSON is ~700 tokens at p90, so this only costs
                    // extra time on unusually long outputs.
                    maxOutputTokens: 1536,
                    temperature: 0.1,
                    timeoutSeconds: 180,
                    maxAttemptsPerRuntime: 1
                )
            )
        }
    }
}
