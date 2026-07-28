# SeeCal iOS Foundation

This directory contains a Swift package foundation for the SeeCal iPhone app with the runtime policy you requested:

- Primary runtime: MLX Swift
- Backup runtime: MNN (only with Qwen3.5 native multimodal parity)
- CoreML: R&D spike only
- No server fallback path

## Modules

- `SeeCalDomain`: strict scan request/result contracts and JSON validation.
- `SeeCalInference`: runtime abstractions, Qwen prompt contract, and runtime gate evaluator.
- `SeeCalPersistence`: meal log storage models.
- `SeeCalApp`: app view model + root view scaffold.

## Required output schema

```json
{
  "total_calories": 640,
  "protein_g": 42,
  "fat_g": 22,
  "carbs_g": 71,
  "confidence": 0.86,
  "items": [{"name": "chicken", "estimated_grams": 150, "calories": 280, "protein_g": 34, "fat_g": 8, "carbs_g": 0}],
  "uncertainty_flags": ["portion_uncertain"]
}
```

## Runtime gates

Run gate checks from metrics JSON:

```bash
python ios/SeeCal/Scripts/run_runtime_gates.py \
  --mlx ios/SeeCal/Docs/metrics_mlx_example.json \
  --mnn ios/SeeCal/Docs/metrics_mnn_example.json \
  --coreml ios/SeeCal/Docs/metrics_coreml_example.json
```

## Tests

```bash
cd ios/SeeCal
swift test
```

Xcode 26's bundled clang has a stricter default for `consteval` than the `fmt` version vendored
inside pinned `mlx-swift` expects, so both `swift build` and `swift test` fail against it unless
you disable that check:

```bash
swift build -Xcxx -DFMT_CONSTEVAL=
swift test -Xcxx -DFMT_CONSTEVAL=
```

## Inference reliability defaults

- Runtime orchestrator timeout: `180s` per attempt (`RuntimeConfiguration` validates the configured
  value is in `1...300`)
- Attempts per runtime: `1` (can be increased when wiring app-specific policies)
- Runtime fallback order: `MLX -> MNN` when MNN is configured

## Production integration

`adapterPath` below points at a converted LoRA adapter directory, not a raw mlx-vlm training
output. Adapters are produced by running `convert_adapter_for_swift.py` (in `ml/`) against an
mlx-vlm adapter directory (e.g. `adapters_v4/`, `adapters_v5/` once training lands) — see
`AGENTS.md` for the training side. The examples below reference `adapters_v4_swift/`, the current
converted adapter; once v5 finishes training and is converted, switch these to `adapters_v5_swift/`.
Do **not** point `adapterPath` at `adapters_v3` — that adapter is confirmed broken (mode collapse,
see `AGENTS.md` Training Results).

Use production bootstrap with an MLX runner closure:

```swift
let config = QwenRuntimeConfig(
    modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
    adapterPath: "/models/adapters_v4_swift",
    runtimePolicy: .mlxOnly,
    maxOutputTokens: 512,
    temperature: 0.1,
    timeoutSeconds: 180,
    maxAttemptsPerRuntime: 1
)

let viewModel = await SeeCalBootstrap.makeProductionViewModel(
    config: config,
    mlxRunner: { imagePath, prompt in
        // Call your MLX Swift Qwen3.5-native multimodal inference here.
        // Must return strict JSON string matching SeeCal schema.
        try await myMLXEngine.generate(imagePath: imagePath, prompt: prompt)
    },
    mnnRunner: nil // Optional fallback runner
)
```

Direct MLX path (no manual runner closure):

```swift
let viewModel = try await SeeCalBootstrap.makeProductionViewModelUsingMLX(
    config: config
)
```

Or use a ready-to-mount root view that loads the model:

```swift
import SwiftUI
import SeeCalApp
import SeeCalInference

@main
struct SeeCaliOSApp: App {
    var body: some Scene {
        WindowGroup {
            ProductionRootView(
                config: QwenRuntimeConfig(
                    modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
                    adapterPath: "/models/adapters_v4_swift",
                    runtimePolicy: .mlxOnly
                )
            )
        }
    }
}
```

Or build a runner directly from SeeCal's MLX helper:

```swift
let runner = try await MLXQwen35RunnerBuilder.makeRunner(config: config)
let viewModel = try await SeeCalBootstrap.makeProductionViewModel(
    config: config,
    mlxRunner: runner
)
```

For previews/local UI testing without model runtime:

```swift
let previewViewModel = await SeeCalBootstrap.makeDevelopmentViewModel()
```

You can also decode runtime config JSON at startup:

```swift
let data = try Data(contentsOf: Bundle.main.url(forResource: "runtime_config", withExtension: "json")!)
let config = try RuntimeConfigLoader.load(from: data)
```

Important:
- This helper expects model id `mlx-community/Qwen3.5-4B-MLX-4bit`.
- Your app target must link `MLXLMCommon` and `MLXVLM`; otherwise builder throws `mlxPackagesNotLinked`.

## Xcode dependency wiring

In your iOS app target, add package dependencies:
1. `https://github.com/ml-explore/mlx-swift`
2. `https://github.com/ml-explore/mlx-swift-examples` (if you need sample app patterns)
3. `https://github.com/ml-explore/mlx-swift-lm`

Ensure the target links products providing:
- `MLXLMCommon`
- `MLXVLM`

## iOS permissions (required for food photo testing)

Add to your iOS app target `Info.plist`:
- `NSPhotoLibraryUsageDescription` = `SeeCal needs photo library access to analyze food images.`
- `NSCameraUsageDescription` = `SeeCal needs camera access to capture food photos.`

## Bundling MLX model for device builds

For iPhone device runs, the model must be inside app resources.

1. In Xcode, add the folder `Qwen3.5-4B-MLX-4bit` as a folder reference under `SeeCalApp/Models/mlx-community/`.
2. Ensure target membership includes `SeeCalApp`.
3. The app resolves model path from bundle location:
   `Models/mlx-community/Qwen3.5-4B-MLX-4bit`.

Notes:
- Simulator builds can fall back to local dev path `~/.lmstudio/...` if the bundle is missing.
- Device builds require the bundled model folder.

## LoRA adapter

The LoRA adapter is loaded at startup via `LoRAContainer` when an adapter directory is found.
Resolution order (`ModelAssetResolver.resolveAdapterPath`):
1. Bundled `Models/adapters` inside the app bundle.
2. `Documents/App Support` (on-device, user- or app-installed adapter).
3. A simulator-only fallback to the repo path, for local development without bundling.

Behavior at the two failure points is intentionally different:
- **Missing adapter** (none of the above resolve to a directory): the app falls back to the base
  model and logs loudly that it is running without an adapter — this is a silent-looking but
  logged degradation, not an error.
- **Invalid adapter** (a directory is found but fails to load — wrong format, missing config,
  incompatible scale, etc.): this raises a typed error. There is no silent fallback to the base
  model in this case — an adapter that resolves but won't load is treated as a bug to fix, not a
  degraded mode to paper over.
