import SeeCalDomain
import XCTest
@testable import SeeCalInference

/// LoRA adapter loading tests plus a scaffold for verifying Swift/Python parity.
///
/// ## Parity verification procedure (manual, requires model weights on disk)
///
/// The goal is to confirm the Swift stack (base model + adapter fused via
/// LoRAContainer) produces the same nutrition JSON as the Python reference
/// (mlx-vlm, base model + unfused adapter):
///
/// 1. Convert the trained adapter (from the repo root):
///        .venv/bin/python convert_adapter_for_swift.py \
///            --adapter-path adapters_v4 --output-path adapters_v4_swift
///
/// 2. Produce the Python reference output for one test image:
///        .venv/bin/python 03_infer.py \
///            --image dataset_clean/dish_XXXX/overhead.jpg \
///            --adapter-path adapters_v4
///    Save the printed JSON to a file, e.g. /tmp/python_reference.json.
///
/// 3. Run the Swift side on the same image (macOS, Apple silicon):
///        SEECAL_PARITY_MODEL_DIR=~/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit \
///        SEECAL_PARITY_ADAPTER_DIR=$PWD/../../adapters_v4_swift \
///        SEECAL_PARITY_IMAGE=$PWD/../../dataset_clean/dish_XXXX/overhead.jpg \
///        swift test --filter AdapterParityTests/testAdapterInferenceParityScaffold
///    The test prints the generated JSON.
///
/// 4. Compare: with temperature 0.1 exact token-level equality is not guaranteed
///    (sampling, and fuse() requantizes the merged 4-bit weights, see
///    MLXQwen35RunnerBuilder.applyAdapter), so compare semantically:
///    total_calories / protein_g / fat_g / carbs_g within a few percent and the
///    same leading ingredients. A large divergence (e.g. Swift output matching
///    the *base* model's output) indicates the adapter was not applied.
///
/// 5. Base-model sanity check: run step 3 again without SEECAL_PARITY_ADAPTER_DIR
///    and confirm the output differs from the adapter run — this proves the
///    adapter actually changes the generation.
///
/// ## CLI toolchain caveats (as of Xcode 26.6)
///
/// - The pinned mlx-swift revision vendors an fmt version whose consteval format
///   strings do not compile with Xcode 26.6's clang under C++20. Work around with:
///       swift build/test ... -Xcxx -DFMT_CONSTEVAL=
/// - `swift test` (CLI) does not produce mlx's Metal shader library, so MLX fails
///   at runtime with "Failed to load the default metallib". Workaround: compile the
///   8 vendored kernels and colocate the library with the test binary:
///       cd .build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal
///       for f in *.metal; do xcrun -sdk macosx metal -c "$f" -I. -o "/tmp/${f%.metal}.air"; done
///       xcrun -sdk macosx metallib /tmp/*.air -o mlx.metallib
///       cp mlx.metallib <pkg>/.build/arm64-apple-macosx/debug/SeeCalPackageTests.xctest/Contents/MacOS/
///   Xcode builds (the app target) handle both automatically... if the fmt issue
///   is fixed upstream; otherwise add OTHER_CPLUSPLUSFLAGS = -DFMT_CONSTEVAL= there too.
final class AdapterParityTests: XCTestCase {

    /// A configured-but-missing adapter directory must fail loudly before any
    /// model weights are touched, never silently fall back to the base model.
    func testMissingAdapterDirectoryThrowsBeforeModelLoad() async {
        let config = QwenRuntimeConfig(
            modelPath: QwenRuntimeConfig.recommendedModelID,
            adapterPath: "/nonexistent/adapter/dir"
        )
        do {
            _ = try await MLXQwen35RunnerBuilder.makeRunner(config: config)
            XCTFail("Expected adapterPathNotFound to be thrown")
        } catch let error as MLXRunnerBuilderError {
            XCTAssertEqual(error, .adapterPathNotFound("/nonexistent/adapter/dir"))
        } catch {
            XCTFail("Expected MLXRunnerBuilderError.adapterPathNotFound, got \(error)")
        }
    }

    /// An adapter directory missing its safetensors/config must also fail loudly.
    func testIncompleteAdapterDirectoryThrowsBeforeModelLoad() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("seecal-empty-adapter-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = QwenRuntimeConfig(
            modelPath: QwenRuntimeConfig.recommendedModelID,
            adapterPath: dir.path
        )
        do {
            _ = try await MLXQwen35RunnerBuilder.makeRunner(config: config)
            XCTFail("Expected adapterLoadFailed to be thrown")
        } catch let error as MLXRunnerBuilderError {
            guard case .adapterLoadFailed = error else {
                XCTFail("Expected adapterLoadFailed, got \(error)")
                return
            }
        } catch {
            XCTFail("Expected MLXRunnerBuilderError.adapterLoadFailed, got \(error)")
        }
    }

    /// End-to-end scaffold: loads the real model + adapter and runs one inference.
    /// Skipped unless the SEECAL_PARITY_* environment variables are set (see the
    /// procedure in the type-level comment). This is a harness, not an assertion
    /// of numeric parity — the comparison against 03_infer.py output is manual.
    func testAdapterInferenceParityScaffold() async throws {
        let env = ProcessInfo.processInfo.environment
        guard
            let modelDir = env["SEECAL_PARITY_MODEL_DIR"],
            let imagePath = env["SEECAL_PARITY_IMAGE"]
        else {
            throw XCTSkip("Set SEECAL_PARITY_MODEL_DIR and SEECAL_PARITY_IMAGE (and optionally SEECAL_PARITY_ADAPTER_DIR) to run")
        }
        let adapterDir = env["SEECAL_PARITY_ADAPTER_DIR"]

        let config = QwenRuntimeConfig(
            modelPath: (modelDir as NSString).expandingTildeInPath,
            adapterPath: adapterDir.map { ($0 as NSString).expandingTildeInPath },
            maxOutputTokens: 1024,
            temperature: 0.1,
            timeoutSeconds: 300
        )
        let runner = try await MLXQwen35RunnerBuilder.makeRunner(config: config)

        // Same prompt the app sends — and byte-identical to the Python side
        // (02_prepare_finetune.py / 03_infer.py), per the global prompt constraint.
        let prompt = QwenPromptBuilder().buildPrompt(
            request: FoodScanRequest(imagePath: imagePath, mealType: .lunch)
        )
        let output = try await runner((imagePath as NSString).expandingTildeInPath, prompt)

        print("[AdapterParityTests] adapter=\(adapterDir ?? "<none — base model>")")
        print("[AdapterParityTests] output JSON:\n\(output)")

        // Minimal structural assertion: output parses as a JSON object with the
        // expected top-level nutrition keys.
        let data = try XCTUnwrap(output.data(using: .utf8))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for key in ["total_calories", "protein_g", "fat_g", "carbs_g", "items"] {
            XCTAssertNotNil(object[key], "missing key \(key) in model output")
        }
    }
}
