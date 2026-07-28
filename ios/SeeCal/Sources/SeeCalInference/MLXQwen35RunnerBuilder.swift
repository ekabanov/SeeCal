import CoreImage
import Darwin
import Foundation
import SeeCalDiagnostics

#if canImport(MLXLMCommon) && canImport(MLXVLM)
import MLX
import MLXLMCommon
import MLXVLM
#endif

/// Cross-module hook for releasing MLX's GPU buffer cache under memory
/// pressure. The app wires this to `UIApplication.didReceiveMemoryWarning` so a
/// long session that creeps toward the jetsam limit sheds cached buffers before
/// being killed. No-op when MLX isn't linked (e.g. unit-test builds).
public enum SeeCalMemory {
    public static func releaseCaches() {
        #if canImport(MLXLMCommon) && canImport(MLXVLM)
        MLX.Memory.clearCache()
        #endif
    }
}

public enum MLXRunnerBuilderError: Error, Equatable, CustomStringConvertible {
    case mlxPackagesNotLinked
    case localModelPathNotFound(String)
    case adapterPathNotFound(String)
    case adapterLoadFailed(path: String, underlying: String)
    case invalidImagePath(String)
    case emptyGeneration

    public var description: String {
        switch self {
        case .mlxPackagesNotLinked:
            return "MLX runner requires MLXLMCommon and MLXVLM to be added as dependencies of the SeeCalInference package target"
        case let .localModelPathNotFound(path):
            return "Local model path not found: \(path)"
        case let .adapterPathNotFound(path):
            return "Configured LoRA adapter path not found (expected directory with adapter_config.json + adapters.safetensors): \(path)"
        case let .adapterLoadFailed(path, underlying):
            return "Failed to load LoRA adapter at \(path): \(underlying)"
        case let .invalidImagePath(path):
            return "Image path not found: \(path)"
        case .emptyGeneration:
            return "Model returned empty text"
        }
    }
}

public enum MLXQwen35RunnerBuilder {
    /// Returns the process resident memory in MB, or -1 on failure.
    private static func memMB() -> String {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let ret: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard ret == KERN_SUCCESS else { return "?MB" }
        return String(format: "%.0fMB", Double(info.resident_size) / 1_048_576)
    }

    /// MLX's own view of memory (active/cache/peak in MB) — distinguishes live
    /// tensors from the retained buffer cache, so the log shows whether the
    /// cache cap is holding.
    private static func mlxMem() -> String {
        #if canImport(MLXLMCommon) && canImport(MLXVLM)
        let s = MLX.Memory.snapshot()
        let mb = { (b: Int) in b / 1_048_576 }
        return "active=\(mb(s.activeMemory))MB cache=\(mb(s.cacheMemory))MB peak=\(mb(s.peakMemory))MB"
        #else
        return "n/a"
        #endif
    }

    /// Metal buffer-cache ceiling. MLX's cache limit otherwise defaults to the
    /// device memory limit (~5–6 GB on an 8 GB iPhone), so it retains gigabytes
    /// of freed intermediate buffers during load/fuse/warmup — the confirmed
    /// cause of SeeCal ballooning to ~5.5 GB and being jetsam-killed on launch.
    /// MLX's own docs note even ~2 MB performs comparably; 48 MB is a safe margin.
    private static let mlxCacheLimitBytes = 48 * 1024 * 1024

    private static func memoryFields() -> [String: String] {
        ["resident_memory": memMB(), "mlx_memory": mlxMem()]
    }

    private static func elapsedMilliseconds(since date: Date) -> String {
        String(Int(Date().timeIntervalSince(date) * 1_000))
    }

    public static func makeRunner(config: QwenRuntimeConfig) async throws -> MLXSwiftQwenVisionEngine.Runner {
        _ = try config.validated()

        #if canImport(MLXLMCommon) && canImport(MLXVLM)
        SeeCalDiagnostics.record(
            .notice,
            category: "mlx",
            name: "runner_build_started",
            fields: [
                "cache_limit_mb": String(mlxCacheLimitBytes / (1024 * 1024)),
                "model_source": config.modelPath.hasPrefix("/") ? "local_directory" : "model_id",
                "adapter_configured": String(config.adapterPath != nil)
            ]
        )
        if config.modelPath.hasPrefix("/") {
            var isDirectory: ObjCBool = false
            let exists = FileManager.default.fileExists(atPath: config.modelPath, isDirectory: &isDirectory)
            guard exists && isDirectory.boolValue else {
                SeeCalDiagnostics.record(
                    .fault,
                    category: "mlx",
                    name: "local_model_directory_missing"
                )
                throw MLXRunnerBuilderError.localModelPathNotFound(config.modelPath)
            }
            logLocalModelPreflight(path: config.modelPath)
            validateLocalModelType(path: config.modelPath)
        }

        // Fail fast (before the multi-GB model load) if an adapter is configured
        // but its directory is missing. An adapterPath that is set must load —
        // silently falling back to the base model would hide a broken deployment.
        if let adapterPath = config.adapterPath {
            try validateAdapterDirectory(path: adapterPath)
        }

        // Cap the Metal buffer cache after filesystem-only preflight but BEFORE
        // any model allocation. Keeping preflight first means a missing adapter
        // still fails cleanly in host tests where no Metal library is present.
        MLX.Memory.cacheLimit = mlxCacheLimitBytes
        SeeCalDiagnostics.record(
            .info,
            category: "mlx",
            name: "cache_limit_configured",
            fields: ["cache_limit_mb": String(mlxCacheLimitBytes / (1024 * 1024))]
        )

        let modelConfiguration: ModelConfiguration
        if config.modelPath.hasPrefix("/") {
            modelConfiguration = ModelConfiguration(directory: URL(fileURLWithPath: config.modelPath, isDirectory: true))
        } else {
            modelConfiguration = ModelConfiguration(id: config.modelPath)
        }
        // Background memory monitor: records every 3 s while loadContainer runs.
        // If the crash is fatalError/precondition (bypasses catch), the last
        // tick line and mem reading show exactly how far loading got.
        let memMonitor = Task {
            var tick = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                guard !Task.isCancelled else { break }
                tick += 1
                SeeCalDiagnostics.record(
                    .info,
                    category: "mlx",
                    name: "model_container_loading",
                    fields: ["tick": String(tick)].merging(memoryFields()) { current, _ in current }
                )
            }
        }

        do {
            let loadStarted = Date()
            SeeCalDiagnostics.record(
                .notice,
                category: "mlx",
                name: "model_container_load_started",
                fields: memoryFields()
            )
            let modelContainer = try await VLMModelFactory.shared.loadContainer(configuration: modelConfiguration)
            memMonitor.cancel()
            SeeCalDiagnostics.record(
                .notice,
                category: "mlx",
                name: "model_container_loaded",
                fields: ["duration_ms": elapsedMilliseconds(since: loadStarted)]
                    .merging(memoryFields()) { current, _ in current }
            )

            // Apply the LoRA adapter (produced by convert_adapter_for_swift.py from an
            // mlx-vlm training run) before warmup so warmup compiles the adapted graph.
            if let adapterPath = config.adapterPath {
                try await applyAdapter(at: adapterPath, to: modelContainer)
            }

            // Warm up Metal shader JIT compilation with a vision inference.
            // Text-only warmup (~0.5s) doesn't trigger vision kernels; a real image does.
            // Without this the first user inference takes 60-90s instead of ~15s.
            let warmupStarted = Date()
            SeeCalDiagnostics.record(
                .notice,
                category: "mlx",
                name: "warmup_started",
                fields: memoryFields()
            )
            let warmupImage: CIImage = {
                // Match actual food photo resolution (576×768) so the same Metal kernel
                // specializations are triggered as during real inference.
                return CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5))
                    .cropped(to: CGRect(x: 0, y: 0, width: 576, height: 768))
            }()
            let warmupInput = UserInput(
                chat: [.user("hi", images: [.ciImage(warmupImage)])],
                additionalContext: ["enable_thinking": false]
            )
            if let warmupPrepared = try? await modelContainer.prepare(input: warmupInput) {
                let warmupParams = GenerateParameters(maxTokens: 1, temperature: 1.0)
                let warmupStream = try? await modelContainer.generate(input: warmupPrepared, parameters: warmupParams)
                if let stream = warmupStream {
                    for await _ in stream {}
                }
            }
            // Warmup allocates full vision + LLM activation buffers; release
            // them back to the system so steady-state memory settles near the
            // model's resident size rather than the warmup peak.
            MLX.Memory.clearCache()
            SeeCalDiagnostics.record(
                .notice,
                category: "mlx",
                name: "warmup_succeeded",
                fields: ["duration_ms": elapsedMilliseconds(since: warmupStarted)]
                    .merging(memoryFields()) { current, _ in current }
            )

            return { imagePath, prompt in
                guard FileManager.default.fileExists(atPath: imagePath) else {
                    SeeCalDiagnostics.record(
                        .error,
                        category: "mlx",
                        name: "inference_image_missing"
                    )
                    throw MLXRunnerBuilderError.invalidImagePath(imagePath)
                }

                let imageURL = URL(fileURLWithPath: imagePath)
                let input = UserInput(
                    chat: [.user(prompt, images: [.url(imageURL)])],
                    additionalContext: ["enable_thinking": false]
                )
                let inferStarted = Date()
                SeeCalDiagnostics.record(
                    .notice,
                    category: "mlx",
                    name: "inference_started",
                    fields: [
                        "prompt_characters": String(prompt.count),
                        "max_output_tokens": String(config.maxOutputTokens),
                        "temperature": String(config.temperature)
                    ].merging(memoryFields()) { current, _ in current }
                )

                let parameters = GenerateParameters(
                    maxTokens: config.maxOutputTokens,
                    temperature: Float(config.temperature),
                    repetitionPenalty: 1.1
                )
                var generated = ""
                let prepareStarted = Date()
                let prepared = try await modelContainer.prepare(input: input)
                let tokenCount = prepared.text.tokens.shape.reduce(1, *)
                SeeCalDiagnostics.record(
                    .info,
                    category: "mlx",
                    name: "input_prepared",
                    fields: [
                        "duration_ms": elapsedMilliseconds(since: prepareStarted),
                        "token_count": String(tokenCount)
                    ].merging(memoryFields()) { current, _ in current }
                )
                let generateStarted = Date()
                let stream = try await modelContainer.generate(input: prepared, parameters: parameters)
                for await event in stream {
                    if case let .chunk(text) = event {
                        generated.append(text)
                    }
                }
                SeeCalDiagnostics.record(
                    .info,
                    category: "mlx",
                    name: "generation_finished",
                    fields: [
                        "duration_ms": elapsedMilliseconds(since: generateStarted),
                        "generated_characters": String(generated.count)
                    ].merging(memoryFields()) { current, _ in current }
                )
                // Release this scan's activation + KV-cache buffers so repeated
                // scans don't accumulate memory over a session. The cache cap
                // already bounds the pool, but freeing per-inference keeps
                // steady-state near the model's resident size. Watch `active=` in
                // the log across scans: if it climbs, that's a live-tensor leak
                // to chase; if only cache/peak spike, this clear handles it.
                MLX.Memory.clearCache()
                SeeCalDiagnostics.record(
                    .info,
                    category: "mlx",
                    name: "inference_cache_cleared",
                    fields: memoryFields()
                )

                var text = generated.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else {
                    throw MLXRunnerBuilderError.emptyGeneration
                }

                // Strip Qwen3.5 thinking block (<think>...</think>) before JSON parsing.
                if let thinkEnd = text.range(of: "</think>") {
                    text = String(text[thinkEnd.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                } else if text.hasPrefix("<think>") {
                    // Thinking block not closed — discard it entirely.
                    text = ""
                }

                // Extract the first JSON object in case the model added surrounding text.
                if !text.hasPrefix("{") {
                    if let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}") {
                        text = String(text[start...end])
                    }
                }

                // Repair truncated JSON: if the model stopped mid-generation, close any
                // unclosed brackets after the last complete `}`.
                if !text.hasSuffix("}") {
                    if let lastBrace = text.lastIndex(of: "}") {
                        let candidate = String(text[...lastBrace])
                        var braceDepth = 0
                        var bracketDepth = 0
                        for ch in candidate {
                            switch ch {
                            case "{": braceDepth += 1
                            case "}": braceDepth -= 1
                            case "[": bracketDepth += 1
                            case "]": bracketDepth -= 1
                            default: break
                            }
                        }
                        if braceDepth >= 0 && bracketDepth >= 0 {
                            text = candidate
                                + String(repeating: "]", count: bracketDepth)
                                + String(repeating: "}", count: braceDepth)
                            SeeCalDiagnostics.record(
                                .error,
                                category: "mlx",
                                name: "truncated_json_repaired",
                                fields: [
                                    "brackets_added": String(bracketDepth),
                                    "braces_added": String(braceDepth)
                                ]
                            )
                        }
                    }
                }

                SeeCalDiagnostics.record(
                    .notice,
                    category: "mlx",
                    name: "inference_finished",
                    fields: [
                        "duration_ms": elapsedMilliseconds(since: inferStarted),
                        "output_characters": String(text.count)
                    ]
                )
                return text
            }
        } catch {
            memMonitor.cancel()
            SeeCalDiagnostics.record(
                .fault,
                category: "mlx",
                name: "runner_build_failed",
                fields: SeeCalDiagnostics.errorFields(error)
                    .merging(memoryFields()) { current, _ in current }
            )
            throw error
        }
        #else
        SeeCalDiagnostics.record(
            .fault,
            category: "mlx",
            name: "mlx_packages_not_linked"
        )
        throw MLXRunnerBuilderError.mlxPackagesNotLinked
        #endif
    }

    /// Throws unless `path` is a directory containing the two files
    /// `LoRAContainer.from(directory:)` reads.
    private static func validateAdapterDirectory(path: String) throws {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
            SeeCalDiagnostics.record(
                .fault,
                category: "mlx",
                name: "adapter_directory_missing"
            )
            throw MLXRunnerBuilderError.adapterPathNotFound(path)
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let required = ["adapter_config.json", "adapters.safetensors"]
        let missing = required.filter {
            !fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        guard missing.isEmpty else {
            SeeCalDiagnostics.record(
                .fault,
                category: "mlx",
                name: "adapter_files_missing",
                fields: ["missing_files": missing.sorted().joined(separator: ",")]
            )
            throw MLXRunnerBuilderError.adapterLoadFailed(
                path: path,
                underlying: "missing required files: \(missing.joined(separator: ", "))"
            )
        }
    }

    #if canImport(MLXLMCommon) && canImport(MLXVLM)
    /// Loads the LoRA adapter and fuses it into the model's layers.
    ///
    /// Any failure throws `MLXRunnerBuilderError.adapterLoadFailed` — an adapter that is
    /// configured but cannot be applied must never silently degrade to the base model.
    private static func applyAdapter(at path: String, to modelContainer: ModelContainer) async throws {
        let started = Date()
        SeeCalDiagnostics.record(
            .notice,
            category: "mlx",
            name: "adapter_fuse_started",
            fields: memoryFields()
        )
        do {
            let adapterURL = URL(fileURLWithPath: path, isDirectory: true)
            let adapter = try LoRAContainer.from(directory: adapterURL)
            try await modelContainer.perform { context in
                // fuse(with:) applies the adapter (load) and then merges the LoRA
                // deltas into the base layers, removing the per-token x@A@B cost.
                // On the 4-bit model this requantizes the merged weights (QLoRALinear
                // dequantizes, adds scale*A@B, requantizes at the same groupSize/bits),
                // which introduces a small quantization error on the delta. For output
                // numerically closest to Python `03_infer.py` (base + unfused adapter),
                // replace this call with `try adapter.load(into: context.model)`.
                try adapter.fuse(with: context.model)
            }
            // The fuse dequantizes every layer to fp16, adds the delta, and
            // requantizes — leaving those fp16 temporaries in the buffer cache.
            // Drop them immediately so they don't stack under the warmup.
            MLX.Memory.clearCache()
        } catch {
            SeeCalDiagnostics.record(
                .fault,
                category: "mlx",
                name: "adapter_fuse_failed",
                fields: SeeCalDiagnostics.errorFields(error)
                    .merging(memoryFields()) { current, _ in current }
            )
            throw MLXRunnerBuilderError.adapterLoadFailed(path: path, underlying: String(describing: error))
        }
        SeeCalDiagnostics.record(
            .notice,
            category: "mlx",
            name: "adapter_fused",
            fields: ["duration_ms": elapsedMilliseconds(since: started)]
                .merging(memoryFields()) { current, _ in current }
        )
    }
    #endif

    private static func logLocalModelPreflight(path: String) {
        let fm = FileManager.default
        let required = [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "processor_config.json"
        ]
        let existing = required.filter { fm.fileExists(atPath: URL(fileURLWithPath: path).appendingPathComponent($0).path) }
        let missing = required.filter { !existing.contains($0) }
        SeeCalDiagnostics.record(
            .info,
            category: "mlx",
            name: "local_model_preflight",
            fields: [
                "required_file_count": String(required.count),
                "existing_file_count": String(existing.count),
                "missing_file_count": String(missing.count)
            ]
        )
        if !missing.isEmpty {
            SeeCalDiagnostics.record(
                .error,
                category: "mlx",
                name: "local_model_preflight_files_missing",
                fields: ["missing_files": missing.sorted().joined(separator: ",")]
            )
        }

        do {
            let topLevel = try fm.contentsOfDirectory(atPath: path).sorted()
            SeeCalDiagnostics.record(
                .debug,
                category: "mlx",
                name: "local_model_directory_inspected",
                fields: ["entry_count": String(topLevel.count)]
            )
        } catch {
            SeeCalDiagnostics.record(
                .error,
                category: "mlx",
                name: "local_model_directory_inspection_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
        }
    }

    private static func validateLocalModelType(path: String) {
        let configURL = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            SeeCalDiagnostics.record(
                .error,
                category: "mlx",
                name: "model_config_unreadable"
            )
            return
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelType = object["model_type"] as? String
        else {
            SeeCalDiagnostics.record(
                .error,
                category: "mlx",
                name: "model_type_unreadable"
            )
            return
        }

        SeeCalDiagnostics.record(
            .info,
            category: "mlx",
            name: "model_type_detected",
            fields: ["model_type": modelType]
        )
    }
}
