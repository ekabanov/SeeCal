import CoreImage
import Darwin
import Foundation
import os

#if canImport(MLXLMCommon) && canImport(MLXVLM)
import MLXLMCommon
import MLXVLM
#endif

public enum MLXRunnerBuilderError: Error, Equatable, CustomStringConvertible {
    case mlxPackagesNotLinked
    case localModelPathNotFound(String)
    case invalidImagePath(String)
    case emptyGeneration

    public var description: String {
        switch self {
        case .mlxPackagesNotLinked:
            return "MLX runner requires MLXLMCommon and MLXVLM to be added as dependencies of the SeeCalInference package target"
        case let .localModelPathNotFound(path):
            return "Local model path not found: \(path)"
        case let .invalidImagePath(path):
            return "Image path not found: \(path)"
        case .emptyGeneration:
            return "Model returned empty text"
        }
    }
}

public enum MLXQwen35RunnerBuilder {
    private static let logger = Logger(subsystem: "SeeCal", category: "MLXRunnerBuilder")

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

    public static func makeRunner(config: QwenRuntimeConfig) async throws -> MLXSwiftQwenVisionEngine.Runner {
        _ = try config.validated()

        #if canImport(MLXLMCommon) && canImport(MLXVLM)
        print("[SeeCal][MLXRunnerBuilder] makeRunner start modelPath=\(config.modelPath)")
        if config.modelPath.hasPrefix("/") {
            var isDirectory: ObjCBool = false
            let exists = FileManager.default.fileExists(atPath: config.modelPath, isDirectory: &isDirectory)
            guard exists && isDirectory.boolValue else {
                logger.error("Configured local model path does not exist: \(config.modelPath, privacy: .public)")
                print("[SeeCal][MLXRunnerBuilder] local path missing: \(config.modelPath)")
                throw MLXRunnerBuilderError.localModelPathNotFound(config.modelPath)
            }
            logLocalModelPreflight(path: config.modelPath)
            validateLocalModelType(path: config.modelPath)
        }

        let modelConfiguration: ModelConfiguration
        if config.modelPath.hasPrefix("/") {
            modelConfiguration = ModelConfiguration(directory: URL(fileURLWithPath: config.modelPath, isDirectory: true))
            logger.log("Starting MLX runner build from local directory: \(config.modelPath, privacy: .public)")
            print("[SeeCal][MLXRunnerBuilder] loading from local directory")
        } else {
            modelConfiguration = ModelConfiguration(id: config.modelPath)
            logger.log("Starting MLX runner build for model id: \(config.modelPath, privacy: .public)")
            print("[SeeCal][MLXRunnerBuilder] loading from model id")
        }
        // Background memory monitor: prints every 3 s while loadContainer runs.
        // If the crash is fatalError/precondition (bypasses catch), the last
        // tick line and mem reading show exactly how far loading got.
        let memMonitor = Task {
            var tick = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                guard !Task.isCancelled else { break }
                tick += 1
                print("[SeeCal][MLXRunnerBuilder] loading… tick=\(tick) mem=\(memMB())")
            }
        }

        do {
            let loadStarted = Date()
            print("[SeeCal][MLXRunnerBuilder] loadContainer start mem=\(memMB())")
            let modelContainer = try await VLMModelFactory.shared.loadContainer(configuration: modelConfiguration)
            memMonitor.cancel()
            logger.log("Model container loaded for model id: \(config.modelPath, privacy: .public)")
            print("[SeeCal][MLXRunnerBuilder] model container loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStarted)))s mem=\(memMB())")

            // Warm up Metal shader JIT compilation with a vision inference.
            // Text-only warmup (~0.5s) doesn't trigger vision kernels; a real image does.
            // Without this the first user inference takes 60-90s instead of ~15s.
            let warmupStarted = Date()
            print("[SeeCal][MLXRunnerBuilder] warmup start mem=\(memMB())")
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
            print("[SeeCal][MLXRunnerBuilder] warmup done in \(String(format: "%.2f", Date().timeIntervalSince(warmupStarted)))s mem=\(memMB())")

            return { imagePath, prompt in
                guard FileManager.default.fileExists(atPath: imagePath) else {
                    throw MLXRunnerBuilderError.invalidImagePath(imagePath)
                }

                let imageURL = URL(fileURLWithPath: imagePath)
                let input = UserInput(
                    chat: [.user(prompt, images: [.url(imageURL)])],
                    additionalContext: ["enable_thinking": false]
                )
                let inferStarted = Date()
                print("[SeeCal][MLXRunnerBuilder] infer start imagePath=\(imagePath) mem=\(memMB())")
                print("[SeeCal][MLXRunnerBuilder] prompt (\(prompt.count) chars): \(prompt)")
                print("[SeeCal][MLXRunnerBuilder] params: maxTokens=\(config.maxOutputTokens) temp=\(config.temperature) repPenalty=1.1 enable_thinking=false")

                let parameters = GenerateParameters(
                    maxTokens: config.maxOutputTokens,
                    temperature: Float(config.temperature),
                    repetitionPenalty: 1.1
                )
                var generated = ""
                let prepareStarted = Date()
                let prepared = try await modelContainer.prepare(input: input)
                let tokenCount = prepared.text.tokens.shape.reduce(1, *)
                print("[SeeCal][MLXRunnerBuilder] prepare done in \(String(format: "%.2f", Date().timeIntervalSince(prepareStarted)))s tokens=\(tokenCount) mem=\(memMB())")
                let generateStarted = Date()
                let stream = try await modelContainer.generate(input: prepared, parameters: parameters)
                for await event in stream {
                    if case let .chunk(text) = event {
                        generated.append(text)
                    }
                }
                print("[SeeCal][MLXRunnerBuilder] generate done in \(String(format: "%.2f", Date().timeIntervalSince(generateStarted)))s mem=\(memMB())")

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
                            print("[SeeCal][MLXRunnerBuilder] repaired truncated JSON: added \(bracketDepth)] and \(braceDepth)}")
                        }
                    }
                }

                print("[SeeCal][MLXRunnerBuilder] infer done total \(String(format: "%.2f", Date().timeIntervalSince(inferStarted)))s chars=\(text.count)")
                print("[SeeCal][MLXRunnerBuilder] raw output (\(text.count) chars): \(text)")
                return text
            }
        } catch {
            memMonitor.cancel()
            logger.error("Failed to build MLX runner for model id: \(config.modelPath, privacy: .public). Error: \(String(describing: error), privacy: .public)")
            print("[SeeCal][MLXRunnerBuilder] model load FAILED mem=\(memMB()) error=\(String(describing: error))")
            throw error
        }
        #else
        logger.error("MLX packages are not linked in app target")
        print("[SeeCal][MLXRunnerBuilder] MLX packages not linked")
        throw MLXRunnerBuilderError.mlxPackagesNotLinked
        #endif
    }

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
        logger.log("Local model preflight. Existing required files: \(existing.joined(separator: ", "), privacy: .public)")
        print("[SeeCal][MLXRunnerBuilder] preflight existing files: \(existing.joined(separator: ", "))")
        if !missing.isEmpty {
            logger.error("Local model preflight missing files: \(missing.joined(separator: ", "), privacy: .public)")
            print("[SeeCal][MLXRunnerBuilder] preflight missing files: \(missing.joined(separator: ", "))")
        }

        do {
            let topLevel = try fm.contentsOfDirectory(atPath: path).sorted()
            logger.log("Local model directory entries (\(topLevel.count, privacy: .public)): \(topLevel.joined(separator: ", "), privacy: .public)")
            print("[SeeCal][MLXRunnerBuilder] top-level entries count=\(topLevel.count)")
        } catch {
            logger.error("Failed to read local model directory entries. Error: \(String(describing: error), privacy: .public)")
            print("[SeeCal][MLXRunnerBuilder] failed to list directory: \(String(describing: error))")
        }
    }

    private static func validateLocalModelType(path: String) {
        let configURL = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            print("[SeeCal][MLXRunnerBuilder] could not read config.json at \(configURL.path)")
            return
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelType = object["model_type"] as? String
        else {
            print("[SeeCal][MLXRunnerBuilder] could not parse model_type from config.json")
            return
        }

        print("[SeeCal][MLXRunnerBuilder] detected model_type=\(modelType)")
    }
}
