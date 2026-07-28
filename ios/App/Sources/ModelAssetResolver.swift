import Foundation
import os

enum ModelAssetResolver {
    private static let logger = Logger(subsystem: "SeeCal", category: "ModelAssetResolver")

    // Add the model folder to app resources at:
    // Models/mlx-community/Qwen3.5-4B-MLX-4bit
    private static let bundledSubdirectory = "Models/mlx-community"
    private static let bundledModelFolderName = "Qwen3.5-4B-MLX-4bit"

    // Add the LoRA adapter folder (output of convert_adapter_for_swift.py,
    // containing adapter_config.json + adapters.safetensors) to app resources at:
    // Models/adapters
    private static let bundledAdapterSubdirectory = "Models"
    private static let bundledAdapterFolderName = "adapters"
    private static let adapterRequiredFiles = ["adapter_config.json", "adapters.safetensors"]
    private static let conditionedAdapterVersion = "v8-conditioned"
    private static let bundledSpecialistSubdirectory = "Models/visual-specialist"
    private static let bundledSpecialistFolderName = "SeeCalVisualSpecialist.mlmodelc"

    static func resolveModelPath() -> String {
        if let bundledPath = bundledModelPath() {
            logger.log("Using bundled model path: \(bundledPath, privacy: .public)")
            print("[SeeCal][ModelAssetResolver] using bundled path: \(bundledPath)")
            return bundledPath
        }

        if let discoveredPath = discoverBundledModelPath() {
            logger.log("Using discovered bundled model path: \(discoveredPath, privacy: .public)")
            print("[SeeCal][ModelAssetResolver] using discovered path: \(discoveredPath)")
            return discoveredPath
        }

#if targetEnvironment(simulator)
        let localDevPath = "/Users/jevgenikabanov/models/mlx-community/Qwen3.5-4B-MLX-4bit"
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: localDevPath, isDirectory: &isDirectory), isDirectory.boolValue {
            logger.log("Bundled model not found; using simulator fallback path: \(localDevPath, privacy: .public)")
            print("[SeeCal][ModelAssetResolver] using simulator fallback path: \(localDevPath)")
            return localDevPath
        }
#else
        // Device: look in Documents (for development side-loading via Finder file sharing)
        // and Application Support (for production download-on-first-launch).
        let fm = FileManager.default
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let candidates: [URL] = [
            documents.appendingPathComponent("Models/mlx-community/\(bundledModelFolderName)", isDirectory: true),
            documents.appendingPathComponent("\(bundledModelFolderName)", isDirectory: true),
            appSupport.appendingPathComponent("Models/mlx-community/\(bundledModelFolderName)", isDirectory: true),
        ]
        for candidate in candidates {
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: candidate.path, isDirectory: &isDir), isDir.boolValue {
                logger.log("Using device model path: \(candidate.path, privacy: .public)")
                print("[SeeCal][ModelAssetResolver] using device path: \(candidate.path)")
                return candidate.path
            }
        }
#endif

        // Return intended bundled path so downstream error is explicit.
        let expected = Bundle.main.bundleURL
            .appendingPathComponent(bundledSubdirectory, isDirectory: true)
            .appendingPathComponent(bundledModelFolderName, isDirectory: true)
            .path
        logger.error("Bundled model folder is missing. Expected at: \(expected, privacy: .public)")
        print("[SeeCal][ModelAssetResolver] bundled folder missing, expected: \(expected)")
        return expected
    }

    /// Resolves the LoRA adapter directory using the same lookup order as the model:
    /// bundled resources, then (device) Documents / Application Support.
    /// Returns nil when no adapter is present — the app then runs the base model.
    /// If a directory is found it is returned as-is; MLXQwen35RunnerBuilder fails
    /// loudly if a configured adapter cannot actually be loaded.
    static func resolveAdapterPath() -> String? {
        if let bundledPath = bundledAdapterPath() {
            logger.log("Using bundled adapter path: \(bundledPath, privacy: .public)")
            print("[SeeCal][ModelAssetResolver] using bundled adapter path: \(bundledPath)")
            return bundledPath
        }

#if targetEnvironment(simulator)
        // Keep the selected conditioned adapter in sync with
        // SHIPPING_ADAPTER in ios/App/copy_weights.sh.
        let localDevPaths = [
            "/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/ml/adapters_v8_numeric_4b_swift"
        ]
        for localDevPath in localDevPaths where isConditionedAdapterDirectory(localDevPath) {
            logger.log("Bundled adapter not found; using simulator fallback adapter path: \(localDevPath, privacy: .public)")
            print("[SeeCal][ModelAssetResolver] using simulator fallback adapter path: \(localDevPath)")
            return localDevPath
        }
#else
        // Device: mirror the model lookup — Documents (Finder side-loading) and
        // Application Support (download-on-first-launch).
        let fm = FileManager.default
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let candidates: [URL] = [
            documents.appendingPathComponent("\(bundledAdapterSubdirectory)/\(bundledAdapterFolderName)", isDirectory: true),
            documents.appendingPathComponent("\(bundledAdapterFolderName)", isDirectory: true),
            appSupport.appendingPathComponent("\(bundledAdapterSubdirectory)/\(bundledAdapterFolderName)", isDirectory: true),
        ]
        for candidate in candidates where isConditionedAdapterDirectory(candidate.path) {
            logger.log("Using device adapter path: \(candidate.path, privacy: .public)")
            print("[SeeCal][ModelAssetResolver] using device adapter path: \(candidate.path)")
            return candidate.path
        }
#endif

        logger.log("No LoRA adapter found; running base model")
        print("[SeeCal][ModelAssetResolver] no adapter found, running base model")
        return nil
    }

    /// Resolve the compiled Core ML visual specialist paired with the
    /// conditioned v8 adapter. Unlike an optional legacy adapter, this asset is
    /// part of the model contract: callers receive the expected path even when
    /// it is missing so bootstrap fails explicitly.
    static func resolveVisualSpecialistModelPath() -> String {
        if let base = Bundle.main.resourceURL?
            .appendingPathComponent(bundledSpecialistSubdirectory, isDirectory: true) {
            let bundled = base
                .appendingPathComponent(bundledSpecialistFolderName, isDirectory: true)
                .path
            if isDirectory(bundled) {
                logger.log("Using bundled visual specialist: \(bundled, privacy: .public)")
                return bundled
            }
        }

#if targetEnvironment(simulator)
        let localDevPath =
            "/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/ml/runs/visual-specialist/deployment/SeeCalVisualSpecialist.mlmodelc"
        if isDirectory(localDevPath) {
            logger.log("Using simulator visual specialist: \(localDevPath, privacy: .public)")
            return localDevPath
        }
#else
        let fm = FileManager.default
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let candidates = [
            documents.appendingPathComponent(
                "\(bundledSpecialistSubdirectory)/\(bundledSpecialistFolderName)",
                isDirectory: true
            ),
            appSupport.appendingPathComponent(
                "\(bundledSpecialistSubdirectory)/\(bundledSpecialistFolderName)",
                isDirectory: true
            ),
        ]
        for candidate in candidates where isDirectory(candidate.path) {
            logger.log("Using device visual specialist: \(candidate.path, privacy: .public)")
            return candidate.path
        }
#endif

        let expected = Bundle.main.bundleURL
            .appendingPathComponent(bundledSpecialistSubdirectory, isDirectory: true)
            .appendingPathComponent(bundledSpecialistFolderName, isDirectory: true)
            .path
        logger.error("Visual specialist is missing. Expected at: \(expected, privacy: .public)")
        return expected
    }

    private static func bundledAdapterPath() -> String? {
        guard
            let base = Bundle.main.resourceURL?
                .appendingPathComponent(bundledAdapterSubdirectory, isDirectory: true)
        else {
            return nil
        }

        let adapterPath = base.appendingPathComponent(bundledAdapterFolderName, isDirectory: true).path
        return isConditionedAdapterDirectory(adapterPath) ? adapterPath : nil
    }

    private static func isAdapterDirectory(_ path: String) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        return adapterRequiredFiles.allSatisfy {
            fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    private static func isConditionedAdapterDirectory(_ path: String) -> Bool {
        guard isAdapterDirectory(path) else { return false }
        let configURL = URL(fileURLWithPath: path, isDirectory: true)
            .appendingPathComponent("adapter_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = object["seecal_adapter_version"] as? String else {
            return false
        }
        return version == conditionedAdapterVersion
    }

    private static func isDirectory(_ path: String) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    private static func bundledModelPath() -> String? {
        guard
            let base = Bundle.main.resourceURL?
                .appendingPathComponent(bundledSubdirectory, isDirectory: true)
        else {
            return nil
        }

        let modelPath = base.appendingPathComponent(bundledModelFolderName, isDirectory: true).path
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDirectory)
        return exists && isDirectory.boolValue ? modelPath : nil
    }

    private static func discoverBundledModelPath() -> String? {
        guard let resourceURL = Bundle.main.resourceURL else {
            return nil
        }

        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(at: resourceURL, includingPropertiesForKeys: nil) else {
            return nil
        }

        for case let url as URL in enumerator {
            guard url.lastPathComponent == "model.safetensors" else {
                continue
            }

            let candidateDir = url.deletingLastPathComponent()
            let requiredFiles = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "processor_config.json"
            ]
            let hasAll = requiredFiles.allSatisfy {
                fileManager.fileExists(atPath: candidateDir.appendingPathComponent($0).path)
            }
            if hasAll {
                return candidateDir.path
            }
        }

        return nil
    }
}
