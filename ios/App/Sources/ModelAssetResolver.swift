import Foundation
import SeeCalDiagnostics

enum ModelAssetResolver {
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
    private static let factoredAdapterVersion = "factored-e3-foodseg-c1"
    private static let supportedAdapterVersions = [
        conditionedAdapterVersion,
        factoredAdapterVersion,
    ]
    private static let bundledSpecialistSubdirectory = "Models/visual-specialist"
    private static let bundledSpecialistFolderName = "SeeCalVisualSpecialist.mlmodelc"
    private static let bundledScaleSubdirectory = "Models/scale"
    private static let bundledScaleFolderName = "SeeCalScaleC1.mlmodelc"
    private static let factoredScaleCalibrationMarginGrams = 129.55438232421875
    private static let bundledNutritionDatabase = "Models/nutrition/seecal-nutrition.sqlite"

    struct InferenceAssets {
        let adapterPath: String?
        let visualSpecialistModelPath: String?
        let factoredScaleModelPath: String?
        let factoredScaleCalibrationMarginGrams: Double
    }

    static func resolveInferenceAssets() -> InferenceAssets {
        let adapterPath = resolveAdapterPath()
        switch adapterPath.flatMap(adapterVersion) {
        case factoredAdapterVersion:
            return InferenceAssets(
                adapterPath: adapterPath,
                visualSpecialistModelPath: nil,
                factoredScaleModelPath: resolveFactoredScaleModelPath(),
                factoredScaleCalibrationMarginGrams: Self.factoredScaleCalibrationMarginGrams
            )
        case conditionedAdapterVersion:
            return InferenceAssets(
                adapterPath: adapterPath,
                visualSpecialistModelPath: resolveVisualSpecialistModelPath(),
                factoredScaleModelPath: nil,
                factoredScaleCalibrationMarginGrams: 0
            )
        default:
            return InferenceAssets(
                adapterPath: adapterPath,
                visualSpecialistModelPath: nil,
                factoredScaleModelPath: nil,
                factoredScaleCalibrationMarginGrams: 0
            )
        }
    }

    static func resolveModelPath() -> String {
        if let bundledPath = bundledModelPath() {
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "model_resolved",
                fields: ["source": "bundled"]
            )
            return bundledPath
        }

        if let discoveredPath = discoverBundledModelPath() {
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "model_resolved",
                fields: ["source": "discovered_bundle"]
            )
            return discoveredPath
        }

#if targetEnvironment(simulator)
        let localDevPath = "/Users/jevgenikabanov/models/mlx-community/Qwen3.5-4B-MLX-4bit"
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: localDevPath, isDirectory: &isDirectory), isDirectory.boolValue {
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "model_resolved",
                fields: ["source": "simulator_fallback"]
            )
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
                SeeCalDiagnostics.record(
                    .notice,
                    category: "model_assets",
                    name: "model_resolved",
                    fields: ["source": "device_side_load"]
                )
                return candidate.path
            }
        }
#endif

        // Return intended bundled path so downstream error is explicit.
        let expected = Bundle.main.bundleURL
            .appendingPathComponent(bundledSubdirectory, isDirectory: true)
            .appendingPathComponent(bundledModelFolderName, isDirectory: true)
            .path
        SeeCalDiagnostics.record(
            .fault,
            category: "model_assets",
            name: "model_not_found"
        )
        return expected
    }

    /// Resolves the LoRA adapter directory using the same lookup order as the model:
    /// bundled resources, then (device) Documents / Application Support.
    /// Returns nil when no supported, correctly stamped adapter is present.
    static func resolveAdapterPath() -> String? {
        if let bundledPath = bundledAdapterPath() {
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "adapter_resolved",
                fields: ["source": "bundled"]
            )
            return bundledPath
        }

#if targetEnvironment(simulator)
        // Keep the selected conditioned adapter in sync with
        // SHIPPING_ADAPTER in ios/App/copy_weights.sh.
        let localDevPaths = [
            "/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/ml/runs/factored/e3-v2-foodseg-1024-warm-start-12000-lr1e-5/adapter_swift",
            "/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/ml/adapters_v8_numeric_4b_swift"
        ]
        for localDevPath in localDevPaths where isSupportedAdapterDirectory(localDevPath) {
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "adapter_resolved",
                fields: ["source": "simulator_fallback"]
            )
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
        for candidate in candidates where isSupportedAdapterDirectory(candidate.path) {
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "adapter_resolved",
                fields: ["source": "device_side_load"]
            )
            return candidate.path
        }
#endif

        SeeCalDiagnostics.record(
            .error,
            category: "model_assets",
            name: "supported_adapter_not_found"
        )
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
                SeeCalDiagnostics.record(
                    .notice,
                    category: "model_assets",
                    name: "visual_specialist_resolved",
                    fields: ["source": "bundled"]
                )
                return bundled
            }
        }

#if targetEnvironment(simulator)
        let localDevPath =
            "/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/ml/runs/visual-specialist/deployment/SeeCalVisualSpecialist.mlmodelc"
        if isDirectory(localDevPath) {
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "visual_specialist_resolved",
                fields: ["source": "simulator_fallback"]
            )
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
            SeeCalDiagnostics.record(
                .notice,
                category: "model_assets",
                name: "visual_specialist_resolved",
                fields: ["source": "device_side_load"]
            )
            return candidate.path
        }
#endif

        let expected = Bundle.main.bundleURL
            .appendingPathComponent(bundledSpecialistSubdirectory, isDirectory: true)
            .appendingPathComponent(bundledSpecialistFolderName, isDirectory: true)
            .path
        SeeCalDiagnostics.record(
            .fault,
            category: "model_assets",
            name: "visual_specialist_not_found"
        )
        return expected
    }

    /// Resolve SCALE C1 for the factored device-test stack. As with the v8
    /// specialist, return the expected bundled path when absent so bootstrap
    /// reports a concrete missing-model error.
    static func resolveFactoredScaleModelPath() -> String {
        if let base = Bundle.main.resourceURL?
            .appendingPathComponent(bundledScaleSubdirectory, isDirectory: true) {
            let bundled = base
                .appendingPathComponent(bundledScaleFolderName, isDirectory: true)
                .path
            if isDirectory(bundled) {
                SeeCalDiagnostics.record(
                    .notice,
                    category: "model_assets",
                    name: "factored_scale_resolved",
                    fields: ["source": "bundled"]
                )
                return bundled
            }
        }

#if targetEnvironment(simulator)
        let localDevPath =
            "/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/ml/runs/factored/deployment/compiled/SeeCalScaleC1.mlmodelc"
        if isDirectory(localDevPath) {
            return localDevPath
        }
#else
        let fm = FileManager.default
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let candidates = [
            documents.appendingPathComponent(
                "\(bundledScaleSubdirectory)/\(bundledScaleFolderName)",
                isDirectory: true
            ),
            appSupport.appendingPathComponent(
                "\(bundledScaleSubdirectory)/\(bundledScaleFolderName)",
                isDirectory: true
            ),
        ]
        if let candidate = candidates.first(where: { isDirectory($0.path) }) {
            return candidate.path
        }
#endif

        SeeCalDiagnostics.record(
            .fault,
            category: "model_assets",
            name: "factored_scale_not_found"
        )
        return Bundle.main.bundleURL
            .appendingPathComponent(bundledScaleSubdirectory, isDirectory: true)
            .appendingPathComponent(bundledScaleFolderName, isDirectory: true)
            .path
    }

    static func resolveNutritionDatabaseURL() -> URL? {
        if let resourceURL = Bundle.main.resourceURL {
            let bundled = resourceURL.appendingPathComponent(bundledNutritionDatabase)
            if FileManager.default.fileExists(atPath: bundled.path) {
                return bundled
            }
        }

#if targetEnvironment(simulator)
        let local = URL(
            fileURLWithPath:
                "/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal/ml/datasets/fdc/seecal-nutrition.sqlite"
        )
        if FileManager.default.fileExists(atPath: local.path) {
            return local
        }
#else
        let fileManager = FileManager.default
        let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let candidates = [
            documents.appendingPathComponent(bundledNutritionDatabase),
            appSupport.appendingPathComponent(bundledNutritionDatabase),
        ]
        if let candidate = candidates.first(where: {
            fileManager.fileExists(atPath: $0.path)
        }) {
            return candidate
        }
#endif

        SeeCalDiagnostics.record(
            .error,
            category: "model_assets",
            name: "nutrition_database_not_found"
        )
        return nil
    }

    private static func bundledAdapterPath() -> String? {
        guard
            let base = Bundle.main.resourceURL?
                .appendingPathComponent(bundledAdapterSubdirectory, isDirectory: true)
        else {
            return nil
        }

        let adapterPath = base.appendingPathComponent(bundledAdapterFolderName, isDirectory: true).path
        return isSupportedAdapterDirectory(adapterPath) ? adapterPath : nil
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

    private static func adapterVersion(_ path: String) -> String? {
        guard isAdapterDirectory(path) else { return nil }
        let configURL = URL(fileURLWithPath: path, isDirectory: true)
            .appendingPathComponent("adapter_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = object["seecal_adapter_version"] as? String else {
            return nil
        }
        return version
    }

    private static func isSupportedAdapterDirectory(_ path: String) -> Bool {
        guard let version = adapterVersion(path) else { return false }
        return supportedAdapterVersions.contains(version)
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
