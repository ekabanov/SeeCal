import Foundation

/// What Settings §8's "On-device model" card displays: "Qwen3.5-4B · SeeCal
/// adapter <version>" + a quantization figure in the subtitle. Every field
/// other than `modelLabel` is best-effort, read from the bundled model/adapter
/// directories' own config files at runtime (never hardcoded) — see
/// `ModelInfoResolver.resolve`.
public struct ModelInfo: Equatable, Sendable {
    public var modelLabel: String
    /// nil when no adapter is configured, its directory is missing, or no
    /// version could be determined from it — the UI renders this as "not
    /// bundled" rather than fabricating a version number.
    public var adapterVersionLabel: String?
    /// nil when `modelPath` is nil or its `config.json` couldn't be read/
    /// parsed for a `quantization.bits` field.
    public var quantizationLabel: String?

    public init(modelLabel: String, adapterVersionLabel: String?, quantizationLabel: String?) {
        self.modelLabel = modelLabel
        self.adapterVersionLabel = adapterVersionLabel
        self.quantizationLabel = quantizationLabel
    }

    /// Placeholder for simulator/test environments where no real model/adapter
    /// path is configured at all (spec §8: "fall back to a 'not bundled'
    /// placeholder in simulator/tests — do NOT hardcode a fake version").
    public static let notBundled = ModelInfo(modelLabel: "Qwen3.5-4B", adapterVersionLabel: nil, quantizationLabel: nil)
}

/// Resolves `ModelInfo` from the model/adapter directory paths already
/// produced by (the app target's) `ModelAssetResolver` — this type takes
/// plain path strings so it stays testable without any bundle/resource
/// machinery.
public enum ModelInfoResolver {
    public static func resolve(
        modelPath: String?,
        adapterPath: String?,
        fileManager: FileManager = .default
    ) -> ModelInfo {
        ModelInfo(
            modelLabel: "Qwen3.5-4B",
            adapterVersionLabel: adapterVersionLabel(adapterPath: adapterPath, fileManager: fileManager),
            quantizationLabel: quantizationLabel(modelPath: modelPath, fileManager: fileManager)
        )
    }

    // MARK: - Adapter version

    /// Prefers an explicit version string recorded in `adapter_config.json`
    /// under `seecal_adapter_version` / `adapter_version` / `version` (forward
    /// compatible with a future `convert_adapter_for_swift.py` that stamps
    /// one — see CLAUDE.md's toolchain notes; today's converter does not).
    /// Falls back to parsing a "_v<N>" suffix off the adapter directory's own
    /// name, matching this repo's `adapters_vN[_swift]` convention (e.g.
    /// `adapters_v5_swift` → "v5"). Returns `nil` — never a fabricated
    /// default — when neither source yields a version.
    static func adapterVersionLabel(adapterPath: String?, fileManager: FileManager) -> String? {
        guard let adapterPath, !adapterPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        let directoryURL = URL(fileURLWithPath: adapterPath, isDirectory: true)
        if let configVersion = versionFromAdapterConfig(at: directoryURL, fileManager: fileManager) {
            return configVersion
        }
        return versionSuffix(fromDirectoryName: directoryURL.lastPathComponent)
    }

    static func versionFromAdapterConfig(at directoryURL: URL, fileManager: FileManager) -> String? {
        let configURL = directoryURL.appendingPathComponent("adapter_config.json")
        guard
            fileManager.fileExists(atPath: configURL.path),
            let data = try? Data(contentsOf: configURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        for key in ["seecal_adapter_version", "adapter_version", "version"] {
            if let version = object[key] as? String, !version.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return version
            }
        }
        return nil
    }

    /// Parses a trailing "_v<digits>" component out of directory names like
    /// "adapters_v5_swift" or "adapters_v6" (matching whichever "_v<N>"
    /// occurrence appears LAST, so a hypothetical "adapters_v5_v2_swift" would
    /// still resolve to the meaningful trailing one). A generic name with no
    /// such suffix (e.g. the production bundle folder "adapters") yields nil.
    static func versionSuffix(fromDirectoryName name: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: "_v([0-9]+)", options: [.caseInsensitive]) else {
            return nil
        }
        let range = NSRange(name.startIndex..<name.endIndex, in: name)
        guard
            let match = regex.matches(in: name, options: [], range: range).last,
            let numberRange = Range(match.range(at: 1), in: name)
        else {
            return nil
        }
        return "v" + name[numberRange]
    }

    // MARK: - Quantization

    /// Reads `config.json`'s `quantization.bits` field (e.g.
    /// `{"bits": 4, "group_size": 64, "mode": "affine"}`, as produced by MLX's
    /// quantized model export) → "4-bit quantized".
    static func quantizationLabel(modelPath: String?, fileManager: FileManager) -> String? {
        guard let modelPath, !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        let configURL = URL(fileURLWithPath: modelPath, isDirectory: true).appendingPathComponent("config.json")
        guard
            fileManager.fileExists(atPath: configURL.path),
            let data = try? Data(contentsOf: configURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let quantization = object["quantization"] as? [String: Any],
            let bits = quantization["bits"] as? Int
        else {
            return nil
        }
        return "\(bits)-bit quantized"
    }
}
