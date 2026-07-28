import Foundation

public enum RuntimeConfigError: Error, Equatable, CustomStringConvertible {
    case emptyModelPath
    case emptyVisualSpecialistModelPath
    case missingAdapterForVisualSpecialist
    case invalidMaxOutputTokens(Int)
    case invalidTemperature(Double)
    case invalidTimeoutSeconds(Double)
    case invalidMaxAttemptsPerRuntime(Int)

    public var description: String {
        switch self {
        case .emptyModelPath:
            return "modelPath must not be empty"
        case .emptyVisualSpecialistModelPath:
            return "visualSpecialistModelPath must be nil or a non-empty path"
        case .missingAdapterForVisualSpecialist:
            return "a visual specialist requires its conditioned adapterPath"
        case let .invalidMaxOutputTokens(value):
            return "maxOutputTokens must be in 64...4096, got \(value)"
        case let .invalidTemperature(value):
            return "temperature must be in 0...2, got \(value)"
        case let .invalidTimeoutSeconds(value):
            return "timeoutSeconds must be in 1...300, got \(value)"
        case let .invalidMaxAttemptsPerRuntime(value):
            return "maxAttemptsPerRuntime must be in 1...3, got \(value)"
        }
    }
}

public struct QwenRuntimeConfig: Codable, Equatable, Sendable {
    public enum RuntimePolicy: String, Codable, CaseIterable, Sendable {
        case mlxOnly = "mlx_only"
        case mlxWithMNNFallback = "mlx_with_mnn_fallback"
    }

    public var modelPath: String
    public var adapterPath: String?
    /// Compiled Core ML specialist (`.mlmodelc`) used by conditioned adapters.
    /// Nil preserves the legacy, unconditioned prompt contract.
    public var visualSpecialistModelPath: String?
    public var runtimePolicy: RuntimePolicy
    public var maxOutputTokens: Int
    public var temperature: Double
    public var timeoutSeconds: Double
    public var maxAttemptsPerRuntime: Int

    public static let recommendedModelID = "mlx-community/Qwen3.5-4B-MLX-4bit"

    public init(
        modelPath: String = QwenRuntimeConfig.recommendedModelID,
        adapterPath: String? = nil,
        visualSpecialistModelPath: String? = nil,
        runtimePolicy: RuntimePolicy = .mlxOnly,
        maxOutputTokens: Int = 512,
        temperature: Double = 0.1,
        timeoutSeconds: Double = 8,
        maxAttemptsPerRuntime: Int = 1
    ) {
        self.modelPath = modelPath
        self.adapterPath = adapterPath
        self.visualSpecialistModelPath = visualSpecialistModelPath
        self.runtimePolicy = runtimePolicy
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
        self.timeoutSeconds = timeoutSeconds
        self.maxAttemptsPerRuntime = maxAttemptsPerRuntime
    }

    @discardableResult
    public func validated() throws -> QwenRuntimeConfig {
        if modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw RuntimeConfigError.emptyModelPath
        }
        if let visualSpecialistModelPath,
           visualSpecialistModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw RuntimeConfigError.emptyVisualSpecialistModelPath
        }
        if visualSpecialistModelPath != nil {
            guard let adapterPath,
                  !adapterPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw RuntimeConfigError.missingAdapterForVisualSpecialist
            }
        }
        if !(64...4096).contains(maxOutputTokens) {
            throw RuntimeConfigError.invalidMaxOutputTokens(maxOutputTokens)
        }
        if !(0...2).contains(temperature) {
            throw RuntimeConfigError.invalidTemperature(temperature)
        }
        if !(1...300).contains(timeoutSeconds) {
            throw RuntimeConfigError.invalidTimeoutSeconds(timeoutSeconds)
        }
        if !(1...3).contains(maxAttemptsPerRuntime) {
            throw RuntimeConfigError.invalidMaxAttemptsPerRuntime(maxAttemptsPerRuntime)
        }
        return self
    }
}

public enum RuntimeConfigLoader {
    public static func load(from data: Data) throws -> QwenRuntimeConfig {
        let config = try JSONDecoder().decode(QwenRuntimeConfig.self, from: data)
        return try config.validated()
    }

    public static func load(from jsonString: String) throws -> QwenRuntimeConfig {
        guard let data = jsonString.data(using: .utf8) else {
            throw NSError(domain: "RuntimeConfigLoader", code: 1)
        }
        return try load(from: data)
    }
}
