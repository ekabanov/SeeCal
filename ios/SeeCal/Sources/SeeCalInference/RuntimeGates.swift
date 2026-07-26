import Foundation
import SeeCalDomain

public struct GateMetric: Codable, Equatable, Sendable {
    public let runtimeName: String
    public let modelFamily: String
    public let medianLatencyMs: Double
    public let p95LatencyMs: Double
    public let maxMemoryMB: Double
    public let validSchemaRate: Double
    public let meanAbsoluteErrorCalories: Double

    public init(
        runtimeName: String,
        modelFamily: String,
        medianLatencyMs: Double,
        p95LatencyMs: Double,
        maxMemoryMB: Double,
        validSchemaRate: Double,
        meanAbsoluteErrorCalories: Double
    ) {
        self.runtimeName = runtimeName
        self.modelFamily = modelFamily
        self.medianLatencyMs = medianLatencyMs
        self.p95LatencyMs = p95LatencyMs
        self.maxMemoryMB = maxMemoryMB
        self.validSchemaRate = validSchemaRate
        self.meanAbsoluteErrorCalories = meanAbsoluteErrorCalories
    }
}

public struct GateDecision: Codable, Equatable, Sendable {
    public let gateName: String
    public let passed: Bool
    public let reason: String

    public init(gateName: String, passed: Bool, reason: String) {
        self.gateName = gateName
        self.passed = passed
        self.reason = reason
    }
}

public struct RuntimeGateEvaluator {
    public init() {}

    public func gateA(metric: GateMetric) -> GateDecision {
        let isQwen35Native = metric.modelFamily == "qwen3.5-native-multimodal"
        let passed = isQwen35Native && metric.validSchemaRate >= 0.99 && metric.medianLatencyMs <= 6000
        let reason = passed
            ? "MLX gate passed with stable schema and acceptable latency"
            : "MLX gate failed: require qwen3.5-native-multimodal + schema>=0.99 + median<=6000ms"
        return GateDecision(gateName: "Gate A (MLX)", passed: passed, reason: reason)
    }

    public func gateB(primary: GateMetric, fallback: GateMetric) -> GateDecision {
        let familyParity = fallback.modelFamily == primary.modelFamily
        let schemaParity = abs(primary.validSchemaRate - fallback.validSchemaRate) <= 0.01
        let maeDelta = abs(primary.meanAbsoluteErrorCalories - fallback.meanAbsoluteErrorCalories)
        let passed = familyParity && schemaParity && maeDelta <= 25

        let reason = passed
            ? "MNN fallback parity validated"
            : "MNN fallback failed parity requirements (family/schema/MAE)"

        return GateDecision(gateName: "Gate B (MNN)", passed: passed, reason: reason)
    }

    public func gateC(metric: GateMetric) -> GateDecision {
        let passed = metric.modelFamily == "qwen3.5-native-multimodal"
            && metric.validSchemaRate >= 0.99
            && metric.medianLatencyMs <= 6000
            && metric.maxMemoryMB <= 5500

        let reason = passed
            ? "CoreML spike cleared R&D feasibility gate"
            : "CoreML spike did not meet feasibility constraints"

        return GateDecision(gateName: "Gate C (CoreML R&D)", passed: passed, reason: reason)
    }
}
