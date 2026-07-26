import XCTest
@testable import SeeCalInference

final class RuntimeGateEvaluatorTests: XCTestCase {
    func testGateAPassesForQwen35NativeAndLatencyBudget() {
        let evaluator = RuntimeGateEvaluator()
        let metric = GateMetric(
            runtimeName: "mlx_swift",
            modelFamily: "qwen3.5-native-multimodal",
            medianLatencyMs: 4200,
            p95LatencyMs: 5800,
            maxMemoryMB: 4200,
            validSchemaRate: 1.0,
            meanAbsoluteErrorCalories: 68
        )

        let decision = evaluator.gateA(metric: metric)
        XCTAssertTrue(decision.passed)
    }

    func testGateBFailsWhenFamilyDiffers() {
        let evaluator = RuntimeGateEvaluator()
        let primary = GateMetric(
            runtimeName: "mlx_swift",
            modelFamily: "qwen3.5-native-multimodal",
            medianLatencyMs: 4300,
            p95LatencyMs: 5900,
            maxMemoryMB: 4200,
            validSchemaRate: 0.99,
            meanAbsoluteErrorCalories: 62
        )
        let fallback = GateMetric(
            runtimeName: "mnn",
            modelFamily: "qwen3-text-only",
            medianLatencyMs: 3900,
            p95LatencyMs: 5200,
            maxMemoryMB: 3400,
            validSchemaRate: 0.99,
            meanAbsoluteErrorCalories: 65
        )

        let decision = evaluator.gateB(primary: primary, fallback: fallback)
        XCTAssertFalse(decision.passed)
    }

    func testGateCPassesOnlyWhenCoreMLMeetsBudgets() {
        let evaluator = RuntimeGateEvaluator()
        let metric = GateMetric(
            runtimeName: "coreml_spike",
            modelFamily: "qwen3.5-native-multimodal",
            medianLatencyMs: 5900,
            p95LatencyMs: 9000,
            maxMemoryMB: 5300,
            validSchemaRate: 0.995,
            meanAbsoluteErrorCalories: 72
        )

        let decision = evaluator.gateC(metric: metric)
        XCTAssertTrue(decision.passed)
    }
}
