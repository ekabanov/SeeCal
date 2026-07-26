import XCTest
import SeeCalDomain
@testable import SeeCalApp
@testable import SeeCalInference

final class ProductionFactoryTests: XCTestCase {
    func testThrowsWhenPolicyRequiresMNNButRunnerMissing() async throws {
        let config = QwenRuntimeConfig(
            modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            runtimePolicy: .mlxWithMNNFallback
        )

        await XCTAssertThrowsErrorAsync {
            _ = try await MainActor.run {
                try SeeCalProductionFactory.makeViewModel(
                    config: config,
                    mlxRunner: { _, _ in "{}" },
                    mnnRunner: nil
                )
            }
        }
    }

    func testBuildsViewModelForMLXOnlyPolicy() async throws {
        let config = QwenRuntimeConfig(
            modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            runtimePolicy: .mlxOnly
        )

        let viewModel = try await MainActor.run {
            try SeeCalProductionFactory.makeViewModel(
                config: config,
                mlxRunner: { _, _ in
                    """
                    {
                      "total_calories": 100,
                      "protein_g": 10,
                      "fat_g": 2,
                      "carbs_g": 8,
                      "confidence": 0.9,
                      "items": [{"name": "egg", "estimated_grams": 50, "calories": 100, "protein_g": 10, "fat_g": 2, "carbs_g": 8}],
                      "uncertainty_flags": []
                    }
                    """
                }
            )
        }

        XCTAssertNotNil(viewModel)
    }
}

private func XCTAssertThrowsErrorAsync(
    _ expression: @escaping () async throws -> Void,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        try await expression()
        XCTFail("Expected throw", file: file, line: line)
    } catch {
        XCTAssertTrue(true)
    }
}
