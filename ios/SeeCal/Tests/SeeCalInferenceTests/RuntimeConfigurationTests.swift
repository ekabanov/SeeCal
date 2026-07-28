import XCTest
@testable import SeeCalInference

final class RuntimeConfigurationTests: XCTestCase {
    func testDefaultConfigUsesRecommendedQwenModelID() throws {
        let config = QwenRuntimeConfig()
        XCTAssertEqual(config.modelPath, QwenRuntimeConfig.recommendedModelID)
        XCTAssertNil(config.visualSpecialistModelPath)
    }

    func testLoadsValidConfig() throws {
        let json = """
        {
          "modelPath": "mlx-community/Qwen3.5-4B-MLX-4bit",
          "adapterPath": "/models/adapters_v3",
          "runtimePolicy": "mlx_only",
          "maxOutputTokens": 512,
          "temperature": 0.1,
          "timeoutSeconds": 8,
          "maxAttemptsPerRuntime": 1
        }
        """

        let config = try RuntimeConfigLoader.load(from: json)
        XCTAssertEqual(config.modelPath, "mlx-community/Qwen3.5-4B-MLX-4bit")
        XCTAssertEqual(config.runtimePolicy, .mlxOnly)
        XCTAssertNil(config.visualSpecialistModelPath)
    }

    func testLoadsConditionedVisualSpecialistPath() throws {
        let json = """
        {
          "modelPath": "mlx-community/Qwen3.5-4B-MLX-4bit",
          "adapterPath": "/models/adapters",
          "visualSpecialistModelPath": "/models/SeeCalVisualSpecialist.mlmodelc",
          "runtimePolicy": "mlx_only",
          "maxOutputTokens": 1536,
          "temperature": 0.1,
          "timeoutSeconds": 180,
          "maxAttemptsPerRuntime": 1
        }
        """

        let config = try RuntimeConfigLoader.load(from: json)
        XCTAssertEqual(
            config.visualSpecialistModelPath,
            "/models/SeeCalVisualSpecialist.mlmodelc"
        )
    }

    func testRejectsVisualSpecialistWithoutConditionedAdapter() {
        let config = QwenRuntimeConfig(
            visualSpecialistModelPath: "/models/SeeCalVisualSpecialist.mlmodelc"
        )
        XCTAssertThrowsError(try config.validated()) { error in
            XCTAssertEqual(error as? RuntimeConfigError, .missingAdapterForVisualSpecialist)
        }
    }

    func testRejectsInvalidTemperature() {
        let json = """
        {
          "modelPath": "mlx-community/Qwen3.5-4B-MLX-4bit",
          "runtimePolicy": "mlx_only",
          "maxOutputTokens": 512,
          "temperature": 3.0,
          "timeoutSeconds": 8,
          "maxAttemptsPerRuntime": 1
        }
        """

        XCTAssertThrowsError(try RuntimeConfigLoader.load(from: json))
    }

    func testRejectsInvalidTimeout() {
        let json = """
        {
          "modelPath": "mlx-community/Qwen3.5-4B-MLX-4bit",
          "runtimePolicy": "mlx_only",
          "maxOutputTokens": 512,
          "temperature": 0.1,
          "timeoutSeconds": 0.5,
          "maxAttemptsPerRuntime": 1
        }
        """

        XCTAssertThrowsError(try RuntimeConfigLoader.load(from: json))
    }
}
