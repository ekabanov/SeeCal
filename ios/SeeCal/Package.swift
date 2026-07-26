// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SeeCal",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "SeeCalDomain", targets: ["SeeCalDomain"]),
        .library(name: "SeeCalInference", targets: ["SeeCalInference"]),
        .library(name: "SeeCalPersistence", targets: ["SeeCalPersistence"]),
        .library(name: "SeeCalApp", targets: ["SeeCalApp"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main")
    ],
    targets: [
        .target(
            name: "SeeCalDomain"
        ),
        .target(
            name: "SeeCalInference",
            dependencies: [
                "SeeCalDomain",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm")
            ]
        ),
        .target(
            name: "SeeCalPersistence",
            dependencies: ["SeeCalDomain"]
        ),
        .target(
            name: "SeeCalApp",
            dependencies: ["SeeCalDomain", "SeeCalInference", "SeeCalPersistence"]
        ),
        .testTarget(
            name: "SeeCalDomainTests",
            dependencies: ["SeeCalDomain"],
            resources: [
                .copy("Fixtures/valid_scan_result.json"),
                .copy("Fixtures/invalid_scan_result_missing_items.json")
            ]
        ),
        .testTarget(
            name: "SeeCalInferenceTests",
            dependencies: ["SeeCalInference", "SeeCalDomain"]
        ),
        .testTarget(
            name: "SeeCalAppTests",
            dependencies: ["SeeCalApp", "SeeCalInference", "SeeCalPersistence", "SeeCalDomain"]
        ),
        .testTarget(
            name: "SeeCalPersistenceTests",
            dependencies: ["SeeCalPersistence", "SeeCalDomain"]
        )
    ]
)
