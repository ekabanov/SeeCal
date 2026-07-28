// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SeeCal",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "SeeCalDiagnostics", targets: ["SeeCalDiagnostics"]),
        .library(name: "SeeCalDomain", targets: ["SeeCalDomain"]),
        .library(name: "SeeCalInference", targets: ["SeeCalInference"]),
        .library(name: "SeeCalPersistence", targets: ["SeeCalPersistence"]),
        .library(name: "SeeCalApp", targets: ["SeeCalApp"])
    ],
    dependencies: [
        // Fork of ml-explore/mlx-swift pinned to the same commit we already
        // resolved (70346ae) plus one line: FMT_CONSTEVAL= defined on the Cmlx
        // target. Xcode 26 / clang 21 otherwise fails to build the vendored fmt
        // from the GUI ("call to consteval function ... is not a constant
        // expression"); the flag can't be injected into a package target from
        // the consuming project, so it must live in mlx-swift's own manifest.
        // Same package identity ("mlx-swift"), so this also overrides the
        // transitive mlx-swift that mlx-swift-lm pulls in. Drop the fork and
        // return to ml-explore/mlx-swift once upstream bumps fmt >= 12.1.0.
        .package(url: "https://github.com/ekabanov/mlx-swift", branch: "seecal-fmt"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main")
    ],
    targets: [
        .target(
            name: "SeeCalDiagnostics"
        ),
        .target(
            name: "SeeCalDomain"
        ),
        .target(
            name: "SeeCalInference",
            dependencies: [
                "SeeCalDiagnostics",
                "SeeCalDomain",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm")
            ]
        ),
        .target(
            name: "SeeCalPersistence",
            dependencies: ["SeeCalDiagnostics", "SeeCalDomain"]
        ),
        .target(
            name: "SeeCalApp",
            dependencies: ["SeeCalDiagnostics", "SeeCalDomain", "SeeCalInference", "SeeCalPersistence"]
        ),
        .testTarget(
            name: "SeeCalDiagnosticsTests",
            dependencies: ["SeeCalDiagnostics"]
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
