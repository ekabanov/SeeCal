import XCTest
@testable import SeeCalInference

final class ModelInfoResolverTests: XCTestCase {
    private var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelInfoResolverTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    // MARK: - Not-bundled fallback (spec §8)

    func testNilPathsResolveToNotBundledPlaceholder() {
        let info = ModelInfoResolver.resolve(modelPath: nil, adapterPath: nil)
        XCTAssertEqual(info, ModelInfo.notBundled)
        XCTAssertNil(info.adapterVersionLabel)
        XCTAssertNil(info.quantizationLabel)
        XCTAssertEqual(info.modelLabel, "Qwen3.5-4B")
    }

    func testEmptyStringPathsAlsoResolveToNilFields() {
        let info = ModelInfoResolver.resolve(modelPath: "", adapterPath: "")
        XCTAssertNil(info.adapterVersionLabel)
        XCTAssertNil(info.quantizationLabel)
    }

    // MARK: - Adapter version: directory-name suffix parsing

    func testVersionSuffixParsesTrailingVNumber() {
        XCTAssertEqual(ModelInfoResolver.versionSuffix(fromDirectoryName: "adapters_v5_swift"), "v5")
        XCTAssertEqual(ModelInfoResolver.versionSuffix(fromDirectoryName: "adapters_v6"), "v6")
        XCTAssertEqual(ModelInfoResolver.versionSuffix(fromDirectoryName: "adapters_v12_fixed"), "v12")
    }

    func testVersionSuffixCaseInsensitive() {
        XCTAssertEqual(ModelInfoResolver.versionSuffix(fromDirectoryName: "Adapters_V7_Swift"), "v7")
    }

    func testVersionSuffixNilForGenericProductionFolderName() {
        // The production bundle folder is named generically ("adapters", no
        // version suffix) — must not fabricate a version.
        XCTAssertNil(ModelInfoResolver.versionSuffix(fromDirectoryName: "adapters"))
    }

    // MARK: - Adapter version: resolve() end to end

    func testAdapterVersionResolvedFromDirectoryNameWhenNoConfigVersionKey() throws {
        let adapterDir = tempDirectory.appendingPathComponent("adapters_v5_swift", isDirectory: true)
        try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)
        try #"{"fine_tune_type":"lora","num_layers":32}"#
            .write(to: adapterDir.appendingPathComponent("adapter_config.json"), atomically: true, encoding: .utf8)

        let info = ModelInfoResolver.resolve(modelPath: nil, adapterPath: adapterDir.path)
        XCTAssertEqual(info.adapterVersionLabel, "v5")
    }

    func testAdapterVersionPrefersExplicitConfigVersionOverDirectoryName() throws {
        // Directory name says v4, but the config carries a more specific/newer
        // version string — the config should win.
        let adapterDir = tempDirectory.appendingPathComponent("adapters_v4_swift", isDirectory: true)
        try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)
        try #"{"fine_tune_type":"lora","version":"v4.1-patched"}"#
            .write(to: adapterDir.appendingPathComponent("adapter_config.json"), atomically: true, encoding: .utf8)

        let info = ModelInfoResolver.resolve(modelPath: nil, adapterPath: adapterDir.path)
        XCTAssertEqual(info.adapterVersionLabel, "v4.1-patched")
    }

    func testAdapterVersionNilWhenDirectoryMissingEntirely() {
        let missingPath = tempDirectory.appendingPathComponent("does-not-exist", isDirectory: true).path
        let info = ModelInfoResolver.resolve(modelPath: nil, adapterPath: missingPath)
        XCTAssertNil(info.adapterVersionLabel)
    }

    func testAdapterVersionNilWhenNameHasNoVersionAndConfigHasNoVersionKey() throws {
        let adapterDir = tempDirectory.appendingPathComponent("adapters", isDirectory: true)
        try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)
        try #"{"fine_tune_type":"lora","num_layers":32}"#
            .write(to: adapterDir.appendingPathComponent("adapter_config.json"), atomically: true, encoding: .utf8)

        let info = ModelInfoResolver.resolve(modelPath: nil, adapterPath: adapterDir.path)
        XCTAssertNil(info.adapterVersionLabel)
    }

    // MARK: - Quantization

    func testQuantizationLabelReadFromModelConfigBits() throws {
        let modelDir = tempDirectory.appendingPathComponent("Qwen3.5-4B-MLX-4bit", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5","quantization":{"group_size":64,"bits":4,"mode":"affine"}}"#
            .write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let info = ModelInfoResolver.resolve(modelPath: modelDir.path, adapterPath: nil)
        XCTAssertEqual(info.quantizationLabel, "4-bit quantized")
    }

    func testQuantizationLabelNilWhenConfigMissingQuantizationField() throws {
        let modelDir = tempDirectory.appendingPathComponent("unquantized-model", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5"}"#
            .write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let info = ModelInfoResolver.resolve(modelPath: modelDir.path, adapterPath: nil)
        XCTAssertNil(info.quantizationLabel)
    }

    func testQuantizationLabelNilWhenModelPathDoesNotExist() {
        let missingPath = tempDirectory.appendingPathComponent("no-such-model", isDirectory: true).path
        let info = ModelInfoResolver.resolve(modelPath: missingPath, adapterPath: nil)
        XCTAssertNil(info.quantizationLabel)
    }

    // MARK: - modelLabel is always the fixed product name

    func testModelLabelIsAlwaysQwen35_4B() {
        XCTAssertEqual(ModelInfoResolver.resolve(modelPath: nil, adapterPath: nil).modelLabel, "Qwen3.5-4B")
        XCTAssertEqual(ModelInfoResolver.resolve(modelPath: "/some/path", adapterPath: "/other/path").modelLabel, "Qwen3.5-4B")
    }
}
