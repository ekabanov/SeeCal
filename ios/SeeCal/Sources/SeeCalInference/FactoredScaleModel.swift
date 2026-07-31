import CoreGraphics
import CoreImage
@preconcurrency import CoreML
import Foundation
import SeeCalDiagnostics
import SeeCalDomain

public protocol ScalePredicting: Sendable {
    func predict(imagePath: String) async throws -> FactoredScalePrediction
}

public enum CoreMLScaleError: Error, Equatable, CustomStringConvertible {
    case modelNotFound(String)
    case imageNotFound(String)
    case imageDecodeFailed
    case missingMassOutput
    case invalidMassOutputShape([Int])

    public var description: String {
        switch self {
        case let .modelNotFound(path): return "SCALE model not found at \(path)"
        case let .imageNotFound(path): return "SCALE input image not found at \(path)"
        case .imageDecodeFailed: return "SCALE could not decode the input image"
        case .missingMassOutput: return "SCALE output is missing mass_log1p"
        case let .invalidMassOutputShape(shape):
            return "SCALE mass_log1p shape must be [1, 3], got \(shape)"
        }
    }
}

/// Total-mass-only MobileNet exported by `python -m visual_specialist.scale`.
public actor CoreMLScalePredictor: ScalePredicting {
    private static let imageSize = 224
    private static let resizeShortSide = 232
    private static let means: [Float] = [0.485, 0.456, 0.406]
    private static let standardDeviations: [Float] = [0.229, 0.224, 0.225]

    private let model: MLModel
    private let calibrationMarginGrams: Double
    private let context = CIContext(options: [.cacheIntermediates: false])

    public init(modelPath: String, calibrationMarginGrams: Double) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw CoreMLScaleError.modelNotFound(modelPath)
        }
        guard calibrationMarginGrams.isFinite, calibrationMarginGrams >= 0 else {
            throw FactoredPipelineError.invalidScaleInterval
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(
            contentsOf: URL(fileURLWithPath: modelPath, isDirectory: true),
            configuration: configuration
        )
        self.calibrationMarginGrams = calibrationMarginGrams
    }

    public func predict(imagePath: String) async throws -> FactoredScalePrediction {
        guard FileManager.default.fileExists(atPath: imagePath) else {
            throw CoreMLScaleError.imageNotFound(imagePath)
        }
        guard var image = CIImage(
            contentsOf: URL(fileURLWithPath: imagePath),
            options: [.applyOrientationProperty: true]
        ) else {
            throw CoreMLScaleError.imageDecodeFailed
        }
        let extent = image.extent
        image = image.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        )
        let scale = CGFloat(Self.resizeShortSide) / min(extent.width, extent.height)
        image = image.applyingFilter(
            "CILanczosScaleTransform",
            parameters: [
                kCIInputScaleKey: scale,
                kCIInputAspectRatioKey: 1.0,
            ]
        )
        let scaledExtent = image.extent
        let crop = CGRect(
            x: scaledExtent.midX - CGFloat(Self.imageSize) / 2,
            y: scaledExtent.midY - CGFloat(Self.imageSize) / 2,
            width: CGFloat(Self.imageSize),
            height: CGFloat(Self.imageSize)
        )
        image = image.cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))

        var pixels = [UInt8](repeating: 0, count: Self.imageSize * Self.imageSize * 4)
        context.render(
            image,
            toBitmap: &pixels,
            rowBytes: Self.imageSize * 4,
            bounds: CGRect(x: 0, y: 0, width: Self.imageSize, height: Self.imageSize),
            format: .RGBA8,
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        let input = try MLMultiArray(
            shape: [1, 3, NSNumber(value: Self.imageSize), NSNumber(value: Self.imageSize)],
            dataType: .float32
        )
        let destination = input.dataPointer.bindMemory(
            to: Float32.self,
            capacity: input.count
        )
        let planeSize = Self.imageSize * Self.imageSize
        for pixelIndex in 0..<planeSize {
            let sourceIndex = pixelIndex * 4
            for channel in 0..<3 {
                let unit = Float(pixels[sourceIndex + channel]) / 255
                destination[channel * planeSize + pixelIndex] =
                    (unit - Self.means[channel]) / Self.standardDeviations[channel]
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: input)
        ])
        let output = try await model.prediction(from: provider)
        guard let mass = output.featureValue(for: "mass_log1p")?.multiArrayValue else {
            throw CoreMLScaleError.missingMassOutput
        }
        let shape = mass.shape.map(\.intValue)
        guard shape == [1, 3] else {
            throw CoreMLScaleError.invalidMassOutputShape(shape)
        }
        let decoded = (0..<3).map { quantile in
            let logValue = mass[[0, NSNumber(value: quantile)]].doubleValue
            return max(0, expm1(min(10, max(0, logValue))))
        }
        // Quantile crossing can occur in unconstrained heads. Ordering the
        // endpoints preserves an honest interval while keeping P50 as the point
        // estimate clamped inside it.
        let low = max(0, min(decoded[0], decoded[2]) - calibrationMarginGrams)
        let high = max(decoded[0], decoded[2]) + calibrationMarginGrams
        let estimate = min(high, max(low, decoded[1]))
        return try FactoredScalePrediction(
            massGrams: .init(estimate: estimate, low: low, high: high)
        )
    }
}

public struct VisualSpecialistScaleAdapter: ScalePredicting {
    private let specialist: any VisualSpecialistPredicting

    public init(specialist: any VisualSpecialistPredicting) {
        self.specialist = specialist
    }

    public func predict(imagePath: String) async throws -> FactoredScalePrediction {
        let prediction = try await specialist.predict(imagePath: imagePath)
        return try FactoredScalePrediction(massGrams: prediction.massG)
    }
}

public protocol FoodIdentifying: Sendable {
    func identify(request: FoodScanRequest) async throws -> FoodIdentification
}

public struct QwenFoodIdentifier: FoodIdentifying {
    private let engine: any NativeQwenVisionEngine

    public init(engine: any NativeQwenVisionEngine) {
        self.engine = engine
    }

    public func identify(request: FoodScanRequest) async throws -> FoodIdentification {
        let raw = try await engine.run(
            imagePath: request.imagePath,
            prompt: FactoredIdentificationPrompt.text(userHint: request.userHint)
        )
        return try IdentificationJSONParser.parseStrict(raw)
    }
}

public enum FactoredInferenceOutput: Equatable, Sendable {
    case notFood
    case meal(FactoredMeal)
}

/// Stage-3 shape kept off the shipping path until all model/device gates pass.
/// IDENTIFY and SCALE execute concurrently, then deterministic resolution and
/// assembly run without another model call.
public struct FactoredNutritionInferencePipeline: Sendable {
    private let identifier: any FoodIdentifying
    private let scalePredictor: any ScalePredicting
    private let resolver: any NutritionResolving

    public init(
        identifier: any FoodIdentifying,
        scalePredictor: any ScalePredicting,
        resolver: any NutritionResolving
    ) {
        self.identifier = identifier
        self.scalePredictor = scalePredictor
        self.resolver = resolver
    }

    public func infer(request: FoodScanRequest) async throws -> FactoredInferenceOutput {
        async let identificationTask = identifier.identify(request: request)
        async let scaleTask = scalePredictor.predict(imagePath: request.imagePath)
        let identification = try await identificationTask
        if identification.notFood {
            _ = try? await scaleTask
            return .notFood
        }
        let scale = try await scaleTask
        var resolutions = [NutritionResolution]()
        for item in identification.items {
            resolutions.append(try await resolver.resolve(name: item.name))
        }
        return .meal(
            try FactoredAssembler.assemble(
                identification: identification,
                scale: scale,
                resolutions: resolutions
            )
        )
    }
}

/// Shipping adapter from the factored pipeline to the app's existing scan
/// runtime contract. The richer factored result remains available inside the
/// pipeline; the current review UI consumes its resolved point estimates and
/// lets the user correct names, amounts, and nutrition before saving.
public struct FactoredNutritionRuntime: InferenceRuntime {
    public let name = "factored_qwen_scale_resolve"
    public let modelFamily = "qwen3.5-factored"

    private let pipeline: FactoredNutritionInferencePipeline

    public init(pipeline: FactoredNutritionInferencePipeline) {
        self.pipeline = pipeline
    }

    public func isAvailable() async -> Bool {
        true
    }

    public func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        switch try await pipeline.infer(request: request) {
        case .notFood:
            throw InferenceError.notFood
        case let .meal(meal):
            guard let result = meal.foodScanResult else {
                let unresolved = meal.items
                    .filter { $0.nutrition == nil }
                    .map(\.identification.name)
                throw InferenceError.humanInputRequired(
                    recognizedItems: meal.items.map(\.identification.name),
                    unresolvedItems: unresolved
                )
            }
            SeeCalDiagnostics.record(
                .notice,
                category: "inference",
                name: "factored_assembly_succeeded",
                fields: [
                    "item_count": String(meal.items.count),
                    "confirmation_reasons": meal.confirmationReasons
                        .map(\.rawValue)
                        .sorted()
                        .joined(separator: ","),
                ]
            )
            return try result.validated()
        }
    }
}
