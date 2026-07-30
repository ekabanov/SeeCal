import CoreGraphics
import CoreImage
@preconcurrency import CoreML
import Foundation

public struct VisualSpecialistInterval: Equatable, Sendable {
    public let estimate: Double
    public let low: Double
    public let high: Double

    public init(estimate: Double, low: Double, high: Double) {
        self.estimate = estimate
        self.low = low
        self.high = high
    }
}

public struct VisualSpecialistPrediction: Equatable, Sendable {
    public let massG: VisualSpecialistInterval
    public let calories: VisualSpecialistInterval
    public let proteinG: VisualSpecialistInterval
    public let fatG: VisualSpecialistInterval
    public let carbsG: VisualSpecialistInterval

    public init(
        massG: VisualSpecialistInterval,
        calories: VisualSpecialistInterval,
        proteinG: VisualSpecialistInterval,
        fatG: VisualSpecialistInterval,
        carbsG: VisualSpecialistInterval
    ) {
        self.massG = massG
        self.calories = calories
        self.proteinG = proteinG
        self.fatG = fatG
        self.carbsG = carbsG
    }
}

public protocol VisualSpecialistPredicting: Sendable {
    func predict(imagePath: String) async throws -> VisualSpecialistPrediction
}

public enum VisualSpecialistError: Error, Equatable, CustomStringConvertible {
    case modelNotFound(String)
    case imageNotFound(String)
    case imageDecodeFailed
    case missingNumericOutput
    case invalidNumericOutputShape([Int])

    public var description: String {
        switch self {
        case let .modelNotFound(path):
            return "Visual specialist model not found at \(path)"
        case let .imageNotFound(path):
            return "Visual specialist input image not found at \(path)"
        case .imageDecodeFailed:
            return "Visual specialist could not decode the input image"
        case .missingNumericOutput:
            return "Visual specialist output is missing numeric_log1p"
        case let .invalidNumericOutputShape(shape):
            return "Visual specialist numeric_log1p shape must be [1, 5, 3], got \(shape)"
        }
    }
}

/// MobileNetV3 specialist exported by `ml/visual_specialist/export_coreml.py`.
///
/// Its preprocessing is byte-for-behaviour equivalent to the evaluation
/// pipeline: RGB, resize the short edge to 232, center-crop 224, ImageNet
/// normalization, and NCHW Float32 input.
public actor CoreMLVisualSpecialist: VisualSpecialistPredicting {
    private static let imageSize = 224
    private static let resizeShortSide = 232
    private static let means: [Float] = [0.485, 0.456, 0.406]
    private static let standardDeviations: [Float] = [0.229, 0.224, 0.225]

    // Split-conformal additive interval widening fitted on the untouched
    // Nutrition5K calibration fold. Field order matches numeric_log1p.
    private static let calibrationMargins: [Double] = [
        3.1726465225219727,
        5.746240234375023,
        0.46084418296813956,
        0.7484344482421861,
        0.3998540878295902,
    ]

    private let model: MLModel
    private let context = CIContext(options: [.cacheIntermediates: false])

    public init(modelPath: String) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw VisualSpecialistError.modelNotFound(modelPath)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(
            contentsOf: URL(fileURLWithPath: modelPath, isDirectory: true),
            configuration: configuration
        )
    }

    public func predict(imagePath: String) async throws -> VisualSpecialistPrediction {
        guard FileManager.default.fileExists(atPath: imagePath) else {
            throw VisualSpecialistError.imageNotFound(imagePath)
        }
        guard var image = CIImage(
            contentsOf: URL(fileURLWithPath: imagePath),
            options: [.applyOrientationProperty: true]
        ) else {
            throw VisualSpecialistError.imageDecodeFailed
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
        let destination = input.dataPointer.bindMemory(to: Float32.self, capacity: input.count)
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
        // The actor serializes access to the Core ML model.
        let output = try await model.prediction(from: provider)
        guard let numeric = output.featureValue(for: "numeric_log1p")?.multiArrayValue else {
            throw VisualSpecialistError.missingNumericOutput
        }
        let shape = numeric.shape.map(\.intValue)
        guard shape == [1, 5, 3] else {
            throw VisualSpecialistError.invalidNumericOutputShape(shape)
        }

        func interval(field: Int) -> VisualSpecialistInterval {
            func decoded(_ quantile: Int) -> Double {
                let logValue = numeric[[0, NSNumber(value: field), NSNumber(value: quantile)]].doubleValue
                // Match visual_specialist.predict: clamp in log space before
                // expm1 so corrupt/extreme output can never overflow the prompt.
                return max(0, expm1(min(10, max(0, logValue))))
            }
            let margin = Self.calibrationMargins[field]
            return VisualSpecialistInterval(
                estimate: decoded(1),
                low: max(0, decoded(0) - margin),
                high: decoded(2) + margin
            )
        }

        return VisualSpecialistPrediction(
            massG: interval(field: 0),
            calories: interval(field: 1),
            proteinG: interval(field: 2),
            fatG: interval(field: 3),
            carbsG: interval(field: 4)
        )
    }
}

/// Defers Core ML model construction until the first photo analysis. The
/// compiled specialist is small, but keeping every model allocation out of app
/// launch makes opening straight into the camera consistently cheap.
public actor LazyCoreMLVisualSpecialist: VisualSpecialistPredicting {
    private let modelPath: String
    private var specialist: CoreMLVisualSpecialist?

    public init(modelPath: String) {
        self.modelPath = modelPath
    }

    public func predict(imagePath: String) async throws -> VisualSpecialistPrediction {
        let specialist: CoreMLVisualSpecialist
        if let loaded = self.specialist {
            specialist = loaded
        } else {
            let loaded = try CoreMLVisualSpecialist(modelPath: modelPath)
            self.specialist = loaded
            specialist = loaded
        }
        return try await specialist.predict(imagePath: imagePath)
    }
}

public enum VisualSpecialistPromptRenderer {
    public static let prefix =
        "Auxiliary visual measurement (fallible; use as evidence, not ground truth):"

    public static func render(_ prediction: VisualSpecialistPrediction?) -> String {
        guard let prediction else {
            return "\(prefix)\n{\"available\":false}"
        }

        func number(_ value: Double) -> String {
            String(format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), value)
        }
        func field(_ value: VisualSpecialistInterval) -> String {
            "{\"estimate\":\(number(value.estimate)),\"low\":\(number(value.low)),\"high\":\(number(value.high))}"
        }

        return "\(prefix)\n" +
            "{\"available\":true," +
            "\"mass_g\":\(field(prediction.massG))," +
            "\"calories\":\(field(prediction.calories))," +
            "\"protein_g\":\(field(prediction.proteinG))," +
            "\"fat_g\":\(field(prediction.fatG))," +
            "\"carbs_g\":\(field(prediction.carbsG))}"
    }
}
