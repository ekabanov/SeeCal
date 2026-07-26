import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum InferenceImagePreprocessor {
    static func downsampledJPEG(
        from data: Data,
        maxPixelSize: Int = 768,
        jpegQuality: CGFloat = 0.72
    ) throws -> Data {
        let sourceOptions: [CFString: Any] = [
            kCGImageSourceShouldCache: false
        ]
        guard let source = CGImageSourceCreateWithData(data as CFData, sourceOptions as CFDictionary) else {
            throw NSError(domain: "InferenceImagePreprocessor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid image data"])
        }

        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: false,
            kCGImageSourceShouldCache: false,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize
        ]
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbnailOptions as CFDictionary) else {
            throw NSError(domain: "InferenceImagePreprocessor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not create thumbnail"])
        }

        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw NSError(domain: "InferenceImagePreprocessor", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not create output image"])
        }

        let destinationProperties: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: jpegQuality
        ]
        CGImageDestinationAddImage(destination, thumbnail, destinationProperties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw NSError(domain: "InferenceImagePreprocessor", code: 4, userInfo: [NSLocalizedDescriptionKey: "Could not finalize output image"])
        }

        return output as Data
    }
}
