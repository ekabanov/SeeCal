import Foundation
import SwiftUI
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// MARK: - Capture types

/// A single captured still (JPEG-encodable data). Depth payloads (LiDAR maps)
/// will extend this when D5 lands; for now a photo is just its image bytes.
public struct CapturedPhoto: Equatable, Sendable {
    public var imageData: Data

    public init(imageData: Data) {
        self.imageData = imageData
    }
}

public enum CameraAuthorization: Equatable, Sendable {
    case notDetermined
    case authorized
    case denied
}

public enum CaptureServiceError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case cameraUnavailable
    case captureFailed(String)

    public var description: String {
        switch self {
        case .cameraUnavailable:
            return "Camera unavailable on this device"
        case let .captureFailed(reason):
            return "Photo capture failed: \(reason)"
        }
    }

    public var errorDescription: String? { description }
}

/// Device-frame gravity sample driving the camera's level indicator (`.level` /
/// `.bubble` in the prototype). When the phone is held flat over the plate
/// (screen up, camera down) gravity is ~(0, 0, −1), so `x`/`y` are ≈ 0; tilting
/// pushes them toward ±1. Only x/y matter for the bubble.
public struct GravityReading: Equatable, Sendable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    public static let flat = GravityReading(x: 0, y: 0)

    /// Tilt tolerance under which the device counts as "flat": bubble centers,
    /// ring turns from amber to neutral, hint flips to "Framed — capture when
    /// ready" (prototype: `level = off < 0.25` on its normalized drift scale).
    public static let levelTolerance = 0.18

    public var offsetMagnitude: Double {
        (x * x + y * y).squareRoot()
    }

    public var isLevel: Bool {
        offsetMagnitude <= Self.levelTolerance
    }
}

// MARK: - CaptureService protocol

/// Abstraction over the camera hardware (spec §5: "Behind a `CaptureService`
/// protocol; simulator/mock implementation returns a bundled sample photo so the
/// whole flow runs in tests and simulator"). Production uses
/// `AVFoundationCaptureService` (iOS device); tests, the simulator, and the macOS
/// test host use `MockCaptureService`.
@MainActor
public protocol CaptureService: AnyObject {
    /// Whether this device can capture depth alongside the photo (LiDAR). Gates
    /// the distance chip and the "LiDAR depth · capturing portion size" pill —
    /// FALSE everywhere for now; depth capture arrives with D5, and until then
    /// those affordances simply don't render (spec §5: "hide chip on non-LiDAR
    /// devices").
    var supportsDepthCapture: Bool { get }

    var authorizationStatus: CameraAuthorization { get }

    /// Requests camera permission if not yet determined; returns the resulting
    /// status either way.
    func requestAccess() async -> CameraAuthorization

    func startSession()
    func stopSession()

    func capturePhoto() async throws -> CapturedPhoto

    /// Continuous device-gravity samples for the level indicator. The stream
    /// finishes when the session stops (or never emits where motion hardware is
    /// unavailable).
    func gravityUpdates() -> AsyncStream<GravityReading>

    /// Barcodes recognized from the same live camera stream. Photo capture
    /// remains available at all times; recognition is an automatic side channel.
    func barcodeUpdates() -> AsyncStream<DetectedBarcode>

    /// The live viewfinder content filling the camera screen. The AVFoundation
    /// implementation returns a preview-layer wrapper; the mock returns a static
    /// stand-in scene so the screen stays exercisable everywhere.
    func makePreviewView() -> AnyView
}

public extension CaptureService {
    func barcodeUpdates() -> AsyncStream<DetectedBarcode> {
        AsyncStream { $0.finish() }
    }
}

// MARK: - Mock implementation

/// Deterministic `CaptureService` for tests, the simulator, and macOS: always
/// authorized, captures a generated sample plate photo instantly, and exposes a
/// manual `sendGravity` pump so tests (and the simulator) can drive the level
/// indicator.
@MainActor
public final class MockCaptureService: CaptureService {
    public let supportsDepthCapture = false

    public private(set) var authorizationStatus: CameraAuthorization
    public private(set) var captureCount = 0
    public private(set) var isSessionRunning = false

    /// Override the next capture's outcome (e.g. force a failure in tests).
    /// `nil` (the default) returns a freshly generated sample photo.
    public var nextCaptureResult: Result<CapturedPhoto, Error>?

    private var gravityContinuations: [UUID: AsyncStream<GravityReading>.Continuation] = [:]
    private var barcodeContinuations: [UUID: AsyncStream<DetectedBarcode>.Continuation] = [:]

    public init(authorizationStatus: CameraAuthorization = .authorized) {
        self.authorizationStatus = authorizationStatus
    }

    public func requestAccess() async -> CameraAuthorization {
        if authorizationStatus == .notDetermined {
            authorizationStatus = .authorized
        }
        return authorizationStatus
    }

    public func startSession() {
        isSessionRunning = true
    }

    public func stopSession() {
        isSessionRunning = false
        for continuation in gravityContinuations.values {
            continuation.finish()
        }
        gravityContinuations.removeAll()
        for continuation in barcodeContinuations.values {
            continuation.finish()
        }
        barcodeContinuations.removeAll()
    }

    public func capturePhoto() async throws -> CapturedPhoto {
        captureCount += 1
        if let nextCaptureResult {
            self.nextCaptureResult = nil
            return try nextCaptureResult.get()
        }
        return CapturedPhoto(imageData: Self.samplePhotoData())
    }

    public func gravityUpdates() -> AsyncStream<GravityReading> {
        let id = UUID()
        return AsyncStream { continuation in
            gravityContinuations[id] = continuation
            continuation.yield(.flat)
        }
    }

    public func barcodeUpdates() -> AsyncStream<DetectedBarcode> {
        let id = UUID()
        return AsyncStream { continuation in
            barcodeContinuations[id] = continuation
        }
    }

    /// Test/simulator hook: pushes a gravity sample to every open stream.
    public func sendGravity(_ reading: GravityReading) {
        for continuation in gravityContinuations.values {
            continuation.yield(reading)
        }
    }

    public func sendBarcode(_ barcode: DetectedBarcode) {
        for continuation in barcodeContinuations.values {
            continuation.yield(barcode)
        }
    }

    public func makePreviewView() -> AnyView {
        AnyView(MockViewfinderView())
    }

    /// A generated overhead "plate on a dark table" JPEG (CoreGraphics, works on
    /// both iOS and the macOS test host) standing in for a real camera frame —
    /// valid image data end to end, so the photo store's downsampling and the
    /// runtime's image loading both exercise their real code paths.
    public static func samplePhotoData() -> Data {
        let width = 480
        let height = 360
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return Data()
        }

        // Dark table backdrop.
        context.setFillColor(CGColor(red: 0.15, green: 0.14, blue: 0.125, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        // Plate.
        let plateRect = CGRect(x: 110, y: 50, width: 260, height: 260)
        context.setFillColor(CGColor(red: 0.93, green: 0.92, blue: 0.89, alpha: 1))
        context.fillEllipse(in: plateRect)
        context.setStrokeColor(CGColor(red: 0.81, green: 0.80, blue: 0.76, alpha: 1))
        context.setLineWidth(3)
        context.strokeEllipse(in: plateRect.insetBy(dx: 28, dy: 28))

        // Food blobs (rice / chicken / broccoli-ish patches).
        context.setFillColor(CGColor(red: 0.92, green: 0.88, blue: 0.78, alpha: 1))
        context.fillEllipse(in: CGRect(x: 150, y: 150, width: 100, height: 80))
        context.setFillColor(CGColor(red: 0.90, green: 0.74, blue: 0.56, alpha: 1))
        context.fillEllipse(in: CGRect(x: 240, y: 100, width: 90, height: 70))
        context.setFillColor(CGColor(red: 0.30, green: 0.48, blue: 0.25, alpha: 1))
        context.fillEllipse(in: CGRect(x: 230, y: 190, width: 70, height: 60))

        guard let image = context.makeImage() else { return Data() }

        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else {
            return Data()
        }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return Data() }
        return output as Data
    }
}

/// The mock's viewfinder: the prototype camera's `.table` radial-gradient scene,
/// so the simulator/macOS camera screen looks like the design rather than a
/// black hole.
private struct MockViewfinderView: View {
    var body: some View {
        RadialGradient(
            colors: [
                Color(hex: 0x3D3A33),
                Color(hex: 0x262420),
                Color(hex: 0x141311)
            ],
            center: UnitPoint(x: 0.5, y: 0.42),
            startRadius: 0,
            endRadius: 420
        )
        .overlay(alignment: .center) {
            ZStack {
                Circle()
                    .fill(Color(hex: 0xECEAE3))
                Circle()
                    .stroke(Color(hex: 0xCFCCC2).opacity(0.7), lineWidth: 2)
                    .padding(26)
            }
            .frame(width: 270, height: 270)
            .offset(y: -40)
        }
        .ignoresSafeArea()
    }
}

// MARK: - Default service factory

/// The platform-appropriate capture service: a real AVFoundation camera on an
/// iOS device, the mock everywhere else (simulator lacks camera hardware; the
/// macOS test host runs the whole flow against the mock).
@MainActor
public func makeDefaultCaptureService() -> CaptureService {
    #if os(iOS) && !targetEnvironment(simulator)
    return AVFoundationCaptureService()
    #else
    return MockCaptureService()
    #endif
}
