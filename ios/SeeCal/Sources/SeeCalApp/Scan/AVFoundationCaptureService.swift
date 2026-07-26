#if os(iOS)
import AVFoundation
import CoreMotion
import SwiftUI
import UIKit

/// Real-camera `CaptureService` (spec §5): AVFoundation photo capture with a
/// live preview layer, plus CoreMotion device gravity for the level indicator.
/// Depth/LiDAR capture is deliberately NOT wired yet (`supportsDepthCapture`
/// is false) — it arrives with depth-plan Task D5; until then the distance chip
/// and LiDAR pill never render.
///
/// This type only compiles on iOS; the macOS test host and the simulator use
/// `MockCaptureService` (see `makeDefaultCaptureService()`).
@MainActor
public final class AVFoundationCaptureService: CaptureService {
    public let supportsDepthCapture = false

    private let camera = CameraSessionBox()
    private let motion = MotionBox()

    public init() {}

    public var authorizationStatus: CameraAuthorization {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return .authorized
        case .notDetermined:
            return .notDetermined
        default:
            return .denied
        }
    }

    public func requestAccess() async -> CameraAuthorization {
        if authorizationStatus == .notDetermined {
            _ = await AVCaptureDevice.requestAccess(for: .video)
        }
        return authorizationStatus
    }

    public func startSession() {
        guard authorizationStatus == .authorized else { return }
        camera.start()
    }

    public func stopSession() {
        camera.stop()
        motion.stop()
    }

    public func capturePhoto() async throws -> CapturedPhoto {
        try await withCheckedThrowingContinuation { continuation in
            camera.capturePhoto { result in
                continuation.resume(with: result.map(CapturedPhoto.init(imageData:)))
            }
        }
    }

    public func gravityUpdates() -> AsyncStream<GravityReading> {
        motion.gravityStream()
    }

    public func makePreviewView() -> AnyView {
        AnyView(CameraPreviewView(session: camera.session).ignoresSafeArea())
    }
}

// MARK: - AVCaptureSession box

/// Owns the capture session and photo output; all mutation happens on its own
/// serial queue (AVFoundation requirement — never block the main thread on
/// `startRunning`). `@unchecked Sendable` because the queue is the isolation.
private final class CameraSessionBox: NSObject, AVCapturePhotoCaptureDelegate, @unchecked Sendable {
    let session = AVCaptureSession()

    private let queue = DispatchQueue(label: "seecal.camera.session")
    private let output = AVCapturePhotoOutput()
    private var configured = false
    private var configurationError: Error?
    private var pendingCaptures: [Int64: (Result<Data, Error>) -> Void] = [:]

    func start() {
        queue.async {
            self.configureIfNeeded()
            guard self.configurationError == nil, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stop() {
        queue.async {
            guard self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    func capturePhoto(completion: @escaping @Sendable (Result<Data, Error>) -> Void) {
        queue.async {
            self.configureIfNeeded()
            if let error = self.configurationError {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            if !self.session.isRunning {
                self.session.startRunning()
            }
            let settings = AVCapturePhotoSettings()
            self.pendingCaptures[settings.uniqueID] = { result in
                DispatchQueue.main.async { completion(result) }
            }
            self.output.capturePhoto(with: settings, delegate: self)
        }
    }

    private func configureIfNeeded() {
        guard !configured else { return }
        configured = true

        session.beginConfiguration()
        session.sessionPreset = .photo
        defer { session.commitConfiguration() }

        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input),
            session.canAddOutput(output)
        else {
            configurationError = CaptureServiceError.cameraUnavailable
            return
        }
        session.addInput(input)
        session.addOutput(output)
    }

    // AVCapturePhotoCaptureDelegate — called on the session's internal queue;
    // hop to ours before touching state.
    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        let uniqueID = photo.resolvedSettings.uniqueID
        let data = photo.fileDataRepresentation()
        queue.async {
            guard let completion = self.pendingCaptures.removeValue(forKey: uniqueID) else { return }
            if let error {
                completion(.failure(CaptureServiceError.captureFailed(error.localizedDescription)))
            } else if let data {
                completion(.success(data))
            } else {
                completion(.failure(CaptureServiceError.captureFailed("no image data produced")))
            }
        }
    }
}

// MARK: - CoreMotion box

/// Owns the CMMotionManager. Gravity callbacks are delivered on the main queue
/// and forwarded into the AsyncStream; `@unchecked Sendable` because the manager
/// is only touched from the main queue after init.
private final class MotionBox: @unchecked Sendable {
    private let manager = CMMotionManager()

    func gravityStream() -> AsyncStream<GravityReading> {
        AsyncStream { continuation in
            guard manager.isDeviceMotionAvailable else {
                continuation.finish()
                return
            }
            manager.deviceMotionUpdateInterval = 1.0 / 30.0
            manager.startDeviceMotionUpdates(to: .main) { motion, _ in
                guard let gravity = motion?.gravity else { return }
                continuation.yield(GravityReading(x: gravity.x, y: gravity.y))
            }
            continuation.onTermination = { [weak self] _ in
                self?.manager.stopDeviceMotionUpdates()
            }
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
    }
}

// MARK: - Preview layer wrapper

private struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}
#endif
