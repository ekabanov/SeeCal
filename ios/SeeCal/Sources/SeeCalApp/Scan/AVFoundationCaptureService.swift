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

    /// Watchdog for a single photo capture. A healthy capture completes in well
    /// under a second; this only exists so a capture that will *never* call back
    /// fails loudly instead of hanging the shutter forever. Generous enough to
    /// survive a slow capture under memory pressure.
    private static let captureTimeout: DispatchTimeInterval = .seconds(10)

    private let queue = DispatchQueue(label: "seecal.camera.session")
    private let output = AVCapturePhotoOutput()
    private var configured = false
    private var configurationError: Error?
    /// Queue-isolated; see `PendingCaptureRegistry` for the exactly-once contract.
    private var pendingCaptures = PendingCaptureRegistry()

    override init() {
        super.init()
        // `startRunning()` does not throw: when the media server refuses or tears
        // down the session (seen on device as repeated FigCaptureSourceRemote
        // err=-17281 under memory pressure), AVFoundation reports it *only* via
        // this notification. Without observing it a dead session is completely
        // silent — black preview, no error, and any in-flight capture never calls
        // back.
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(sessionRuntimeError(_:)),
            name: AVCaptureSession.runtimeErrorNotification,
            object: session
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

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
            let id = settings.uniqueID
            self.pendingCaptures.register(id: id) { result in
                DispatchQueue.main.async { completion(result) }
            }
            // Watchdog: whichever of {delegate, timeout} claims the capture first
            // wins, so the continuation is resumed exactly once.
            self.queue.asyncAfter(deadline: .now() + Self.captureTimeout) { [weak self] in
                guard let self, let timedOut = self.pendingCaptures.claim(id: id) else { return }
                timedOut(.failure(CaptureServiceError.captureFailed(
                    "camera did not respond in time — try again")))
            }
            self.output.capturePhoto(with: settings, delegate: self)
        }
    }

    /// Configures the session on first use. Only marks itself done on SUCCESS:
    /// an earlier version set `configured = true` up front, so a single transient
    /// failure (plausible under memory pressure, when acquiring the device can
    /// fail) latched `cameraUnavailable` for the whole process lifetime — every
    /// later attempt short-circuited and the camera never came back.
    private func configureIfNeeded() {
        guard !configured else { return }
        configurationError = nil

        session.beginConfiguration()
        session.sessionPreset = .photo
        var succeeded = false
        defer {
            session.commitConfiguration()
            // Retry on the next call unless this attempt actually wired things up.
            configured = succeeded
        }

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
        succeeded = true
    }

    /// The session hit a runtime error. Fail every in-flight capture so the
    /// shutter reports a problem instead of hanging, then make one best-effort
    /// restart attempt — these errors are usually transient (the device log shows
    /// AVFoundation recovering on its own).
    @objc private func sessionRuntimeError(_ notification: Notification) {
        let reason = (notification.userInfo?[AVCaptureSessionErrorKey] as? NSError)?
            .localizedDescription ?? "capture session failed"
        queue.async {
            for completion in self.pendingCaptures.claimAll() {
                completion(.failure(CaptureServiceError.captureFailed(reason)))
            }
            guard !self.session.isRunning else { return }
            self.session.startRunning()
        }
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
            // nil => the watchdog (or a session failure) already completed this one.
            guard let completion = self.pendingCaptures.claim(id: uniqueID) else { return }
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
