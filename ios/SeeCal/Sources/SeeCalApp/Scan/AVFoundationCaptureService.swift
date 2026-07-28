#if os(iOS)
import AVFoundation
import CoreMotion
import SeeCalDiagnostics
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
        let before = authorizationStatus
        if authorizationStatus == .notDetermined {
            _ = await AVCaptureDevice.requestAccess(for: .video)
        }
        let after = authorizationStatus
        SeeCalDiagnostics.record(
            .notice,
            category: "camera",
            name: "authorization_resolved",
            fields: [
                "before": Self.authorizationName(before),
                "after": Self.authorizationName(after)
            ]
        )
        return after
    }

    public func startSession() {
        guard authorizationStatus == .authorized else {
            SeeCalDiagnostics.record(
                .error,
                category: "camera",
                name: "session_start_blocked_by_authorization",
                fields: ["authorization": Self.authorizationName(authorizationStatus)]
            )
            return
        }
        SeeCalDiagnostics.record(.info, category: "camera", name: "session_start_requested")
        camera.start()
    }

    public func stopSession() {
        SeeCalDiagnostics.record(.info, category: "camera", name: "session_stop_requested")
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

    public func barcodeUpdates() -> AsyncStream<DetectedBarcode> {
        camera.barcodeStream()
    }

    public func makePreviewView() -> AnyView {
        AnyView(CameraPreviewView(session: camera.session).ignoresSafeArea())
    }

    private static func authorizationName(_ authorization: CameraAuthorization) -> String {
        switch authorization {
        case .notDetermined: return "not_determined"
        case .authorized: return "authorized"
        case .denied: return "denied"
        }
    }
}

// MARK: - AVCaptureSession box

/// Owns the capture session and photo output; all mutation happens on its own
/// serial queue (AVFoundation requirement — never block the main thread on
/// `startRunning`). `@unchecked Sendable` because the queue is the isolation.
private final class CameraSessionBox: NSObject, AVCapturePhotoCaptureDelegate, AVCaptureMetadataOutputObjectsDelegate, @unchecked Sendable {
    let session = AVCaptureSession()

    /// Watchdog for a single photo capture. A healthy capture completes in well
    /// under a second; this only exists so a capture that will *never* call back
    /// fails loudly instead of hanging the shutter forever. Generous enough to
    /// survive a slow capture under memory pressure.
    private static let captureTimeout: DispatchTimeInterval = .seconds(10)

    private let queue = DispatchQueue(label: "seecal.camera.session")
    private let output = AVCapturePhotoOutput()
    private let metadataOutput = AVCaptureMetadataOutput()
    private let barcodeStreams = BarcodeStreamRegistry()
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
            if let error = self.configurationError {
                SeeCalDiagnostics.record(
                    .error,
                    category: "camera",
                    name: "session_start_configuration_failed",
                    fields: SeeCalDiagnostics.errorFields(error)
                )
                return
            }
            guard !self.session.isRunning else {
                SeeCalDiagnostics.record(.debug, category: "camera", name: "session_already_running")
                return
            }
            self.session.startRunning()
            SeeCalDiagnostics.record(
                self.session.isRunning ? .notice : .error,
                category: "camera",
                name: self.session.isRunning ? "session_started" : "session_start_did_not_run"
            )
        }
    }

    func stop() {
        queue.async {
            guard self.session.isRunning else { return }
            self.session.stopRunning()
            SeeCalDiagnostics.record(.info, category: "camera", name: "session_stopped")
        }
    }

    func capturePhoto(completion: @escaping @Sendable (Result<Data, Error>) -> Void) {
        queue.async {
            self.configureIfNeeded()
            if let error = self.configurationError {
                SeeCalDiagnostics.record(
                    .error,
                    category: "camera",
                    name: "capture_blocked_by_configuration",
                    fields: SeeCalDiagnostics.errorFields(error)
                )
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            if !self.session.isRunning {
                self.session.startRunning()
            }
            let settings = AVCapturePhotoSettings()
            let id = settings.uniqueID
            SeeCalDiagnostics.record(
                .info,
                category: "camera",
                name: "photo_capture_submitted",
                fields: ["capture_id": String(id), "session_running": String(self.session.isRunning)]
            )
            self.pendingCaptures.register(id: id) { result in
                DispatchQueue.main.async { completion(result) }
            }
            // Watchdog: whichever of {delegate, timeout} claims the capture first
            // wins, so the continuation is resumed exactly once.
            self.queue.asyncAfter(deadline: .now() + Self.captureTimeout) { [weak self] in
                guard let self, let timedOut = self.pendingCaptures.claim(id: id) else { return }
                SeeCalDiagnostics.record(
                    .error,
                    category: "camera",
                    name: "photo_capture_timed_out",
                    fields: ["capture_id": String(id)]
                )
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
            session.canAddOutput(output),
            session.canAddOutput(metadataOutput)
        else {
            configurationError = CaptureServiceError.cameraUnavailable
            SeeCalDiagnostics.record(.error, category: "camera", name: "session_configuration_unavailable")
            return
        }
        session.addInput(input)
        session.addOutput(output)
        session.addOutput(metadataOutput)
        metadataOutput.setMetadataObjectsDelegate(self, queue: queue)
        metadataOutput.metadataObjectTypes = [.ean8, .ean13, .upce, .code128]
        succeeded = true
        SeeCalDiagnostics.record(.notice, category: "camera", name: "session_configuration_succeeded")
    }

    /// The session hit a runtime error. Fail every in-flight capture so the
    /// shutter reports a problem instead of hanging, then make one best-effort
    /// restart attempt — these errors are usually transient (the device log shows
    /// AVFoundation recovering on its own).
    @objc private func sessionRuntimeError(_ notification: Notification) {
        let sessionError = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
        let reason = sessionError?.localizedDescription ?? "capture session failed"
        let errorDomain = sessionError?.domain
        let errorCode = sessionError.map { String($0.code) }
        queue.async {
            let completions = self.pendingCaptures.claimAll()
            var fields = ["pending_capture_count": String(completions.count)]
            if let errorDomain {
                fields["error_domain"] = errorDomain
            }
            if let errorCode {
                fields["error_code"] = errorCode
            }
            SeeCalDiagnostics.record(
                .error,
                category: "camera",
                name: "session_runtime_error",
                fields: fields
            )
            for completion in completions {
                completion(.failure(CaptureServiceError.captureFailed(reason)))
            }
            guard !self.session.isRunning else { return }
            self.session.startRunning()
            SeeCalDiagnostics.record(
                self.session.isRunning ? .notice : .error,
                category: "camera",
                name: self.session.isRunning ? "session_restart_succeeded" : "session_restart_failed"
            )
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
                SeeCalDiagnostics.record(
                    .error,
                    category: "camera",
                    name: "photo_processing_failed",
                    fields: ["capture_id": String(uniqueID)]
                        .merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
                )
                completion(.failure(CaptureServiceError.captureFailed(error.localizedDescription)))
            } else if let data {
                SeeCalDiagnostics.record(
                    .notice,
                    category: "camera",
                    name: "photo_processing_succeeded",
                    fields: ["capture_id": String(uniqueID), "bytes": String(data.count)]
                )
                completion(.success(data))
            } else {
                SeeCalDiagnostics.record(
                    .error,
                    category: "camera",
                    name: "photo_processing_returned_no_data",
                    fields: ["capture_id": String(uniqueID)]
                )
                completion(.failure(CaptureServiceError.captureFailed("no image data produced")))
            }
        }
    }

    func barcodeStream() -> AsyncStream<DetectedBarcode> {
        barcodeStreams.stream()
    }

    func metadataOutput(
        _ output: AVCaptureMetadataOutput,
        didOutput metadataObjects: [AVMetadataObject],
        from connection: AVCaptureConnection
    ) {
        guard let code = metadataObjects
            .compactMap({ $0 as? AVMetadataMachineReadableCodeObject })
            .first,
              let value = code.stringValue
        else { return }

        let symbology: BarcodeSymbology
        switch code.type {
        case .ean8: symbology = .ean8
        case .ean13: symbology = .ean13
        case .upce: symbology = .upce
        case .code128: symbology = .code128
        default: symbology = .unknown
        }
        barcodeStreams.yield(DetectedBarcode(value: value, symbology: symbology))
    }
}

private final class BarcodeStreamRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var continuations: [UUID: AsyncStream<DetectedBarcode>.Continuation] = [:]

    func stream() -> AsyncStream<DetectedBarcode> {
        let id = UUID()
        return AsyncStream { continuation in
            lock.lock()
            continuations[id] = continuation
            lock.unlock()
            continuation.onTermination = { [weak self] _ in
                self?.remove(id)
            }
        }
    }

    func yield(_ barcode: DetectedBarcode) {
        lock.lock()
        let current = Array(continuations.values)
        lock.unlock()
        for continuation in current {
            continuation.yield(barcode)
        }
    }

    private func remove(_ id: UUID) {
        lock.lock()
        continuations[id] = nil
        lock.unlock()
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
