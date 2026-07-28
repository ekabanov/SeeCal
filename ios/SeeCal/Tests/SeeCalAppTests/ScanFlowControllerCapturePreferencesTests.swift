import XCTest
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
@testable import SeeCalApp

private struct NoopRuntime: InferenceRuntime {
    let name = "noop"
    let modelFamily = "qwen3.5-native-multimodal"
    func isAvailable() async -> Bool { false }
    func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        _ = request
        throw InferenceError.runtimeUnavailable("none")
    }
}

/// `CameraScreen` reads `ScanFlowController.capturePreferences` (not the whole
/// view model) to gate its coaching overlays. This covers the plumbing: the
/// controller's exposed value always mirrors the owning view model's, live.
final class ScanFlowControllerCapturePreferencesTests: XCTestCase {
    @MainActor
    func testControllerExposesTheViewModelsCapturePreferencesLive() async {
        let viewModel = AppViewModel(
            orchestrator: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            store: InMemoryMealLogStore()
        )
        let controller = ScanFlowController(
            viewModel: viewModel,
            inference: RuntimeOrchestrator(runtimes: [NoopRuntime()]),
            captureService: MockCaptureService()
        )

        // Starts at the shared default (on).
        XCTAssertTrue(controller.capturePreferences.captureCoachingEnabled)

        await viewModel.updateCapturePreferences(
            CapturePreferences(captureCoachingEnabled: false)
        )

        XCTAssertFalse(controller.capturePreferences.captureCoachingEnabled)
    }
}
