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
/// view model) to gate its coaching overlays and the LiDAR-gated depth
/// affordances (spec §8 scope item: "LiDAR toggle → plumb a preference the
/// (currently dormant) depth path will read"; "coaching toggle → CameraScreen
/// hides level/hint overlays when off"). This covers the plumbing: the
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

        // Starts at the shared default (both on).
        XCTAssertTrue(controller.capturePreferences.useLiDARDepth)
        XCTAssertTrue(controller.capturePreferences.captureCoachingEnabled)

        await viewModel.updateCapturePreferences(
            CapturePreferences(useLiDARDepth: false, captureCoachingEnabled: false)
        )

        XCTAssertFalse(controller.capturePreferences.useLiDARDepth)
        XCTAssertFalse(controller.capturePreferences.captureCoachingEnabled)
    }
}
