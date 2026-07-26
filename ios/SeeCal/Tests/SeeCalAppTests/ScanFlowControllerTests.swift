import XCTest
import SwiftUI
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence
@testable import SeeCalApp

// MARK: - Gated mock inference runner

/// A `ScanInferenceRunning` whose completion the test controls: `infer` parks on
/// a continuation until the test calls `succeed`/`fail`, so tests can assert
/// mid-flight state (analyzing, navigate-away) deterministically.
private actor GatedScanRunner: ScanInferenceRunning {
    private(set) var requests: [FoodScanRequest] = []
    private var continuations: [CheckedContinuation<FoodScanResult, Error>] = []

    func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        requests.append(request)
        return try await withCheckedThrowingContinuation { continuation in
            continuations.append(continuation)
        }
    }

    var pendingCount: Int { continuations.count }
    var requestCount: Int { requests.count }
    var lastRequest: FoodScanRequest? { requests.last }

    func succeed(with result: FoodScanResult) {
        let pending = continuations
        continuations.removeAll()
        for continuation in pending {
            continuation.resume(returning: result)
        }
    }

    func fail(with error: Error) {
        let pending = continuations
        continuations.removeAll()
        for continuation in pending {
            continuation.resume(throwing: error)
        }
    }
}

// MARK: - Gated capture service

/// A `CaptureService` whose `capturePhoto()` parks on a continuation until the
/// test releases it — so tests can close the camera while a capture is still
/// in flight and assert the aborted path.
@MainActor
private final class GatedCaptureService: CaptureService {
    let supportsDepthCapture = false
    private(set) var authorizationStatus: CameraAuthorization = .authorized
    private var continuations: [CheckedContinuation<CapturedPhoto, Error>] = []

    var pendingCaptureCount: Int { continuations.count }

    func requestAccess() async -> CameraAuthorization { .authorized }
    func startSession() {}
    func stopSession() {}

    func capturePhoto() async throws -> CapturedPhoto {
        try await withCheckedThrowingContinuation { continuations.append($0) }
    }

    func finishCapture(with result: Result<CapturedPhoto, Error>) {
        let pending = continuations
        continuations.removeAll()
        for continuation in pending {
            continuation.resume(with: result)
        }
    }

    func gravityUpdates() -> AsyncStream<GravityReading> {
        AsyncStream { $0.finish() }
    }

    func makePreviewView() -> AnyView { AnyView(EmptyView()) }
}

// MARK: - Fixtures

private func sampleScanResult() -> FoodScanResult {
    FoodScanResult(
        totalCalories: 464,
        proteinGrams: 44.2,
        fatGrams: 5.1,
        carbsGrams: 57.2,
        items: [
            ScanItem(name: "white rice", estimatedGrams: 180, calories: 234, proteinGrams: 4.3, fatGrams: 0.4, carbsGrams: 50.9),
            ScanItem(name: "chicken breast", estimatedGrams: 120, calories: 198, proteinGrams: 37.2, fatGrams: 4.3, carbsGrams: 0.0),
            ScanItem(name: "broccoli", estimatedGrams: 95, calories: 32, proteinGrams: 2.7, fatGrams: 0.4, carbsGrams: 6.3)
        ]
    )
}

final class ScanFlowControllerTests: XCTestCase {
    private var runner: GatedScanRunner!
    private var store: InMemoryMealLogStore!
    private var viewModel: AppViewModel!
    private var capture: MockCaptureService!
    private var controller: ScanFlowController!

    @MainActor
    private func makeSUT(now: @escaping @Sendable () -> Date = Date.init) {
        runner = GatedScanRunner()
        store = InMemoryMealLogStore()
        viewModel = AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: []), store: store)
        capture = MockCaptureService()
        let photoDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("seecal-scanflow-tests-\(UUID().uuidString)", isDirectory: true)
        controller = ScanFlowController(
            viewModel: viewModel,
            inference: runner,
            captureService: capture,
            photoStore: CapturedPhotoStore(directory: photoDirectory),
            now: now
        )
    }

    /// Polls until `condition` holds (the controller starts its inference task
    /// asynchronously; this bridges the startup race without sleeps of fixed
    /// length).
    @MainActor
    private func waitUntil(
        timeout: TimeInterval = 2,
        _ condition: @escaping @Sendable () async -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await condition() { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return await condition()
    }

    /// Drives the flow from FAB tap through shutter until inference is in
    /// flight, returning the captured photo path.
    @MainActor
    private func startAnalyzingScan() async throws -> String {
        controller.scanTapped()
        XCTAssertEqual(controller.phase, .capturing)
        XCTAssertTrue(controller.isScanSurfaceVisible)

        await controller.shutterTapped()
        guard case let .analyzing(photoPath) = controller.phase else {
            XCTFail("Expected .analyzing after shutter, got \(controller.phase)")
            throw XCTSkip("unreachable")
        }
        let arrived = await waitUntil { [runner] in await runner!.pendingCount == 1 }
        XCTAssertTrue(arrived, "Inference request never reached the runner")
        return photoPath
    }

    @MainActor
    private func finishInference() async {
        await runner.succeed(with: sampleScanResult())
        await controller.inferenceTask?.value
    }

    // MARK: - Capture → analyze

    @MainActor
    func testShutterCapturesPhotoAndStartsAnalysis() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()

        XCTAssertEqual(capture.captureCount, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: photoPath), "Captured photo should be written to disk")
        let request = await runner.lastRequest
        XCTAssertEqual(request?.imagePath, photoPath, "Inference must run on the captured photo file")
    }

    @MainActor
    func testAutoPresentsResultSheetWhenAnalyzingIsFrontmost() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()

        // User stays on the analyzing screen.
        await finishInference()

        guard case let .presentingResult(draft) = controller.phase else {
            return XCTFail("Expected auto-presented result sheet, got \(controller.phase)")
        }
        XCTAssertEqual(draft.imagePath, photoPath)
        XCTAssertFalse(controller.isBannerVisible)
        XCTAssertNotNil(controller.activeSheetDraft)
    }

    // MARK: - Background continuation (the load-bearing behavior)

    @MainActor
    func testNavigateAwayThenCompletionShowsBannerInsteadOfSheet() async throws {
        makeSUT()
        _ = try await startAnalyzingScan()

        // User taps a tab mid-analysis: surface goes away, inference does NOT.
        controller.leaveScanSurface()
        XCTAssertFalse(controller.isScanSurfaceVisible)
        if case .analyzing = controller.phase {} else {
            XCTFail("Leaving the analyzing screen must not cancel inference; got \(controller.phase)")
        }
        let stillPending = await runner.pendingCount
        XCTAssertEqual(stillPending, 1, "Inference must keep running after navigating away")

        await finishInference()

        guard case .ready = controller.phase else {
            return XCTFail("Expected pending banner result, got \(controller.phase)")
        }
        XCTAssertTrue(controller.isBannerVisible)
        XCTAssertNil(controller.activeSheetDraft, "Sheet must not auto-present when the user navigated away")

        // Banner tap opens the sheet from anywhere.
        controller.bannerTapped()
        XCTAssertFalse(controller.isBannerVisible)
        XCTAssertNotNil(controller.activeSheetDraft)
    }

    @MainActor
    func testNothingIsPersistedWithoutExplicitLog() async throws {
        makeSUT()
        _ = try await startAnalyzingScan()
        await finishInference()

        XCTAssertNotNil(controller.activeSheetDraft)
        let entries = try await store.fetchAll()
        XCTAssertTrue(entries.isEmpty, "Store must stay untouched until the user taps Log meal")
    }

    @MainActor
    func testLogMealPersistsSwitchesToTodayAndToasts() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()
        await finishInference()

        var switchedTo: AppTab?
        controller.onRequestTabSwitch = { switchedTo = $0 }
        let tokenBefore = controller.todayScrollToTopToken

        guard case let .presentingResult(draft) = controller.phase else {
            return XCTFail("Expected result sheet, got \(controller.phase)")
        }
        await controller.logMeal(draft)

        let entries = try await store.fetchAll()
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.imagePath, photoPath)
        XCTAssertEqual(entries.first?.name, "White Rice & Chicken Breast")
        XCTAssertEqual(controller.phase, .idle)
        XCTAssertFalse(controller.isScanSurfaceVisible)
        XCTAssertEqual(controller.toastMessage, "Logged to Today")
        XCTAssertEqual(switchedTo, .today)
        XCTAssertNotEqual(controller.todayScrollToTopToken, tokenBefore, "Today must be told to scroll to top")
        XCTAssertEqual(viewModel.mealEntries.count, 1, "View model must refresh after logging")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: photoPath),
            "A LOGGED meal keeps its photo file — thumbnails need it"
        )
    }

    @MainActor
    func testRapidDoubleLogMealPersistsExactlyOnce() async throws {
        makeSUT()
        _ = try await startAnalyzingScan()
        await finishInference()

        guard case let .presentingResult(draft) = controller.phase else {
            return XCTFail("Expected result sheet, got \(controller.phase)")
        }

        // Two rapid taps on "Log meal": the phase flips off .presentingResult
        // before the first suspension, so the second call must no-op. (Both
        // tasks inherit the main actor, like two button-tap handlers.)
        let controller = self.controller!
        let first = Task { await controller.logMeal(draft) }
        let second = Task { await controller.logMeal(draft) }
        await first.value
        await second.value

        let entries = try await store.fetchAll()
        XCTAssertEqual(entries.count, 1, "A double-tapped Log meal must persist exactly one entry")
        XCTAssertEqual(viewModel.mealEntries.count, 1)
    }

    @MainActor
    func testStartingNewScanClearsPendingBannerResult() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()
        controller.leaveScanSurface()
        await finishInference()
        XCTAssertTrue(controller.isBannerVisible)

        // FAB tap = new scan: pending result is dropped, camera opens.
        controller.scanTapped()

        XCTAssertEqual(controller.phase, .capturing)
        XCTAssertTrue(controller.isScanSurfaceVisible)
        XCTAssertFalse(controller.isBannerVisible)
        XCTAssertNil(controller.activeSheetDraft)
        let entries = try await store.fetchAll()
        XCTAssertTrue(entries.isEmpty, "Discarded pending result must never be persisted")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: photoPath),
            "The dropped pending result's photo file must be cleaned up"
        )
    }

    // MARK: - Reentrancy

    @MainActor
    func testScanFABDuringInFlightAnalysisReturnsToAnalyzingWithoutSecondRun() async throws {
        makeSUT()
        _ = try await startAnalyzingScan()
        controller.leaveScanSurface()

        // FAB tap while inference is in flight: back to the Analyzing screen,
        // no parallel run, no recapture.
        controller.scanTapped()

        XCTAssertTrue(controller.isScanSurfaceVisible)
        if case .analyzing = controller.phase {} else {
            XCTFail("Expected to return to the in-flight analyzing state, got \(controller.phase)")
        }
        XCTAssertEqual(capture.captureCount, 1)
        let requestCount = await runner.requestCount
        XCTAssertEqual(requestCount, 1, "Only one inference may run at a time")
    }

    // MARK: - Error + retry

    @MainActor
    func testInferenceErrorSurfacesMessageAndRetryReusesSamePhoto() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()

        await runner.fail(with: InferenceError.runtimeFailed("model exploded"))
        await controller.inferenceTask?.value

        guard case let .error(message, errorPhotoPath) = controller.phase else {
            return XCTFail("Expected error phase, got \(controller.phase)")
        }
        XCTAssertTrue(message.contains("model exploded"), "The runtime's surfaced error must reach the UI, got: \(message)")
        XCTAssertEqual(errorPhotoPath, photoPath)

        // Retry re-runs inference on the SAME photo — no recapture.
        controller.retry()
        if case .analyzing(let retryPath) = controller.phase {
            XCTAssertEqual(retryPath, photoPath)
        } else {
            XCTFail("Expected re-analysis after retry, got \(controller.phase)")
        }
        let arrived = await waitUntil { [runner] in await runner!.requestCount == 2 }
        XCTAssertTrue(arrived)
        XCTAssertEqual(capture.captureCount, 1, "Retry must not recapture")
        let retryRequest = await runner.lastRequest
        XCTAssertEqual(retryRequest?.imagePath, photoPath)

        await finishInference()
        if case .presentingResult = controller.phase {} else {
            XCTFail("Retry success should present the result, got \(controller.phase)")
        }
    }

    @MainActor
    func testScanFABAfterErrorStartsFreshCaptureInsteadOfDeadEnding() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()

        await runner.fail(with: InferenceError.runtimeFailed("model exploded"))
        await controller.inferenceTask?.value
        guard case .error = controller.phase else {
            return XCTFail("Expected error phase, got \(controller.phase)")
        }

        // FAB tap after a failed inference: NOT the error screen again — a
        // fresh capture, with the failed photo cleaned up.
        controller.scanTapped()

        XCTAssertEqual(controller.phase, .capturing, "The error phase must not be a dead end")
        XCTAssertTrue(controller.isScanSurfaceVisible)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: photoPath),
            "The abandoned failed photo must be deleted"
        )

        // The camera is genuinely reachable: the shutter starts a new analysis.
        await controller.shutterTapped()
        guard case let .analyzing(newPath) = controller.phase else {
            return XCTFail("Expected a new analysis after recapture, got \(controller.phase)")
        }
        XCTAssertNotEqual(newPath, photoPath, "New scan must capture a NEW photo")
        XCTAssertEqual(capture.captureCount, 2)
    }

    @MainActor
    func testNewScanAfterErrorAffordanceReturnsToCamera() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()
        await runner.fail(with: InferenceError.runtimeFailed("boom"))
        await controller.inferenceTask?.value

        // The Analyzing screen's error-block secondary button.
        controller.newScanAfterError()

        XCTAssertEqual(controller.phase, .capturing)
        XCTAssertTrue(controller.isScanSurfaceVisible)
        XCTAssertFalse(FileManager.default.fileExists(atPath: photoPath))
    }

    // MARK: - Close-during-capture (aborted capture must be inert)

    @MainActor
    private func makeGatedCaptureSUT() -> GatedCaptureService {
        runner = GatedScanRunner()
        store = InMemoryMealLogStore()
        viewModel = AppViewModel(orchestrator: RuntimeOrchestrator(runtimes: []), store: store)
        let gated = GatedCaptureService()
        let photoDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("seecal-scanflow-tests-\(UUID().uuidString)", isDirectory: true)
        controller = ScanFlowController(
            viewModel: viewModel,
            inference: runner,
            captureService: gated,
            photoStore: CapturedPhotoStore(directory: photoDirectory)
        )
        return gated
    }

    @MainActor
    func testClosingCameraDuringInFlightCaptureStartsNoAnalysis() async {
        let gated = makeGatedCaptureSUT()
        controller.scanTapped()

        let shutterTask = Task { await controller.shutterTapped() }
        let captureStarted = await waitUntil { @MainActor [gated] in gated.pendingCaptureCount == 1 }
        XCTAssertTrue(captureStarted)

        // User closes the camera while the capture is still in flight…
        controller.closeCamera()
        XCTAssertEqual(controller.phase, .idle)

        // …then the capture lands anyway. Nothing may happen with it.
        gated.finishCapture(with: .success(CapturedPhoto(imageData: MockCaptureService.samplePhotoData())))
        await shutterTask.value

        XCTAssertEqual(controller.phase, .idle, "An abandoned capture must not start an analysis")
        let requests = await runner.requestCount
        XCTAssertEqual(requests, 0, "No inference may run for an abandoned capture")
        XCTAssertNil(controller.captureError)
    }

    @MainActor
    func testCaptureFailureAfterCameraClosedSetsNoStaleError() async {
        let gated = makeGatedCaptureSUT()
        controller.scanTapped()

        let shutterTask = Task { await controller.shutterTapped() }
        let captureStarted = await waitUntil { @MainActor [gated] in gated.pendingCaptureCount == 1 }
        XCTAssertTrue(captureStarted)

        controller.closeCamera()
        gated.finishCapture(with: .failure(CaptureServiceError.captureFailed("session interrupted")))
        await shutterTask.value

        XCTAssertNil(
            controller.captureError,
            "A capture aborted BY closing the camera must not park an alert for the next open"
        )
        XCTAssertEqual(controller.phase, .idle)
    }

    // MARK: - Discard / dismissal semantics

    @MainActor
    func testDiscardNewScanReturnsToCameraWithoutPersistingAndDeletesPhoto() async throws {
        makeSUT()
        let photoPath = try await startAnalyzingScan()
        await finishInference()
        XCTAssertTrue(FileManager.default.fileExists(atPath: photoPath))

        controller.discardResult()

        XCTAssertEqual(controller.phase, .capturing)
        XCTAssertTrue(controller.isScanSurfaceVisible)
        let entries = try await store.fetchAll()
        XCTAssertTrue(entries.isEmpty)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: photoPath),
            "Discard must clean up the captured photo file"
        )
    }

    @MainActor
    func testInteractiveSheetDismissParksResultBehindBanner() async throws {
        makeSUT()
        _ = try await startAnalyzingScan()
        await finishInference()
        guard case .presentingResult = controller.phase else {
            return XCTFail("Expected result sheet, got \(controller.phase)")
        }

        controller.resultSheetDismissed()

        XCTAssertTrue(controller.isBannerVisible, "Swiped-away result must stay reachable via the banner")
        controller.bannerTapped()
        XCTAssertNotNil(controller.activeSheetDraft)
    }

    @MainActor
    func testSheetGramAdjustmentsSurviveInteractiveDismiss() async throws {
        makeSUT()
        _ = try await startAnalyzingScan()
        await finishInference()
        guard case let .presentingResult(draft) = controller.phase else {
            return XCTFail("Expected result sheet, got \(controller.phase)")
        }

        // The sheet edits a local copy and reports it via onDraftChanged →
        // presentedDraftChanged. Simulate one +5 g step (180 → 185).
        var edited = draft
        edited.stepGrams(itemID: edited.items[0].id, by: MealItem.gramStep)
        controller.presentedDraftChanged(edited)

        // Swipe-dismiss parks the result; the banner reopens it.
        controller.resultSheetDismissed()
        XCTAssertTrue(controller.isBannerVisible)
        controller.bannerTapped()

        guard let reopened = controller.activeSheetDraft else {
            return XCTFail("Banner tap should reopen the parked draft")
        }
        XCTAssertEqual(
            reopened.items[0].grams, 185,
            "The parked draft must carry the sheet's adjustments, not the pre-adjustment values"
        )
    }

    @MainActor
    func testCloseCameraReturnsToIdleAndLandsOnToday() {
        makeSUT()
        var switchedTo: AppTab?
        controller.onRequestTabSwitch = { switchedTo = $0 }

        controller.scanTapped()
        controller.closeCamera()

        XCTAssertEqual(controller.phase, .idle)
        XCTAssertFalse(controller.isScanSurfaceVisible)
        XCTAssertEqual(switchedTo, .today, "Camera ✕ returns to Today (spec §5), not the previously selected tab")
    }

    // MARK: - Edit mode (same sheet, Save changes semantics)

    @MainActor
    func testEditSaveChangesUpdatesEntryAndToasts() async throws {
        makeSUT()
        let entry = MealLogEntry(
            mealType: .lunch,
            imagePath: "/tmp/photo.jpg",
            items: [
                MealItem(
                    name: "salmon fillet",
                    grams: 150,
                    base: MealItemBase(grams: 150, kcal: 312, protein: 37.5, fat: 20.0, carbs: 0.0)
                )
            ]
        )
        try await store.save(entry)
        await viewModel.loadEntries()

        controller.beginEdit(entry: entry)
        guard var draft = controller.activeSheetDraft else {
            return XCTFail("Edit draft should present")
        }
        XCTAssertTrue(draft.isEditingExisting)

        draft.stepGrams(itemID: draft.items[0].id, by: MealItem.gramStep)
        await controller.saveChanges(draft)

        let entries = try await store.fetchAll()
        XCTAssertEqual(entries.count, 1, "Save changes must update, not append")
        XCTAssertEqual(entries.first?.id, entry.id)
        XCTAssertEqual(entries.first?.items.first?.grams, 155)
        XCTAssertEqual(controller.toastMessage, "Changes saved")
        XCTAssertNil(controller.activeSheetDraft)
    }

    @MainActor
    func testCancelEditClosesWithoutPersisting() async throws {
        makeSUT()
        let entry = MealLogEntry(
            mealType: .dinner,
            imagePath: "/tmp/photo.jpg",
            items: [
                MealItem(
                    name: "pasta",
                    grams: 220,
                    base: MealItemBase(grams: 220, kcal: 346, protein: 12.8, fat: 2.0, carbs: 68.0)
                )
            ]
        )
        try await store.save(entry)

        controller.beginEdit(entry: entry)
        controller.cancelEdit()

        XCTAssertNil(controller.activeSheetDraft)
        let entries = try await store.fetchAll()
        XCTAssertEqual(entries.first?.items.first?.grams, 220, "Cancel must not write anything")
        XCTAssertNil(controller.toastMessage)
    }

    // MARK: - Derivations

    func testMealNameDerivedFromTopTwoItemsByGrams() {
        XCTAssertEqual(
            ScanFlowController.mealName(from: sampleScanResult()),
            "White Rice & Chicken Breast"
        )

        let single = FoodScanResult(
            totalCalories: 100, proteinGrams: 1, fatGrams: 1, carbsGrams: 1,
            items: [ScanItem(name: "apple", estimatedGrams: 150, calories: 78, proteinGrams: 0.4, fatGrams: 0.3, carbsGrams: 21)]
        )
        XCTAssertEqual(ScanFlowController.mealName(from: single), "Apple")

        let empty = FoodScanResult(totalCalories: 0, proteinGrams: 0, fatGrams: 0, carbsGrams: 0, items: [])
        XCTAssertEqual(ScanFlowController.mealName(from: empty), "Meal")
    }

    func testMealTypeInferredFromCaptureHour() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!

        func date(hour: Int) -> Date {
            calendar.date(from: DateComponents(year: 2026, month: 7, day: 26, hour: hour))!
        }

        XCTAssertEqual(ScanFlowController.mealType(for: date(hour: 8), calendar: calendar), .breakfast)
        XCTAssertEqual(ScanFlowController.mealType(for: date(hour: 12), calendar: calendar), .lunch)
        XCTAssertEqual(ScanFlowController.mealType(for: date(hour: 19), calendar: calendar), .dinner)
        XCTAssertEqual(ScanFlowController.mealType(for: date(hour: 23), calendar: calendar), .snack)
        XCTAssertEqual(ScanFlowController.mealType(for: date(hour: 2), calendar: calendar), .snack)
    }

    func testFoodsFoundSubtitleListsFirstThreeNames() {
        let items = sampleScanResult().items.map(MealItem.init(scanItem:))
        XCTAssertEqual(
            ScanFlowController.foodsFoundSubtitle(items: items),
            "3 foods found: white rice, chicken breast, broccoli"
        )
        XCTAssertEqual(
            ScanFlowController.foodsFoundSubtitle(items: [items[0]]),
            "1 food found: white rice"
        )
    }
}
