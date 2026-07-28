import Foundation
import SeeCalDomain
import SeeCalInference
import SeeCalPersistence

// MARK: - Inference seam

/// The one inference capability the scan flow needs. `RuntimeOrchestrator`
/// (the production path — base model + LoRA via mlx-swift) already has exactly
/// this shape; the protocol exists so tests can substitute a gated mock without
/// touching production behavior.
public protocol ScanInferenceRunning: Sendable {
    func infer(request: FoodScanRequest) async throws -> FoodScanResult
}

extension RuntimeOrchestrator: ScanInferenceRunning {}

// MARK: - Captured-photo store

/// Writes captured photos to disk so `FoodScanRequest.imagePath` (and later the
/// persisted `MealLogEntry.imagePath`) points at a stable file. Images are
/// downsampled to inference size (768px JPEG via `InferenceImagePreprocessor`)
/// on the way in — the stored file is exactly what the model reads.
public struct CapturedPhotoStore: Sendable {
    public let directory: URL

    public init(directory: URL? = nil) {
        if let directory {
            self.directory = directory
        } else {
            let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? FileManager.default.temporaryDirectory
            self.directory = base.appendingPathComponent("SeeCal/photos", isDirectory: true)
        }
    }

    public func save(_ imageData: Data) throws -> String {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("meal-\(UUID().uuidString).jpg")
        // Fall back to the raw bytes if downsampling fails (e.g. an exotic format
        // CGImageSource can't read) — better an oversized photo than a lost one.
        let jpeg = (try? InferenceImagePreprocessor.downsampledJPEG(from: imageData)) ?? imageData
        try jpeg.write(to: url, options: .atomic)
        return url.path
    }

    /// Removes an abandoned capture from disk (discard, new-scan-over-pending,
    /// error abandonment). Best-effort: a leaked file is a nuisance, not a
    /// failure worth surfacing.
    public func delete(atPath path: String) {
        try? FileManager.default.removeItem(atPath: path)
    }
}

// MARK: - Scan flow controller

/// The scan → analyzing → result state machine (spec §5), owned by `RootView`
/// and driven by the camera/analyzing screens, the tab bar, the ready banner,
/// and the result sheet.
///
/// Design invariants (each covered by `ScanFlowControllerTests`):
/// - **Navigation away does NOT cancel inference.** Leaving the scan surface
///   while `.analyzing` keeps the task running; completion lands as `.ready`
///   (persistent banner) instead of auto-presenting the sheet.
/// - **Nothing persists without explicit confirm.** Inference completion only
///   builds a `MealEditDraft`; the store is untouched until `logMeal`.
/// - **One inference at a time.** The Scan FAB during an in-flight analysis
///   returns to the Analyzing screen (`isScanSurfaceVisible = true`) rather
///   than starting a parallel run; `startAnalysis` refuses to double-start.
/// - **A new scan clears any pending banner/result** (`.ready` → `.capturing`).
/// - **Retry re-runs inference on the SAME photo** — no recapture.
@MainActor
public final class ScanFlowController: ObservableObject {
    public enum Phase: Equatable {
        /// No scan activity; tabs behave normally.
        case idle
        /// Camera (viewfinder) is the scan surface's content.
        case capturing
        /// Inference in flight on the captured photo. Survives leaving the
        /// scan surface.
        case analyzing(photoPath: String)
        /// Inference finished while the scan surface was NOT frontmost: the
        /// result waits behind the "Meal analyzed — tap to review" banner.
        case ready(draft: MealEditDraft)
        /// The result sheet is up for a fresh scan (new-scan mode).
        case presentingResult(draft: MealEditDraft)
        /// Inference failed; the analyzing screen shows the runtime's message
        /// with a Retry affordance bound to the same photo.
        case error(message: String, photoPath: String)
        /// The model refused: the photo isn't food (v7 `{"not_food": true}`).
        /// A correct terminal answer — the analyzing screen shows a friendly
        /// "No food detected" state (New scan / Try again on the same photo),
        /// NOT the red error screen.
        case notFood(photoPath: String)
    }

    @Published public private(set) var phase: Phase = .idle
    /// Whether the scan surface (camera or analyzing screen) is what the root
    /// shows instead of the selected tab. Tab taps flip this off WITHOUT
    /// touching an in-flight `.analyzing` phase.
    @Published public private(set) var isScanSurfaceVisible = false
    /// Edit-an-existing-meal draft (tapping a meal row) — orthogonal to `phase`
    /// so editing never disturbs a background analysis.
    @Published public private(set) var editDraft: MealEditDraft?
    /// Transient toast ("Logged to Today" / "Changes saved"); auto-clears after
    /// ~1.9 s, matching the prototype's `showToast`.
    @Published public private(set) var toastMessage: String?
    /// Camera-side failure (capture or file write) surfaced as an alert on the
    /// camera screen; the user stays in the viewfinder.
    @Published public var captureError: String?
    /// Changes whenever "Log meal" lands, telling the Today screen to scroll to
    /// top (spec §5: "switch Today scrolled to top").
    @Published public private(set) var todayScrollToTopToken = UUID()

    /// Set by `RootView` so the controller can land the user on Today after
    /// logging. A closure (not a binding) keeps the controller testable.
    public var onRequestTabSwitch: ((AppTab) -> Void)?

    public let captureService: CaptureService

    /// Settings §8 Capture toggles, read live off the owning `AppViewModel` so
    /// `CameraScreen` can gate its coaching overlays (level indicator, hint
    /// text, depth affordances) without holding the whole view model itself.
    public var capturePreferences: CapturePreferences {
        viewModel.capturePreferences
    }

    private let viewModel: AppViewModel
    private let inference: ScanInferenceRunning
    private let photoStore: CapturedPhotoStore
    private let now: @Sendable () -> Date
    /// Exposed for tests to await inference completion deterministically.
    private(set) var inferenceTask: Task<Void, Never>?
    private var toastTask: Task<Void, Never>?

    public init(
        viewModel: AppViewModel,
        inference: ScanInferenceRunning,
        captureService: CaptureService,
        photoStore: CapturedPhotoStore = CapturedPhotoStore(),
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.viewModel = viewModel
        self.inference = inference
        self.captureService = captureService
        self.photoStore = photoStore
        self.now = now
    }

    // MARK: Derived state

    /// The persistent basil banner above the tab bar ("Meal analyzed — tap to
    /// review"). Visible exactly while a completed result awaits review.
    public var isBannerVisible: Bool {
        if case .ready = phase { return true }
        return false
    }

    /// The tab bar hides only for the camera (prototype: `.tabbar.hidden` when
    /// screen == "camera"); the analyzing screen keeps it so the user can
    /// navigate away without cancelling.
    public var isTabBarHidden: Bool {
        if case .capturing = phase, isScanSurfaceVisible { return true }
        return false
    }

    /// The single sheet source for `RootView`: an edit draft takes priority
    /// (it can only exist while no result sheet is up), else a fresh result.
    public var activeSheetDraft: MealEditDraft? {
        if let editDraft { return editDraft }
        if case let .presentingResult(draft) = phase { return draft }
        return nil
    }

    // MARK: Scan surface entry/exit

    /// Scan FAB tap. During an in-flight analysis this returns to the
    /// Analyzing screen rather than starting a parallel run; after an error it
    /// abandons the failed photo and opens the camera for a fresh capture;
    /// with a pending (banner) result it clears it and starts a fresh capture.
    public func scanTapped() {
        switch phase {
        case .idle:
            phase = .capturing
            isScanSurfaceVisible = true
        case .capturing, .analyzing:
            isScanSurfaceVisible = true
        case .error, .notFood:
            // A failed inference (or a not-food refusal) must not dead-end the
            // flow: the FAB starts over instead of re-showing the same screen.
            newScanAfterError()
        case let .ready(draft):
            // "Starting a new scan clears any pending banner/result" (spec §5).
            if let imagePath = draft.imagePath {
                photoStore.delete(atPath: imagePath)
            }
            phase = .capturing
            isScanSurfaceVisible = true
        case .presentingResult:
            break // sheet is up; the FAB isn't reachable behind it
        }
    }

    /// "New scan" from a terminal analyzing state (`.error` or `.notFood`), and
    /// the FAB while in one: abandon that photo — deleting its file — and return
    /// to the camera for a fresh capture.
    public func newScanAfterError() {
        let photoPath: String
        switch phase {
        case let .error(_, path), let .notFood(path):
            photoPath = path
        default:
            return
        }
        photoStore.delete(atPath: photoPath)
        phase = .capturing
        isScanSurfaceVisible = true
    }

    /// Camera ✕ — abandons the capture and lands on Today (spec §5: the
    /// viewfinder's ✕ "returns to Today", not to whatever tab was selected).
    public func closeCamera() {
        guard case .capturing = phase else { return }
        phase = .idle
        isScanSurfaceVisible = false
        onRequestTabSwitch?(.today)
    }

    /// Any tab tap while the scan surface is up. Leaving the camera abandons
    /// it; leaving the analyzing screen does NOT cancel inference — completion
    /// surfaces as the ready banner instead.
    public func leaveScanSurface() {
        guard isScanSurfaceVisible else { return }
        isScanSurfaceVisible = false
        if case .capturing = phase {
            phase = .idle
        }
    }

    // MARK: Capture → analyze

    public func shutterTapped() async {
        guard case .capturing = phase else { return }
        do {
            let photo = try await captureService.capturePhoto()
            // The camera may have been closed (✕ / tab tap) while the capture
            // was in flight; an abandoned viewfinder must not start an analysis.
            guard case .capturing = phase else { return }
            let path = try photoStore.save(photo.imageData)
            startAnalysis(photoPath: path)
        } catch {
            // Same re-check on the failure path: a capture aborted BECAUSE the
            // camera closed must not park a stale alert for the next open.
            guard case .capturing = phase else { return }
            captureError = error.localizedDescription
        }
    }

    /// Retry after an inference error, or "Try again" after a not-food refusal:
    /// re-runs on the SAME photo, no recapture.
    public func retry() {
        let photoPath: String
        switch phase {
        case let .error(_, path), let .notFood(path):
            photoPath = path
        default:
            return
        }
        startAnalysis(photoPath: photoPath)
    }

    private func startAnalysis(photoPath: String) {
        guard inferenceTask == nil else { return } // one inference at a time
        phase = .analyzing(photoPath: photoPath)
        let mealType = Self.mealType(for: now())
        let request = FoodScanRequest(imagePath: photoPath, mealType: mealType, userHint: nil)
        inferenceTask = Task { [weak self] in
            // Runs on the main actor (inherited); the await inside suspends
            // without blocking UI.
            guard let self else { return }
            do {
                let result = try await self.inference.infer(request: request)
                self.inferenceTask = nil
                self.finishAnalysis(result: result, photoPath: photoPath, mealType: mealType)
            } catch InferenceError.notFood {
                // Definitive refusal — a correct answer, not a failure. Distinct
                // terminal state so the UI reads "No food detected", never the
                // red error/Retry screen.
                self.inferenceTask = nil
                self.phase = .notFood(photoPath: photoPath)
            } catch {
                self.inferenceTask = nil
                self.phase = .error(message: error.localizedDescription, photoPath: photoPath)
            }
        }
    }

    private func finishAnalysis(result: FoodScanResult, photoPath: String, mealType: MealType) {
        let draft = MealEditDraft(
            scanResult: result,
            imagePath: photoPath,
            mealType: mealType,
            name: Self.mealName(from: result)
        )
        if isScanSurfaceVisible {
            // Analyzing screen is frontmost → auto-present the result sheet.
            phase = .presentingResult(draft: draft)
        } else {
            // User navigated away mid-analysis → persistent review banner.
            phase = .ready(draft: draft)
        }
    }

    // MARK: Banner / sheet lifecycle

    public func bannerTapped() {
        guard case let .ready(draft) = phase else { return }
        phase = .presentingResult(draft: draft)
    }

    /// Interactive sheet dismissal (swipe-down / scrim) for a NEW scan: the
    /// result isn't lost — it parks behind the ready banner so the user can
    /// come back to it. (Explicit Discard is `discardResult()`.)
    public func resultSheetDismissed() {
        guard case let .presentingResult(draft) = phase else { return }
        phase = .ready(draft: draft)
    }

    /// Live draft edits from the result sheet (gram steppers). Keeping the
    /// presented draft in sync means an interactive dismissal parks the
    /// ADJUSTED draft behind the banner, not the pre-adjustment one.
    public func presentedDraftChanged(_ draft: MealEditDraft) {
        guard case .presentingResult = phase, !draft.isEditingExisting else { return }
        phase = .presentingResult(draft: draft)
    }

    /// New-scan Discard: drop the result — deleting its photo file — and
    /// return to the camera (prototype: `discardBtn` → `show("camera")` when
    /// the draft isn't an edit).
    public func discardResult() {
        guard case let .presentingResult(draft) = phase else { return }
        if let imagePath = draft.imagePath {
            photoStore.delete(atPath: imagePath)
        }
        phase = .capturing
        isScanSurfaceVisible = true
    }

    /// New-scan "Log meal": persist (the FIRST store write of the whole flow),
    /// toast, land on Today scrolled to top. Guarded against double-taps: the
    /// phase flips off `.presentingResult` BEFORE the first suspension, so a
    /// second rapid call no-ops instead of persisting twice.
    public func logMeal(_ draft: MealEditDraft) async {
        guard case .presentingResult = phase else { return }
        do {
            let entry = try draft.committedEntry()
            phase = .idle
            isScanSurfaceVisible = false
            await viewModel.logMeal(entry)
            todayScrollToTopToken = UUID()
            onRequestTabSwitch?(.today)
            showToast("Logged to Today")
        } catch {
            viewModel.lastError = error.localizedDescription
        }
    }

    // MARK: Edit mode (same sheet component, spec §5)

    public func beginEdit(entry: MealLogEntry) {
        guard activeSheetDraft == nil else { return }
        editDraft = MealEditDraft(entry: entry)
    }

    /// Edit-mode Cancel (or interactive dismissal): just close — no store write.
    public func cancelEdit() {
        editDraft = nil
    }

    /// Edit-mode "Save changes": update the entry in place + toast.
    public func saveChanges(_ draft: MealEditDraft) async {
        do {
            let entry = try draft.committedEntry()
            await viewModel.updateMeal(entry)
            editDraft = nil
            showToast("Changes saved")
        } catch {
            viewModel.lastError = error.localizedDescription
        }
    }

    /// Edit-mode deletion reuses the view model's store + photo cleanup path.
    /// Keep the sheet open when persistence fails so the user can retry.
    public func deleteMeal(_ draft: MealEditDraft) async {
        guard let id = draft.existingEntryID else { return }
        await viewModel.deleteMeal(id: id)
        guard !viewModel.mealEntries.contains(where: { $0.id == id }) else { return }

        editDraft = nil
        todayScrollToTopToken = UUID()
        onRequestTabSwitch?(.today)
        showToast("Meal deleted")
    }

    // MARK: Toast

    private func showToast(_ message: String) {
        toastTask?.cancel()
        toastMessage = message
        toastTask = Task { [weak self] in
            // `.toast` auto-dismiss: prototype's `setTimeout(..., 1900)`.
            try? await Task.sleep(nanoseconds: 1_900_000_000)
            guard !Task.isCancelled else { return }
            self?.toastMessage = nil
        }
    }

    // MARK: Derivations

    /// Meal name for a new scan: the top two items by estimated grams,
    /// word-capitalized and joined with " & " (e.g. "White Rice & Chicken
    /// Breast"). The prototype hardcodes demo names; this rule keeps the name
    /// meaningful for any model output. Falls back to "Meal" if the result
    /// somehow has no usable item names.
    nonisolated public static func mealName(from result: FoodScanResult) -> String {
        let topNames = result.items
            .sorted { $0.estimatedGrams > $1.estimatedGrams }
            .map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines).capitalized }
            .filter { !$0.isEmpty }
            .prefix(2)
        guard !topNames.isEmpty else { return "Meal" }
        return topNames.joined(separator: " & ")
    }

    /// Meal type inferred from the capture time (the prototype has no meal-type
    /// picker anywhere in the scan flow): 05–10 breakfast, 11–15 lunch,
    /// 16–21 dinner, otherwise snack.
    nonisolated public static func mealType(for date: Date, calendar: Calendar = .current) -> MealType {
        switch calendar.component(.hour, from: date) {
        case 5..<11: return .breakfast
        case 11..<16: return .lunch
        case 16..<22: return .dinner
        default: return .snack
        }
    }

    /// The Analyzing screen's "Identifying foods" done-subtitle: "N foods
    /// found: a, b, c" (first three item names, matching the prototype's
    /// `"3 foods found: rice, chicken, broccoli"`).
    nonisolated public static func foodsFoundSubtitle(items: [MealItem]) -> String {
        let names = items.prefix(3).map(\.name).joined(separator: ", ")
        let noun = items.count == 1 ? "food" : "foods"
        return "\(items.count) \(noun) found: \(names)"
    }
}
