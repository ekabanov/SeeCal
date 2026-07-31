import Foundation
import SeeCalDiagnostics

public enum MealReviewAction: String, Equatable, Sendable {
    case draftShown = "draft_shown"
    case itemOpened = "item_opened"
    case replacementSelected = "replacement_selected"
    case amountChanged = "amount_changed"
    case detailsOpened = "details_opened"
    case ingredientAdded = "ingredient_added"
    case ingredientDeleted = "ingredient_deleted"
    case keyboardUsed = "keyboard_used"
    case saved
    case discarded
    case dismissed
}

public struct MealReviewEvent: Equatable, Sendable {
    public let sessionID: UUID
    public let action: MealReviewAction
    public let activeElapsedSeconds: TimeInterval

    public init(
        sessionID: UUID,
        action: MealReviewAction,
        activeElapsedSeconds: TimeInterval
    ) {
        self.sessionID = sessionID
        self.action = action
        self.activeElapsedSeconds = activeElapsedSeconds
    }
}

public struct MealReviewSummary: Equatable, Sendable {
    public let sessionID: UUID
    public let outcome: MealReviewAction
    public let activeElapsedSeconds: TimeInterval
    public let actionCount: Int
    public let correctionActionCount: Int
    public let usedKeyboard: Bool
}

/// Privacy-safe, local instrumentation for the review flow. It records only
/// interaction kinds and timing—never photos, food names, or nutrition values.
public final class MealReviewSessionRecorder {
    public typealias EventSink = (MealReviewEvent) -> Void
    public typealias SummarySink = (MealReviewSummary) -> Void

    private let sessionID: UUID
    private let now: () -> Date
    private let eventSink: EventSink
    private let summarySink: SummarySink
    private var activeStartedAt: Date?
    private var accumulatedActiveTime: TimeInterval = 0
    private var actionCount = 0
    private var correctionActionCount = 0
    private var usedKeyboard = false
    private var hasStarted = false
    private var hasFinished = false

    public init(
        sessionID: UUID,
        now: @escaping () -> Date,
        eventSink: @escaping EventSink,
        summarySink: @escaping SummarySink
    ) {
        self.sessionID = sessionID
        self.now = now
        self.eventSink = eventSink
        self.summarySink = summarySink
    }

    public convenience init() {
        self.init(
            sessionID: UUID(),
            now: Date.init,
            eventSink: Self.diagnosticEventSink,
            summarySink: Self.diagnosticSummarySink
        )
    }

    public func start() {
        guard !hasStarted, !hasFinished else {
            resume()
            return
        }
        hasStarted = true
        activeStartedAt = now()
        emit(.draftShown)
    }

    public func record(_ action: MealReviewAction) {
        guard hasStarted, !hasFinished else { return }
        switch action {
        case .draftShown, .saved, .discarded, .dismissed:
            break
        case .replacementSelected, .amountChanged, .ingredientAdded, .ingredientDeleted:
            actionCount += 1
            correctionActionCount += 1
        case .itemOpened, .detailsOpened:
            actionCount += 1
        case .keyboardUsed:
            guard !usedKeyboard else { return }
            usedKeyboard = true
        }
        emit(action)
    }

    public func pause() {
        guard let activeStartedAt, !hasFinished else { return }
        accumulatedActiveTime += max(0, now().timeIntervalSince(activeStartedAt))
        self.activeStartedAt = nil
    }

    public func resume() {
        guard hasStarted, !hasFinished, activeStartedAt == nil else { return }
        activeStartedAt = now()
    }

    @discardableResult
    public func finish(_ outcome: MealReviewAction) -> MealReviewSummary? {
        guard [.saved, .discarded, .dismissed].contains(outcome),
              hasStarted,
              !hasFinished
        else { return nil }
        pause()
        hasFinished = true
        let summary = MealReviewSummary(
            sessionID: sessionID,
            outcome: outcome,
            activeElapsedSeconds: accumulatedActiveTime,
            actionCount: actionCount,
            correctionActionCount: correctionActionCount,
            usedKeyboard: usedKeyboard
        )
        eventSink(
            MealReviewEvent(
                sessionID: sessionID,
                action: outcome,
                activeElapsedSeconds: accumulatedActiveTime
            )
        )
        summarySink(summary)
        return summary
    }

    private func emit(_ action: MealReviewAction) {
        eventSink(
            MealReviewEvent(
                sessionID: sessionID,
                action: action,
                activeElapsedSeconds: currentActiveTime
            )
        )
    }

    private var currentActiveTime: TimeInterval {
        accumulatedActiveTime
            + (activeStartedAt.map { max(0, now().timeIntervalSince($0)) } ?? 0)
    }

    private static func diagnosticEventSink(_ event: MealReviewEvent) {
        SeeCalDiagnostics.record(
            .info,
            category: "meal_review",
            name: event.action.rawValue,
            fields: [
                "review_session": event.sessionID.uuidString,
                "active_ms": String(Int((event.activeElapsedSeconds * 1_000).rounded()))
            ]
        )
    }

    private static func diagnosticSummarySink(_ summary: MealReviewSummary) {
        SeeCalDiagnostics.record(
            .notice,
            category: "meal_review",
            name: "review_completed",
            fields: [
                "review_session": summary.sessionID.uuidString,
                "outcome": summary.outcome.rawValue,
                "active_ms": String(Int((summary.activeElapsedSeconds * 1_000).rounded())),
                "actions": String(summary.actionCount),
                "corrections": String(summary.correctionActionCount),
                "keyboard_used": String(summary.usedKeyboard)
            ]
        )
    }
}
