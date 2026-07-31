import Foundation
import XCTest
@testable import SeeCalApp

final class MealReviewMetricsTests: XCTestCase {
    func testRecorderExcludesPausedTimeAndSummarizesCorrectionEffort() throws {
        let sessionID = UUID()
        var clock = Date(timeIntervalSince1970: 1_000)
        var events: [MealReviewEvent] = []
        var summaries: [MealReviewSummary] = []
        let recorder = MealReviewSessionRecorder(
            sessionID: sessionID,
            now: { clock },
            eventSink: { events.append($0) },
            summarySink: { summaries.append($0) }
        )

        recorder.start()
        clock.addTimeInterval(3)
        recorder.record(.itemOpened)
        clock.addTimeInterval(2)
        recorder.record(.replacementSelected)
        recorder.record(.estimateHintOpened)
        recorder.record(.estimateHintSubmitted)
        recorder.record(.keyboardUsed)
        recorder.record(.keyboardUsed)
        recorder.pause()
        clock.addTimeInterval(100)
        recorder.resume()
        clock.addTimeInterval(4)
        recorder.record(.amountChanged)

        let summary = try XCTUnwrap(recorder.finish(.saved))
        XCTAssertEqual(summary.sessionID, sessionID)
        XCTAssertEqual(summary.activeElapsedSeconds, 9, accuracy: 0.0001)
        XCTAssertEqual(summary.actionCount, 5)
        XCTAssertEqual(summary.correctionActionCount, 3)
        XCTAssertTrue(summary.usedKeyboard)
        XCTAssertEqual(summaries, [summary])
        XCTAssertEqual(events.map(\.action), [
            .draftShown,
            .itemOpened,
            .replacementSelected,
            .estimateHintOpened,
            .estimateHintSubmitted,
            .keyboardUsed,
            .amountChanged,
            .saved,
        ])
        XCTAssertNil(recorder.finish(.discarded), "A review must finish exactly once")
    }
}
