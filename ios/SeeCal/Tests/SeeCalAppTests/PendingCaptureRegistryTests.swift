import XCTest
@testable import SeeCalApp

/// The registry's whole job is exactly-once delivery. Getting it wrong is not a
/// cosmetic bug: the completions resume a `CheckedContinuation`, so a double
/// claim would trap and a dropped claim would hang the shutter forever.
final class PendingCaptureRegistryTests: XCTestCase {
    private func recorder() -> (PendingCaptureRegistry.Completion, () -> Int) {
        final class Box { var calls = 0 }
        let box = Box()
        return ({ _ in box.calls += 1 }, { box.calls })
    }

    func testClaimReturnsTheRegisteredCompletionOnce() {
        var registry = PendingCaptureRegistry()
        let (completion, calls) = recorder()
        registry.register(id: 42, completion: completion)
        XCTAssertEqual(registry.count, 1)

        let claimed = registry.claim(id: 42)
        XCTAssertNotNil(claimed, "First claim must hand back the completion")
        claimed?(.success(Data()))
        XCTAssertEqual(calls(), 1)
        XCTAssertEqual(registry.count, 0, "Claiming must remove the entry")
    }

    func testSecondClaimReturnsNil() {
        var registry = PendingCaptureRegistry()
        let (completion, _) = recorder()
        registry.register(id: 7, completion: completion)

        XCTAssertNotNil(registry.claim(id: 7))
        XCTAssertNil(
            registry.claim(id: 7),
            "A second claim must return nil — resuming a continuation twice traps"
        )
    }

    func testClaimOfUnknownIDReturnsNil() {
        var registry = PendingCaptureRegistry()
        XCTAssertNil(registry.claim(id: 999))
    }

    func testClaimAllDrainsEveryOutstandingCapture() {
        var registry = PendingCaptureRegistry()
        let (a, aCalls) = recorder()
        let (b, bCalls) = recorder()
        registry.register(id: 1, completion: a)
        registry.register(id: 2, completion: b)

        let drained = registry.claimAll()
        XCTAssertEqual(drained.count, 2)
        for completion in drained {
            completion(.failure(CaptureServiceError.captureFailed("session died")))
        }
        XCTAssertEqual(aCalls(), 1)
        XCTAssertEqual(bCalls(), 1)
        XCTAssertEqual(registry.count, 0)

        // Crucially, the per-id path must not double-deliver afterwards.
        XCTAssertNil(registry.claim(id: 1))
        XCTAssertNil(registry.claim(id: 2))
    }

    func testTimeoutAndDelegateRaceDeliversExactlyOnce() {
        // Models the real race: the watchdog and the delegate both try to
        // complete the same capture. Whoever claims first wins; the loser is a
        // no-op rather than a second resume.
        var registry = PendingCaptureRegistry()
        let (completion, calls) = recorder()
        registry.register(id: 5, completion: completion)

        let watchdog = registry.claim(id: 5)
        let delegate = registry.claim(id: 5)

        XCTAssertNotNil(watchdog)
        XCTAssertNil(delegate)
        watchdog?(.failure(CaptureServiceError.captureFailed("timed out")))
        delegate?(.success(Data()))
        XCTAssertEqual(calls(), 1, "Exactly one outcome must be delivered")
    }
}
