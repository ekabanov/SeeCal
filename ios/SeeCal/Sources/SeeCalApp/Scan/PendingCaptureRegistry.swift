import Foundation

/// Tracks in-flight photo captures so that **exactly one** of the possible
/// outcomes — the AVFoundation delegate callback, the watchdog timeout, or a
/// session runtime error — completes each capture.
///
/// Exactly-once matters because `AVFoundationCaptureService.capturePhoto()`
/// bridges these completions to a `CheckedContinuation`, where both failure
/// modes are severe:
/// - resuming twice **traps** at runtime, and
/// - never resuming hangs the caller forever — which is what a stuck delegate
///   used to do, presenting as a permanently dead shutter with no error.
///
/// Claiming is therefore destructive: the first claimant gets the completion and
/// every later claim gets `nil`.
///
/// This type is deliberately **not** internally synchronised — callers must
/// serialise access. The capture session's serial queue is that isolation, which
/// also satisfies AVFoundation's own threading requirement. Keeping it
/// platform-agnostic (no `#if os(iOS)`, no AVFoundation import) is what makes it
/// unit-testable on the macOS test host, where the real camera type can't build.
struct PendingCaptureRegistry {
    typealias Completion = (Result<Data, Error>) -> Void

    private var pending: [Int64: Completion] = [:]

    /// Number of captures still awaiting an outcome.
    var count: Int { pending.count }

    mutating func register(id: Int64, completion: @escaping Completion) {
        pending[id] = completion
    }

    /// Removes and returns the completion for `id`, or `nil` if it was already
    /// claimed (or never registered).
    mutating func claim(id: Int64) -> Completion? {
        pending.removeValue(forKey: id)
    }

    /// Removes and returns every outstanding completion — used to fail all
    /// in-flight captures at once when the session itself breaks.
    mutating func claimAll() -> [Completion] {
        let all = Array(pending.values)
        pending.removeAll()
        return all
    }
}
