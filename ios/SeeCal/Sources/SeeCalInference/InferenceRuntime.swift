import Foundation
import SeeCalDiagnostics
import SeeCalDomain

public enum InferenceError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case runtimeUnavailable(String)
    case runtimeFailed(String)
    case parsingFailed(String)
    case allRuntimesFailed([RuntimeFailure])
    /// IDENTIFY succeeded, but deterministic nutrition resolution needs a
    /// person's context. This is a recoverable product state, not a runtime
    /// failure: the scan UI keeps the photo and offers a text-hint retry.
    case humanInputRequired(recognizedItems: [String], unresolvedItems: [String])
    /// The model returned the v7 not-food refusal (`{"not_food": true}`) — the
    /// photo isn't food. A *definitive* answer, not a failure: the orchestrator
    /// surfaces it immediately (no retry, no fallback runtime) and the scan flow
    /// renders a "No food detected" state rather than the error/Retry screen.
    case notFood

    /// A single runtime's failure, captured so callers can surface a real message
    /// instead of a generic "no runtime available" string.
    public struct RuntimeFailure: Equatable, Sendable {
        public let runtimeName: String
        public let errorDescription: String

        public init(runtimeName: String, errorDescription: String) {
            self.runtimeName = runtimeName
            self.errorDescription = errorDescription
        }
    }

    public var description: String {
        switch self {
        case let .runtimeUnavailable(name):
            return "Runtime unavailable: \(name)"
        case let .runtimeFailed(reason):
            return "Runtime failed: \(reason)"
        case let .parsingFailed(reason):
            return "Failed to parse JSON: \(reason)"
        case let .allRuntimesFailed(failures):
            if failures.isEmpty {
                return "No available runtime could produce a valid result"
            }
            let details = failures
                .map { "\($0.runtimeName): \($0.errorDescription)" }
                .joined(separator: "; ")
            return "All runtimes failed — \(details)"
        case let .humanInputRequired(recognizedItems, unresolvedItems):
            let recognized = recognizedItems.joined(separator: ", ")
            let unresolved = unresolvedItems.joined(separator: ", ")
            return "I recognized \(recognized), but need your help matching \(unresolved) to nutrition data."
        case .notFood:
            return "No food detected in the photo"
        }
    }

    public var errorDescription: String? {
        description
    }
}

public protocol InferenceRuntime: Sendable {
    var name: String { get }
    var modelFamily: String { get }
    func isAvailable() async -> Bool
    func infer(request: FoodScanRequest) async throws -> FoodScanResult
}

public actor RuntimeOrchestrator {
    private let runtimes: [InferenceRuntime]
    private let timeoutNanoseconds: UInt64
    private let maxAttemptsPerRuntime: Int

    public init(
        runtimes: [InferenceRuntime],
        timeoutNanoseconds: UInt64 = 8_000_000_000,
        maxAttemptsPerRuntime: Int = 1
    ) {
        self.runtimes = runtimes
        self.timeoutNanoseconds = timeoutNanoseconds
        self.maxAttemptsPerRuntime = max(1, maxAttemptsPerRuntime)
    }

    public func infer(request: FoodScanRequest) async throws -> FoodScanResult {
        var failures: [InferenceError.RuntimeFailure] = []
        let overallStartedAt = DispatchTime.now().uptimeNanoseconds
        SeeCalDiagnostics.record(
            .notice,
            category: "inference",
            name: "orchestration_started",
            fields: [
                "runtime_count": String(runtimes.count),
                "max_attempts": String(maxAttemptsPerRuntime),
                "timeout_ms": String(timeoutNanoseconds / 1_000_000)
            ]
        )

        for runtime in runtimes {
            guard await runtime.isAvailable() else {
                SeeCalDiagnostics.record(
                    .error,
                    category: "inference",
                    name: "runtime_unavailable",
                    fields: ["runtime": runtime.name]
                )
                failures.append(.init(runtimeName: runtime.name, errorDescription: "runtime unavailable"))
                continue
            }

            var lastError: Error?
            for attempt in 0..<maxAttemptsPerRuntime {
                let attemptStartedAt = DispatchTime.now().uptimeNanoseconds
                SeeCalDiagnostics.record(
                    .info,
                    category: "inference",
                    name: "runtime_attempt_started",
                    fields: ["runtime": runtime.name, "attempt": String(attempt + 1)]
                )
                do {
                    let result = try await withTimeout(timeoutNanoseconds) {
                        try await runtime.infer(request: request)
                    }
                    SeeCalDiagnostics.record(
                        .notice,
                        category: "inference",
                        name: "runtime_attempt_succeeded",
                        fields: [
                            "runtime": runtime.name,
                            "attempt": String(attempt + 1),
                            "duration_ms": Self.elapsedMilliseconds(since: attemptStartedAt),
                            "overall_duration_ms": Self.elapsedMilliseconds(since: overallStartedAt)
                        ]
                    )
                    return result
                } catch InferenceError.notFood {
                    // Definitive "this isn't food" — surface it as-is. Retrying or
                    // falling through to another runtime would only re-derive the
                    // same answer (or, worse, let a weaker runtime hallucinate food).
                    SeeCalDiagnostics.record(
                        .notice,
                        category: "inference",
                        name: "runtime_returned_not_food",
                        fields: [
                            "runtime": runtime.name,
                            "attempt": String(attempt + 1),
                            "duration_ms": Self.elapsedMilliseconds(since: attemptStartedAt)
                        ]
                    )
                    throw InferenceError.notFood
                } catch let InferenceError.humanInputRequired(recognized, unresolved) {
                    // A human can resolve this without recapturing the photo.
                    // Preserve the structured state instead of flattening it
                    // into allRuntimesFailed or falling through to another
                    // runtime that would discard the useful identification.
                    SeeCalDiagnostics.record(
                        .notice,
                        category: "inference",
                        name: "runtime_requested_human_input",
                        fields: [
                            "runtime": runtime.name,
                            "attempt": String(attempt + 1),
                            "recognized_count": String(recognized.count),
                            "unresolved_count": String(unresolved.count),
                            "duration_ms": Self.elapsedMilliseconds(since: attemptStartedAt)
                        ]
                    )
                    throw InferenceError.humanInputRequired(
                        recognizedItems: recognized,
                        unresolvedItems: unresolved
                    )
                } catch {
                    lastError = error
                    SeeCalDiagnostics.record(
                        .error,
                        category: "inference",
                        name: "runtime_attempt_failed",
                        fields: [
                            "runtime": runtime.name,
                            "attempt": String(attempt + 1),
                            "duration_ms": Self.elapsedMilliseconds(since: attemptStartedAt)
                        ].merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
                    )
                    continue
                }
            }

            if let lastError {
                failures.append(.init(runtimeName: runtime.name, errorDescription: String(describing: lastError)))
            }
        }

        SeeCalDiagnostics.record(
            .fault,
            category: "inference",
            name: "all_runtimes_failed",
            fields: [
                "failure_count": String(failures.count),
                "overall_duration_ms": Self.elapsedMilliseconds(since: overallStartedAt)
            ]
        )
        throw InferenceError.allRuntimesFailed(failures)
    }

    private static func elapsedMilliseconds(since startedAt: UInt64) -> String {
        String((DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000)
    }
}

private func withTimeout<T: Sendable>(
    _ timeoutNanoseconds: UInt64,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }

        group.addTask {
            try await Task.sleep(nanoseconds: timeoutNanoseconds)
            throw InferenceError.runtimeFailed("Inference timed out after \(Double(timeoutNanoseconds) / 1_000_000_000)s")
        }

        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}
