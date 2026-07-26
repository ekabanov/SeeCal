import Foundation
import SeeCalDomain

public enum InferenceError: Error, Equatable, CustomStringConvertible {
    case runtimeUnavailable(String)
    case runtimeFailed(String)
    case parsingFailed(String)

    public var description: String {
        switch self {
        case let .runtimeUnavailable(name):
            return "Runtime unavailable: \(name)"
        case let .runtimeFailed(reason):
            return "Runtime failed: \(reason)"
        case let .parsingFailed(reason):
            return "Failed to parse JSON: \(reason)"
        }
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
        for runtime in runtimes {
            guard await runtime.isAvailable() else {
                continue
            }

            for _ in 0..<maxAttemptsPerRuntime {
                do {
                    return try await withTimeout(timeoutNanoseconds) {
                        try await runtime.infer(request: request)
                    }
                } catch {
                    continue
                }
            }
        }

        throw InferenceError.runtimeUnavailable("No available runtime could produce a valid result")
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
